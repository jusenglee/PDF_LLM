"""PDF 문서 ➜ LLM 요약 최적화 파이프라인
------------------------------------------------
• 비동기 HTTP/2 + 스트리밍으로 Triton Inference Server 호출
• 페이지‑병렬 PDF 추출 (fitz)  + 정확한 토큰 기반 청크 분할
• SQLite 기반 응답 캐시로 재호출 최소화
• 안전한 signal shutdown & graceful connection close
 
Python 3.10 이상 권장, pip install -r requirements.txt ( fitz / PyMuPDF , transformers , httpx [h2] …).
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import signal
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional

import fitz  # PyMuPDF
import httpx
from transformers import AutoTokenizer
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import AsyncIterator, Optional

logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────────────────────────
# 1. 토큰 관리자
# ────────────────────────────────────────────────────────────────────

@dataclass
class ChunkInfo:
    text: str
    token_count: int
    sha256: str


class TokenManager:
    """텍스트를 LLM 컨텍스트 길이에 맞게 청크로 나누는 헬퍼."""

    def __init__(self, model_path: str = "./", ctx_len: int = 2048):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        self.max_ctx = ctx_len
        # 생성 토큰·프롬프트 여유분을 감안해 목표 청크 길이 설정
        self.target_chunk = int(ctx_len * 0.88)  # ≈1800 for 2 k ctx

    # FastTokenizer → pure C++ 경로, add_special_tokens False 중요
    def count(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False).input_ids)

    def create_chunks(self, text: str) -> List[ChunkInfo]:
        words = text.split()
        chunks: List[ChunkInfo] = []
        cur_words: List[str] = []
        cur_tok = 0

        for w in words:
            w_tok = self.count(w)
            if cur_tok + w_tok > self.target_chunk:
                chunk_text = " ".join(cur_words)
                chunks.append(self._build_chunk(chunk_text))
                cur_words, cur_tok = [w], w_tok
            else:
                cur_words.append(w)
                cur_tok += w_tok

        if cur_words:
            chunks.append(self._build_chunk(" ".join(cur_words)))
        return chunks

    def _build_chunk(self, txt: str) -> ChunkInfo:
        ids_len = self.count(txt)
        if ids_len > self.max_ctx:
            # 양쪽 청크 모두 처리
            mid = len(txt) // 2
            left = txt[:mid]
            right = txt[mid:]
            left_chunk = self._build_chunk(left)
            right_chunk = self._build_chunk(right)
            return [left_chunk, right_chunk]  # 리스트로 반환
        return ChunkInfo(txt, ids_len, hashlib.sha256(txt.encode()).hexdigest())


# ────────────────────────────────────────────────────────────────────
# 2. PDF 추출기 (페이지‑병렬, thread‑safe)
# ────────────────────────────────────────────────────────────────────

class AsyncPDFProcessor:
    def __init__(self, concurrency: int = 8):
        self.sem = asyncio.Semaphore(concurrency)
        self.doc_cache = {}  # PDF 문서 캐시

    async def extract(self, pdf_path: str) -> str:
        if pdf_path not in self.doc_cache:
            self.doc_cache[pdf_path] = fitz.open(pdf_path)
        doc = self.doc_cache[pdf_path]
        tasks = [self._extract_page(doc, i) for i in range(len(doc))]
        pages = await asyncio.gather(*tasks)
        return "\n".join(pages)

    async def _extract_page(self, doc, page_no: int) -> str:
        async with self.sem:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None,
                                              lambda: doc.load_page(page_no).get_text("text"))

    async def close(self):
        for doc in self.doc_cache.values():
            doc.close()
        self.doc_cache.clear()



# ────────────────────────────────────────────────────────────────────
# 3. SQLite 캐시 (프로세스/스레드 안전)
# ────────────────────────────────────────────────────────────────────

class ResponseCache:
    def __init__(self, 
                 db_path: Path = Path("cache/response_cache.db"),
                 max_size: int = 10000,
                 ttl: timedelta = timedelta(days=7)
    ):
        self.db_path = db_path
        self.max_size = max_size
        self.ttl = ttl
        self.lock = asyncio.Lock()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        # DB 파일 삭제하지 않고 유지
        self._init_db()

    def _init_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.db_path, check_same_thread=False)
        
        with self.db:
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    val TEXT,
                    created_at TIMESTAMP DEFAULT (datetime('now')) NOT NULL
                )
            """)
            # 인덱스 추가
            self.db.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON cache(created_at)
            """)

    async def close(self):
        """데이터베이스 연결을 안전하게 종료"""
        async with self.lock:
            self.db.close()

    async def cleanup(self):
        """오래된 캐시 항목 정리"""
        async with self.lock:
            with self.db:
                # TTL 초과 항목 삭제
                self.db.execute(
                    "DELETE FROM cache WHERE created_at < datetime('now', ?)",
                    (f"-{self.ttl.days} days",)
                )

                # 최대 크기 초과시 오래된 항목부터 삭제
                self.db.execute("""
                    DELETE FROM cache 
                    WHERE key IN (
                        SELECT key FROM cache 
                        ORDER BY created_at DESC 
                        LIMIT -1 OFFSET ?
                    )
                """, (self.max_size,))

    async def get(self, key: str) -> Optional[str]:
        await self.cleanup()  # 캐시 정리
        async with self.lock:
            cur = self.db.execute(
                "SELECT val FROM cache WHERE key=? AND created_at > datetime('now', ?)",
                (key, f"-{self.ttl.days} days")
            )
            row = cur.fetchone()
            return row[0] if row else None

    async def put(self, key: str, val: str):
        await self.cleanup()  # 캐시 정리
        async with self.lock:
            with self.db:
                self.db.execute(
                    "INSERT OR REPLACE INTO cache (key, val, created_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
                    (key, val)
                )


# ────────────────────────────────────────────────────────────────────
# 4. Triton HTTP/2 스트리밍 클라이언트
# ────────────────────────────────────────────────────────────────────

class TritonClient:
    def __init__(self, url: str, batch: int = 4, timeout: float = 120.0):
        self.url = url.rstrip("/")
        self.sem = asyncio.Semaphore(batch)
        self.cache = ResponseCache()
        self.timeout = timeout
        self._session = None
        self.batch = batch  # batch 속성 추가

    @property
    def session(self) -> httpx.AsyncClient:
        if self._session is None:
            self._session = httpx.AsyncClient(
                http2=True,
                timeout=self.timeout,
                limits=httpx.Limits(max_keepalive_connections=self.batch)
            )
        return self._session

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        if self._session:
            await self._session.aclose()
            self._session = None
        await self.cache.close()

    async def generate_stream(self, prompt: str, max_new_tokens: int = 200) -> AsyncIterator[str]:
        try:
            key = hashlib.sha256(f"{prompt}:{max_new_tokens}".encode()).hexdigest()
            
            try:
                cached = await self.cache.get(key)
                if cached is not None:
                    logger.debug(f"캐시 히트: {key[:8]}...")
                    for part in cached.split("\n"):
                        if part:
                            yield part
                    return
            except Exception as e:
                logger.warning(f"캐시 조회 실패: {e}")

            async with self.sem:
                full = []
                for attempt in range(3):
                    try:
                        async with self.session.stream(
                            "POST",
                            self.url,
                            json={"text_input": prompt, "max_tokens": max_new_tokens}
                        ) as r:
                            r.raise_for_status()
                            async for line in r.aiter_lines():
                                if not line or line.isspace():
                                    continue
                                # event: 헤더 처리 추가
                                if line.startswith('event:'):
                                    continue

                                # SSE 형식 처리
                                if line.startswith('data: '):
                                    try:
                                        # 'data: ' 제거 후 JSON 파싱
                                        json_str = line[6:].strip()
                                        data = json.loads(json_str)
                                        text = data.get("text_output", "")
                                        if text:
                                            full.append(text)
                                            yield text
                                    except json.JSONDecodeError as e:
                                        logger.debug(f"응답 데이터: {line!r}")
                                        logger.warning(f"JSON 파싱 실패: {e}")
                                        continue

                            if full:
                                await self.cache.put(key, "\n".join(full))
                            return

                    except (httpx.HTTPError, asyncio.TimeoutError) as e:
                        if attempt == 2:
                            logger.error(f"Triton 서버 오류: {e}")
                            yield "[Triton Error]"
                            return
                        await asyncio.sleep(0.5 * (attempt + 1))

        except Exception as e:
            logger.error(f"예기치 못한 오류: {e}", exc_info=True)
            yield "[System Error]"


# ────────────────────────────────────────────────────────────────────
# 5. 전체 파이프라인
# ────────────────────────────────────────────────────────────────────

class Pipeline:
    def __init__(self, model_path: str = "./", triton_url: str = "http://203.250.238.30:8888/v2/models/ensemble/generate_stream"):
        self.token_mgr = TokenManager(model_path)
        self.pdf_proc = AsyncPDFProcessor()
        self.triton = TritonClient(triton_url)
        
        # logging 설정 추가
        logging.basicConfig(level=logging.WARNING)
        transformers_logger = logging.getLogger("transformers")
        transformers_logger.setLevel(logging.ERROR)  # transformers 경고 억제

    async def process_document(self, pdf_path: str) -> AsyncIterator[Dict]:
        try:
            # 1) PDF → 텍스트
            raw_text = await self.pdf_proc.extract(pdf_path)
            if not raw_text.strip():
                raise ValueError(f"PDF 파일 '{pdf_path}'에서 텍스트를 추출할 수 없습니다.")
            
            # 2) 텍스트 → 청크
            chunks = self.token_mgr.create_chunks(raw_text)
            if not chunks:
                raise ValueError("텍스트를 청크로 분할할 수 없습니다.")

            queue: asyncio.Queue = asyncio.Queue()

            async def _producer(idx: int, ck: ChunkInfo):
                prompt = f"[Chunk {idx+1}/{len(chunks)}]\n{ck.text}"
                try:
                    async for part in self.triton.generate_stream(prompt):
                        if part and isinstance(part, str):
                            await queue.put({
                                "idx": idx,
                                "total": len(chunks),
                                "text": part.strip()  # 공백 제거
                            })
                except Exception as e:
                    logger.error(f"청크 {idx+1} 처리 중 오류 발생: {e}")
                finally:
                    await queue.put("__DONE__")

            producers = [asyncio.create_task(_producer(i, c)) for i, c in enumerate(chunks)]
            done_cnt = 0
            
            while done_cnt < len(producers):
                item = await queue.get()
                if item == "__DONE__":
                    done_cnt += 1
                elif isinstance(item, dict) and item.get("text"):  # 유효한 텍스트만 전달
                    yield item

            await asyncio.gather(*producers)
            
        except Exception as e:
            logger.error(f"문서 처리 중 오류 발생: {e}")
            raise

    async def close(self):
        await self.triton.close()

# ────────────────────────────────────────────────────────────────────
# 6. graceful shutdown helper
# ────────────────────────────────────────────────────────────────────

async def shutdown(sig, loop):
    print(f"\n⬇️ Signal {sig.name} received; shutting down…")
    tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
    for t in tasks:
        t.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    loop.stop()


# ────────────────────────────────────────────────────────────────────
# 7. 실행 예시 (CLI)
# ────────────────────────────────────────────────────────────────────

async def main(pdf_path: str):
    pipeline = Pipeline()
    try:
        async for chunk in pipeline.process_document(pdf_path):
            print(f"▸ [{chunk['idx']+1}/{chunk['total']}] {chunk['text']}")
    finally:
        await pipeline.close()

import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

# 기존 main3.py 모듈 임포트
from main3 import OptimizedPipeline
from semantic_search import SemanticSearchEngine, SEMANTIC_SEARCH_AVAILABLE

# 로거 설정
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("summary_pipeline.log", mode="a")
    ]
)
logger = logging.getLogger(__name__)

class EmbeddingExtractiveAbstractiveSummarizer:
    """임베딩 기반 추출형 사전 필터링 + 추상적 요약 파이프라인"""

    def __init__(
        self,
        model_path: str = "./",
        triton_url: str = "http://203.250.238.30:8888/v2/models/ensemble/generate_stream",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        # 의미 검색 엔진 초기화
        self.semantic_engine = None
        if SEMANTIC_SEARCH_AVAILABLE:
            try:
                self.semantic_engine = SemanticSearchEngine(model_name=embedding_model)
                logger.info(f"의미 검색 엔진 초기화 성공: {embedding_model}")
            except ImportError as e:
                logger.warning(f"의미 검색 엔진 초기화 실패: {e}")
        else:
            logger.warning("sentence-transformers/faiss 라이브러리가 설치되지 않아 의미 검색 비활성화")

        # 파이프라인 초기화 (기존 최적화된 파이프라인 재사용)
        self.pipeline = OptimizedPipeline(model_path, triton_url)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def close(self):
        """리소스 정리"""
        await self.pipeline.close()

    async def extract_and_summarize(self, text: str, target_length: int = 200) -> Dict[str, Any]:
        """텍스트에서 핵심 문장 추출 후 요약 생성"""
        if not text.strip():
            return {
                "error": "처리할 텍스트가 없습니다",
                "success": False
            }

        # 1. 임베딩 기반 핵심 문장 추출
        if self.semantic_engine is not None:
            try:
                logger.info("임베딩 기반 핵심 문장 추출 중...")
                # 핵심 문장 추출 쿼리
                extraction_query = "이 문서의 핵심 내용과 중요 정보를 요약"
                # 관련 문장만 추출 (top_k는 문서 길이에 따라 동적 조정)
                top_k = min(30, max(10, len(text) // 500))
                filtered_text = self.semantic_engine.extract_relevant_context(
                    extraction_query, text, top_k=top_k
                )

                if filtered_text.strip():
                    # 압축률 계산
                    compression_ratio = len(filtered_text) / len(text)
                    logger.info(f"임베딩 필터링: {len(text)} → {len(filtered_text)} 문자 ({compression_ratio:.3f}배)")

                    # 필터링 성공 시 이를 사용
                    extraction_success = True
                    processed_text = filtered_text
                else:
                    # 필터링 실패 시 원본 사용
                    logger.warning("핵심 문장 추출 실패, 원본 텍스트 사용")
                    extraction_success = False
                    processed_text = text
            except Exception as e:
                logger.error(f"임베딩 처리 중 오류: {e}")
                extraction_success = False
                processed_text = text
        else:
            # 의미 검색 엔진 없음
            extraction_success = False
            processed_text = text

        # 2. 요약 생성
        try:
            # 청킹 및 요약
            summary_result = await self.pipeline.summarizer.smart_chunking_summary(
                processed_text, target_length
            )

            # 결과에 추출 정보 추가
            summary_result["extraction_applied"] = extraction_success
            if extraction_success:
                summary_result["extraction_stats"] = {
                    "original_length": len(text),
                    "filtered_length": len(processed_text),
                    "compression_ratio": len(processed_text) / len(text)
                }

            summary_result["success"] = True
            return summary_result

        except Exception as e:
            logger.error(f"요약 생성 중 오류: {e}")
            return {
                "error": f"요약 처리 오류: {e}",
                "success": False,
                "extraction_applied": extraction_success
            }

    async def process_pdf(self, pdf_path: str, target_length: int = 200) -> Dict[str, Any]:
        """PDF 파일 처리 - 기존 파이프라인 활용"""
        try:
            # 파일 존재 확인
            path = Path(pdf_path)
            if not path.exists():
                return {
                    "error": f"PDF 파일이 존재하지 않습니다: {pdf_path}",
                    "success": False
                }

            # 기존 파이프라인 활용 (임베딩 검색 자동 적용)
            result = await self.pipeline.process_document_optimized(
                pdf_path, target_length
            )

            # 결과에 현재 방식 표시
            if result.get("success", False):
                result["approach"] = "Option A - Embedding-based extractive pre-filter + abstractive summary"

            return result

        except Exception as e:
            logger.error(f"PDF 처리 중 오류: {e}")
            return {
                "error": f"PDF 처리 오류: {e}",
                "success": False
            }

async def main(pdf_path: str = "example3.pdf"):
    """메인 함수 - 파이프라인 실행"""
    print(f"📄 PDF 처리 시작: {pdf_path} (임베딩 기반 추출형 + 추상적 요약 방식)")

    async with EmbeddingExtractiveAbstractiveSummarizer() as summarizer:
        try:
            result = await summarizer.process_pdf(pdf_path)

            if result.get("success", False):
                print("\n✅ 최종 요약 --------------------")
                print(result.get("final_summary", "요약 내용이 없습니다."))

                # 추가 정보 출력
                stats = result.get("processing_stats", {})
                if stats:
                    semantic_used = "임베딩 검색 적용됨" if stats.get("semantic_search_used", False) else "임베딩 검색 미적용"
                    print(f"\n📊 처리 통계: 청크 {stats.get('chunks_created', 0)}개, "  
                          f"압축률 {stats.get('compression_ratio', 0):.3f}, {semantic_used}")
            else:
                error_msg = result.get("error", "알 수 없는 오류")
                print(f"❌ 처리 실패: {error_msg}")

        except Exception as e:
            print(f"❌ 예기치 않은 오류 발생: {e}")

if __name__ == "__main__":
    # 라이브러리 로거 설정
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.ERROR)  # transformers 경고 억제

    # 샘플 PDF 처리
    sample_pdf = "example3.pdf"

    # 이벤트 루프 설정 및 실행
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        loop.run_until_complete(main(sample_pdf))
    except KeyboardInterrupt:
        print("\n⬇️ 프로그램 종료 요청됨...")
        # 실행 중인 모든 태스크 정리
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()

        # 취소된 태스크들이 완료될 때까지 대기
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
    finally:
        try:
            loop.close()
        except Exception as e:
            print(f"루프 종료 중 오류: {e}")
        print("✅ 완료")
if __name__ == "__main__":
    sample_pdf = "example3.pdf"

    # Windows에서는 시그널 핸들러 대신 예외 처리 사용
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main(sample_pdf))
    except KeyboardInterrupt:
        print("\n⬇️ 프로그램 종료 요청됨...")
        tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        for t in tasks:
            t.cancel()
        loop.run_until_complete(asyncio.gather(*tasks, return_exceptions=True))
    finally:
        loop.close()
        print("✅ 완료")