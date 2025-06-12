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