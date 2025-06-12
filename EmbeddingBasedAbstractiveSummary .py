import asyncio
import hashlib
import json
import logging
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import fitz
import os, sys, locale
if os.name == "nt":                       # Windows 한정
    locale.setlocale(locale.LC_ALL, "")   # 현재 로캘 유지
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
# ────────────────────────────────────────────────────

# aiohttp 사용으로 변경하여 호환성 문제 해결
import aiohttp
from aiohttp import ClientError, ClientResponseError, ClientTimeout

# 로거 설정
logger = logging.getLogger(__name__)

@dataclass
class TextChunk:
    text: str
    token_count: int
    priority: float = 1.0

    def __post_init__(self):
        if not self.text.strip():
            raise ValueError("빈 텍스트 청크는 허용되지 않습니다")

class ResponseCache:
    def __init__(self, max_size: int = 1000, ttl: int = 3600):
        from collections import OrderedDict
        import time

        self.cache = OrderedDict()  # LRU 캐시로 사용
        self.timestamps = {}        # 타임스탬프 저장
        self.max_size = max_size
        self.ttl = ttl
        self.hits = 0
        self.misses = 0

    async def get(self, key: str) -> Optional[str]:
        import time

        if key in self.cache:
            # monotonic 시계 사용 (시스템 시간 변경에 영향 받지 않음)
            if time.monotonic() - self.timestamps[key] < self.ttl:
                # LRU 업데이트 - 항목 재배치
                value = self.cache.pop(key)
                self.cache[key] = value  # 맨 뒤로 이동
                self.hits += 1
                return value
            else:
                # TTL 만료된 항목 제거
                self.cache.pop(key, None)
                self.timestamps.pop(key, None)

        self.misses += 1
        return None

    async def put(self, key: str, value: str):
        import time

        # LRU 캐시 크기 제한 (삽입 전 확인)
        if len(self.cache) >= self.max_size and key not in self.cache:
            # 가장 오래 사용되지 않은 항목 제거 (OrderedDict의 첫 항목)
            oldest_key, _ = self.cache.popitem(last=False)
            self.timestamps.pop(oldest_key, None)

        # 항목 추가/갱신
        self.cache[key] = value
        self.timestamps[key] = time.monotonic()  # monotonic 시계 사용

    async def close(self):
        self.cache.clear()
        self.timestamps.clear()

    def get_stats(self) -> dict:
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": f"{hit_rate:.3f}",
            "cache_size": len(self.cache)
        }


class AsyncPDFProcessor:
    def __init__(self, concurrency: int = 8):
        self.sem = asyncio.Semaphore(concurrency)

    async def extract(self, pdf_path: str) -> str:
        """
        PDF 전체를 한 번만 연 뒤, 페이지별 텍스트를 순차-스레드로 추출.
        각 worker 는 *자기만의* Document 를 사용하므로 thread-safe.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

        loop = asyncio.get_running_loop()

        # ── 페이지별 작업 함수 ─────────────────────────────
        def _page_text(idx: int) -> str:
            # 각 스레드가 독립적으로 문서를 열고 닫는다 → 안전
            with fitz.open(pdf_path) as doc:
                return doc.load_page(idx).get_text("text").strip()

        # 한번 열어서 페이지 수만 알아낸다
        with fitz.open(pdf_path) as doc:
            page_count = len(doc)
        if page_count == 0:
            raise ValueError("PDF 파일에 페이지가 없습니다")

        # 세마포어로 동시 실행 제한
        async def _run(idx: int) -> str:
            async with self.sem:
                return await loop.run_in_executor(None, _page_text, idx)

        # 병렬 실행
        pages = await asyncio.gather(*[_run(i) for i in range(page_count)],
                                     return_exceptions=True)

        # 유효 페이지 수집
        valid_pages = []
        for i, page in enumerate(pages):
            if isinstance(page, Exception):
                logger.error(f"[PDF] 페이지 {i} 처리 실패: {page}")
            elif page:
                valid_pages.append(page)

        if not valid_pages:
            raise ValueError("PDF 파일에서 텍스트를 추출할 수 없습니다")

        return "\n\n".join(valid_pages)

    async def close(self):
        """리소스 정리 - 이제 각 작업자가 독립적으로 문서를 닫으므로 필요 없음"""

class AdaptiveTokenManager:
    def __init__(self, model_path: str = "./", ctx_len: int = 2048):
        self.ctx_len = ctx_len
        self.model_path = model_path

    def count_tokens(self, text: str) -> int:
        """간단한 토큰 카운팅 (빈 문자열 검증 추가)"""
        if not text or not text.strip():
            # 빈 텍스트는 0토큰으로 처리 (에러 방지용)
            return 0
        # 최소 1토큰 이상 보장
        return max(1, int(len(text.split()) * 1.3))


    def create_adaptive_chunks(
        self, 
        text: str, 
        target_summary_length: int = 200
    ) -> Tuple[List[TextChunk], Dict[str, Any]]:
        """적응적 청킹 알고리즘"""
        
        total_tokens = self.count_tokens(text)
        if total_tokens <= self.ctx_len * 0.7:
            return [TextChunk(text, int(total_tokens))], {"strategy": "single_chunk"}
        
        # 문단 기반 분할
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        if not paragraphs:
            paragraphs = [text]
        
        chunks = []
        current_chunk = ""
        current_tokens = 0
        max_chunk_tokens = int(self.ctx_len * 0.6)
        
        for para in paragraphs:
            para_tokens = self.count_tokens(para)
            
            if current_tokens + para_tokens <= max_chunk_tokens:
                current_chunk += para + "\n\n"
                current_tokens += para_tokens
            else:
                if current_chunk:
                    chunks.append(TextChunk(current_chunk.strip(), int(current_tokens)))
                
                if para_tokens > max_chunk_tokens:
                    # 긴 문단은 문장 단위로 분할
                    sentences = re.split(r'[.!?]+', para)
                    temp_chunk = ""
                    temp_tokens = 0
                    
                    for sent in sentences:
                        if not sent.strip():
                            continue
                        sent_tokens = self.count_tokens(sent)
                        
                        if temp_tokens + sent_tokens <= max_chunk_tokens:
                            temp_chunk += sent + ". "
                            temp_tokens += sent_tokens
                        else:
                            if temp_chunk:
                                chunks.append(TextChunk(temp_chunk.strip(), int(temp_tokens)))
                            temp_chunk = sent + ". "
                            temp_tokens = sent_tokens
                    
                    if temp_chunk:
                        current_chunk = temp_chunk
                        current_tokens = temp_tokens
                    else:
                        current_chunk = ""
                        current_tokens = 0
                else:
                    current_chunk = para + "\n\n"
                    current_tokens = para_tokens
        
        if current_chunk:
            chunks.append(TextChunk(current_chunk.strip(), int(current_tokens)))
        
        # 우선순위 계산
        for i, chunk in enumerate(chunks):
            chunk.priority = 1.0 + (0.1 * (len(chunks) - i))
        
        allocation = {
            "strategy": "adaptive",
            "total_chunks": len(chunks),
            "avg_chunk_size": sum(c.token_count for c in chunks) / len(chunks),
            "max_chunk_size": max(c.token_count for c in chunks),
            "min_chunk_size": min(c.token_count for c in chunks)
        }
        
        return chunks, allocation

class OptimizedTritonClient:
    MAX_SERVER_TOKENS = 2047

    def __init__(self, url: str, batch: int = 16, timeout: float = 60.0, max_connections: int = 32, tokenizer=None):
        self.url = url.rstrip("/")
        self.sem = asyncio.Semaphore(batch)
        self.cache = ResponseCache()
        self.timeout = timeout
        self.batch = batch
        self.max_connections = max_connections
        self._session = None
        self.tokenizer = tokenizer

    def _trim_for_server(self, user_text: str) -> str:
        """
        ✔  chat_template 가 붙은 **최종** 토큰 수가 2 047 이하가 되도록
           user_text(=서버에 보낼 query) 를 가운데를 잘라 축약.
        """
        tok = self.tokenizer
        if not tok:
            return user_text

        # ① 템플릿 ‘머리·꼬리’ 길이(고정분) 미리 계산
        empty_tpl = tok.apply_chat_template(
            [{"role": "user", "content": ""}],
            tokenize=False, add_generation_prompt=True
        )
        head_tail_tokens = len(tok.encode(empty_tpl, add_special_tokens=False))

        # ② user_text 를 토큰화
        ids_user = tok.encode(user_text, add_special_tokens=False)
        max_user_tokens = self.MAX_SERVER_TOKENS - head_tail_tokens - 2   # 2-토큰 여유

        if len(ids_user) <= max_user_tokens:
            return user_text                        # 이미 안전

        # ③ 가운데 잘라내기
        keep_front = max_user_tokens // 2
        keep_back  = max_user_tokens - keep_front
        new_ids = ids_user[:keep_front] + ids_user[-keep_back:]

        return tok.decode(new_ids, skip_special_tokens=False)

    @property
    def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(
                limit=self.max_connections,
                limit_per_host=self.max_connections,
                keepalive_timeout=30.0,
                enable_cleanup_closed=True
            )
            timeout = aiohttp.ClientTimeout(total=self.timeout, connect=10.0)
            self._session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                raise_for_status=True  # 자동으로 HTTP 오류 발생
            )
        return self._session

    async def __aenter__(self):
        """컨텍스트 매니저 진입 - 세션 초기화 보장"""
        # 세션이 닫혔거나 없으면 새로 생성 (속성 접근자 활용)
        if getattr(self, '_session', None) is None or self._session.is_closed:
            _ = self.session  # 세션 초기화 트리거
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """컨텍스트 매니저 종료 - 안전한 리소스 정리"""
        await self.close()

    async def close(self):
        """안전한 세션 종료"""
        try:
            if self._session and not self._session.closed:
                await self._session.close()
                # 세션이 완전히 정리될 때까지 잠시 대기
                await asyncio.sleep(0.1)
        except Exception as e:
            logger.debug(f"세션 종료 중 오류 (무시됨): {e}")
        finally:
            self._session = None
        
        try:
            await self.cache.close()
        except Exception as e:
            logger.debug(f"캐시 종료 중 오류 (무시됨): {e}")

    async def generate_parallel_optimized(
        self,
        prompts: list[str],
        *,
        max_new_tokens: int = 160,
        dynamic_batch_size: bool = True
    ) -> list[str]:
        """
        동적 배치 크기 조정과 파이프라이닝을 통한 최적화된 병렬 처리
        """
        if not prompts:
            return []

        # 동적 배치 크기 계산
        if dynamic_batch_size:
            avg_prompt_length = sum(len(p) for p in prompts) / len(prompts)
            if avg_prompt_length < 500:
                effective_batch = min(self.batch * 2, 32)  # 짧은 프롬프트는 더 많이
            else:
                effective_batch = max(self.batch // 2, 4)   # 긴 프롬프트는 적게
        else:
            effective_batch = self.batch

        # 파이프라인 처리를 위한 청크 분할
        chunks = [prompts[i:i + effective_batch] for i in range(0, len(prompts), effective_batch)]
        results = [None] * len(prompts)

        async def process_chunk(chunk_prompts: list[str], start_idx: int):
            tasks = []
            for i, prompt in enumerate(chunk_prompts):
                # 각 프롬프트를 독립적으로 처리하는 태스크 생성
                task = asyncio.create_task(
                    self._generate_single_cached(prompt, max_new_tokens),
                    name=f"prompt_{start_idx + i}"
                )
                tasks.append(task)

            # 모든 태스크 완료 대기
            chunk_results = await asyncio.gather(*tasks, return_exceptions=True)

            # 결과를 올바른 위치에 배치
            for i, result in enumerate(chunk_results):
                if isinstance(result, Exception):
                    logger.error(f"프롬프트 {start_idx + i} 처리 실패: {result}")
                    results[start_idx + i] = "[Error]"
                else:
                    results[start_idx + i] = result

        # 청크들을 순차적으로 처리 (최대 동시 처리 수 제한)
        sem = asyncio.Semaphore(3)  # 최대 3개 청크 동시 처리

        async def process_chunk_with_limit(chunk, idx):
            async with sem:
                await process_chunk(chunk, idx * effective_batch)

        # 모든 청크 병렬 처리 시작 (세마포어로 제한)
        await asyncio.gather(*[
            process_chunk_with_limit(chunk, i) 
            for i, chunk in enumerate(chunks)
        ])

        return [r for r in results if r is not None]

    async def _generate_single_cached(self, prompt: str, max_new_tokens: int) -> str:
        """캐시를 활용한 단일 프롬프트 처리 - aiohttp 스트리밍 API 호출 방식"""
        # TensorRT-LLM 최대 입력 제한 처리 (2047 토큰)
        MAX_INPUT_TOKENS = 2047
        prompt = self._trim_for_server(prompt)
        # 토크나이저가 있을 경우에만 길이 제한 적용
        if self.tokenizer:
            try:
                token_count = len(self.tokenizer(prompt, add_special_tokens=False).input_ids)

                if token_count > MAX_INPUT_TOKENS:
                    logger.warning(f"프롬프트 길이 초과: {token_count} 토큰 > {MAX_INPUT_TOKENS} 제한 (잘라냄)")
                    # 토큰 단위로 자르기 위해 토크나이저 사용
                    encoded = self.tokenizer(prompt, add_special_tokens=False)
                    truncated_ids = encoded.input_ids[:MAX_INPUT_TOKENS]
                    prompt = self.tokenizer.decode(truncated_ids)
            except Exception as e:
                logger.warning(f"토큰 길이 계산 중 오류 (무시됨): {e}")

        key = hashlib.sha256(f"{prompt}:{max_new_tokens}".encode()).hexdigest()

        # 캐시 확인
        try:
            cached = await self.cache.get(key)
            if cached is not None:
                return cached
        except Exception as e:
            logger.debug(f"캐시 조회 실패: {e}")

        # 실제 스트리밍 API 호출
        async with self.sem:
            full_text = []

            # 3번까지 재시도
            for attempt in range(3):
                try:
                    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
                    data = {"text_input": prompt, "max_tokens": max_new_tokens}

                    async with self.session.post(
                        self.url,
                        json=data,
                        headers=headers,
                        timeout=self.timeout
                    ) as response:
                        response.raise_for_status()

                        # 스트리밍 응답 처리 (chunked)
                        async for chunk in response.content.iter_chunked(1024):
                            chunk_text = chunk.decode("utf-8").strip()

                            # 줄 단위로 분할하여 처리
                            for line in chunk_text.split('\n'):
                                if not line or line.isspace():
                                    continue

                                # SSE 형식 처리
                                if line.startswith("event:"):
                                    continue
                                if line.startswith("data: "):
                                    try:
                                        json_str = line[6:].strip()
                                        data = json.loads(json_str)
                                        text = data.get("text_output", "")
                                        if text:
                                            full_text.append(text)
                                    except json.JSONDecodeError as e:
                                        # 더 자세한 오류 정보 로깅
                                        logger.debug(f"응답 데이터 파싱 실패: {line!r}")
                                        logger.debug(f"JSONDecodeError: {e}, 위치: {e.pos}, 라인: {e.lineno}, 열: {e.colno}")
                                        continue
                                    except KeyError as e:
                                        logger.debug(f"응답 데이터에서 필요한 키를 찾을 수 없음: {e}")
                                        continue
                                    except Exception as e:
                                        logger.debug(f"응답 처리 중 예상치 못한 오류: {type(e).__name__}: {e}")
                                        continue

                    # 응답 결합
                    result = "".join(full_text)
                    if result:
                        await self.cache.put(key, result)
                        return result
                    else:
                        # 빈 응답 처리
                        raise ValueError("Triton 서버가 빈 응답을 반환했습니다")

                except Exception as e:
                    error_msg = str(e)
                    if "Prompt length" in error_msg and "exceeds maximum input length" in error_msg:
                        logger.error(f"입력 길이 초과 오류 (서버 측): {e}")
                        return "[오류: 입력 길이가 서버 제한을 초과합니다. 텍스트가 자동으로 잘렸지만 서버 측에서 거부했습니다.]"

                    if attempt == 2:  # 마지막 시도
                        logger.error(f"Triton 서버 호출 실패 (최종): {e}")
                        return f"[추론 오류: {str(e)[:50]}]" if len(str(e)) > 50 else f"[추론 오류: {e}]"

                    # 재시도 전 대기
                    logger.warning(f"Triton 서버 호출 실패 (재시도 {attempt+1}/3): {e}")
                    await asyncio.sleep(0.5 * (attempt + 1))

            # 재시도 실패 처리
            return "[Triton 서버 응답 오류]"

class HierarchicalSummarizer:
    def __init__(self, token_mgr: AdaptiveTokenManager, triton_client: OptimizedTritonClient):
        self.token_mgr = token_mgr
        self.triton = triton_client
        
    async def smart_chunking_summary(
        self, 
        text: str, 
        target_length: int = 200
    ) -> Dict[str, Any]:
        """스마트 청킹 기반 계층적 요약"""
        
        chunks, allocation = self.token_mgr.create_adaptive_chunks(text, target_length)
        
        if len(chunks) == 1:
            # 단일 청크는 직접 요약
            prompt = f"다음 텍스트를 {target_length}자 이내로 요약해주세요:\n\n{chunks[0].text}"
            # 토큰 길이 확인 (디버깅용, 토크나이저가 있는 경우만)
            if hasattr(self.triton, 'tokenizer') and self.triton.tokenizer:
                try:
                    token_count = len(self.triton.tokenizer(prompt, add_special_tokens=False).input_ids)
                    if token_count > 2000:
                        logger.warning(f"단일 청크 프롬프트 길이: {token_count} 토큰 (자동 잘림 예정)")
                except Exception:
                    pass  # 토크나이저 오류는 무시

            summary = await self.triton._generate_single_cached(prompt, target_length)
            
            return {
                "chunk_summaries": [summary],
                "final_summary": summary,
                "processing_method": "direct"
            }
        
        # 다중 청크 처리
        chunk_prompts = []
        chars_per_chunk = target_length
        for i, chunk in enumerate(chunks):
            prompt = f"다음 텍스트의 핵심 내용을 {chars_per_chunk}자 이내로 요약해주세요 (파트 {i+1}/{len(chunks)}):\n\n{chunk.text}"
            chunk_prompts.append(prompt)

        # 병렬 요약 생성 - 문자 수를 토큰 수로 변환 (평균 1.5배)
        approx_tokens_per_chunk = max(40, int(chars_per_chunk * 0.67))
        chunk_summaries = await self.triton.generate_parallel_optimized(
            chunk_prompts, 
            max_new_tokens=max(20, approx_tokens_per_chunk)  # 최소 20토큰 보장
        )
        
        # 최종 통합 요약
        combined_text = "\n\n".join([s for s in chunk_summaries if s and not s.startswith("[오류")])  # 오류 응답 필터링

        if not combined_text.strip():
            logger.warning("모든 청크 요약이 실패했습니다. 직접 요약을 시도합니다.")
            # 모든 청크 요약이 실패한 경우 원본 텍스트에서 간단한 요약 추출
            short_text = text[:min(len(text), 2000)]  # 원본 텍스트 앞부분만 사용
            final_prompt = f"다음 텍스트를 {target_length}자 이내로 간결하게 요약해주세요:\n\n{short_text}"
        else:
            final_prompt = f"다음 부분별 요약들을 종합하여 {target_length}자 이내의 최종 요약을 작성해주세요:\n\n{combined_text}"

        # 최종 요약에 충분한 토큰 할당 (한글 문자:토큰 비율 고려)
        approx_tokens = int(target_length * 0.8)  # 여유 있게 토큰 할당 (0.67 → 0.8)
        final_summary = await self.triton._generate_single_cached(final_prompt, max(60, approx_tokens))
        
        return {
            "chunk_summaries": chunk_summaries,
            "final_summary": final_summary,
            "processing_method": "hierarchical"
        }

class OptimizedPipeline:
    def __init__(
        self,
        model_path: str = "./",
        triton_url: str = "http://203.250.238.30:8888/v2/models/ensemble/generate_stream",
        ctx_len: int = 2048
    ):
        self.token_mgr = AdaptiveTokenManager(model_path, ctx_len)
        self.pdf_proc = AsyncPDFProcessor()

        # 토크나이저 가져오기
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        # 토크나이저 주입
        self.triton = OptimizedTritonClient(triton_url, batch=16, tokenizer=tokenizer)
        self.summarizer = HierarchicalSummarizer(self.token_mgr, self.triton)


    async def __aenter__(self):
        """비동기 컨텍스트 관리자 진입점"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """비동기 컨텍스트 관리자 종료 - 자원 정리"""
        await self.close()

    async def process_document_optimized(
        self,
        pdf_path: str,
        target_summary_length: int = 200
    ) -> dict:
        """최적화된 문서 처리 파이프라인"""

        try:
            # 1. PDF 추출
            try:
                raw_text = await self.pdf_proc.extract(pdf_path)
                if not raw_text.strip():
                    logger.warning(f"PDF에서 텍스트가 추출되지 않았습니다: {pdf_path}")
                    print(f"⚠️ PDF에서 텍스트가 추출되지 않았습니다: {pdf_path}")
                    return {
                        "warning": "PDF에 추출 가능한 텍스트가 없습니다",
                        "success": False
                    }
            except Exception as e:
                logger.error(f"PDF 추출 실패: {e}")
                print(f"❌ PDF 처리 오류: {e}")
                return {
                    "error": f"PDF 처리 오류: {e}",
                    "success": False
                }

            # 2. 텍스트 전처리 
            clean_text = raw_text.strip()
            if not clean_text:
                raise ValueError("처리할 텍스트가 없습니다")

            # 3. 적응적 청킹
            try:
                chunks, allocation = self.token_mgr.create_adaptive_chunks(
                    clean_text, target_summary_length
                )
            except Exception as e:
                logger.error(f"텍스트 청킹 실패: {e}")
                print(f"❌ 청킹 오류: {e}")
                return {
                    "error": f"텍스트 청킹 오류: {e}",
                    "success": False
                }

            # 4. 요약 처리
            try:
                result = await self.summarizer.smart_chunking_summary(
                    clean_text, target_summary_length
                )
            except Exception as e:
                logger.error(f"요약 처리 실패: {e}")
                print(f"❌ 요약 오류: {e}")
                return {
                    "error": f"요약 처리 오류: {e}",
                    "success": False
                }

            # 5. 성공 결과 반환
            result.update({
                "token_allocation": allocation,
                "success": True,
                "processing_stats": {
                    "chunks_created": len(chunks),
                    "avg_chunk_size": sum(c.token_count for c in chunks) / len(chunks),
                    "compression_ratio": len(result.get("final_summary", "")) / len(clean_text)
                }
            })

            return result

        except Exception as e:
            logger.error(f"문서 처리 중 오류: {e}")
            print(f"❌ 오류 발생: {e}")
            return {
                "error": str(e),
                "success": False
            }

    async def close(self):
        """안전한 리소스 정리"""
        try:
            await self.triton.close()
        except Exception as e:
            logger.debug(f"Triton 클라이언트 종료 중 오류 (무시됨): {e}")
        
        try:
            await self.pdf_proc.close()
        except Exception as e:
            logger.debug(f"PDF 프로세서 종료 중 오류 (무시됨): {e}")

async def main(pdf_path: str):

    # 컨텍스트 관리자 패턴으로 파이프라인 사용
    # (리소스는 자동으로 정리됨)
    async with OptimizedPipeline() as pipeline:
        try:
            # 페이지 수 확인
            temp_doc = None
            try:
                temp_doc = fitz.open(pdf_path)
                page_count = len(temp_doc)
                logger.info(f"PDF 처리 시작: {pdf_path}, 총 {page_count}페이지")
                print(f"📄 PDF 처리 시작: {pdf_path}, 총 {page_count}페이지")
            except Exception as e:
                logger.error(f"PDF 페이지 수 확인 실패: {e}")
                print(f"📄 PDF 처리 시작: {pdf_path}")
            finally:
                if temp_doc:
                    temp_doc.close()

            # 파일 존재 확인
            if not os.path.exists(pdf_path):
                error_msg = f"PDF 파일이 존재하지 않습니다: {pdf_path}"
                logger.error(error_msg)
                print(f"❌ {error_msg}")
                return

            result = await pipeline.process_document_optimized(pdf_path)

            if result.get("success", False):
                logger.info(f"PDF 처리 성공: {pdf_path}")
                print("▷ 파트별 요약 --------------------")
                for i, s in enumerate(result.get("chunk_summaries", []), 1):
                    print(f"[{i}] {s}")

                print("\n✅ 최종 요약 --------------------")
                print(result.get("final_summary", "요약 내용이 없습니다."))

                stats = result.get('processing_stats', {})
                if stats:
                    print(f"\n📊 처리 통계: 청크 {stats.get('chunks_created', 0)}개, "
                        f"압축률 {stats.get('compression_ratio', 0):.3f}")
                    logger.info(f"처리 통계: {stats}")
            else:
                error_msg = result.get('error', '알 수 없는 오류')
                logger.error(f"PDF 처리 실패: {error_msg}")
                print(f"❌ 처리 실패: {error_msg}")
        except Exception as e:
            logger.exception(f"예기치 않은 오류 발생: {e}")
            print(f"❌ 예기치 않은 오류 발생: {e}")

if __name__ == "__main__":
    # 프로그램 시작점에서 로깅 설정 한 번만 초기화
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("pdf_pipeline.log", mode="a")
        ]
    )

    # 라이브러리 로거 설정
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.ERROR)  # transformers 경고 억제

    sample_pdf = "example3.pdf"

    # Windows에서 안전한 이벤트 루프 처리
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