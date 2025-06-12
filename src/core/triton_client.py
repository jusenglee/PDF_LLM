import asyncio
import hashlib
import json
import logging
import aiohttp
from aiohttp import ClientError, ClientResponseError, ClientTimeout
from src.utils.cache import ResponseCache

logger = logging.getLogger(__name__)

class OptimizedTritonClient:
    MAX_SERVER_TOKENS = 1024

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
        """최종 토큰 수가 MAX_SERVER_TOKENS 이하가 되도록 user_text를 가운데를 잘라 축약"""
        tok = self.tokenizer
        if not tok:
            return user_text

        # ① 템플릿 '머리·꼬리' 길이(고정분) 미리 계산
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
        if getattr(self, '_session', None) is None or self._session.closed:
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
        """동적 배치 크기 조정과 파이프라이닝을 통한 최적화된 병렬 처리"""
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
                    logger.warning(f"프롬프트 길이 초과: {token_count} 토큰 > {MAX_INPUT_TOKENS} 제한 (자동 축소)")

                    # 지시문과 내용 분리 (지시문은 보존)
                    parts = prompt.split("\n\n", 1)
                    instruction = parts[0] if len(parts) > 1 else ""
                    content = parts[1] if len(parts) > 1 else parts[0]

                    # 지시문 토큰 수 계산
                    instruction_tokens = len(self.tokenizer(instruction, add_special_tokens=False).input_ids)

                    # 내용에 할당할 수 있는 최대 토큰 수 계산
                    available_tokens = MAX_INPUT_TOKENS - instruction_tokens - 5  # 여유 토큰

                    # 내용을 토큰 단위로 자르기
                    encoded_content = self.tokenizer(content, add_special_tokens=False)
                    truncated_content_ids = encoded_content.input_ids[:available_tokens]
                    truncated_content = self.tokenizer.decode(truncated_content_ids)

                    # 지시문과 잘린 내용 재결합
                    if instruction:
                        prompt = f"{instruction}\n\n{truncated_content}"
                    else:
                        prompt = truncated_content

                    # 최종 토큰 수 로깅
                    final_tokens = len(self.tokenizer(prompt, add_special_tokens=False).input_ids)
                    logger.info(f"최종 프롬프트 길이: {final_tokens} 토큰 (축소 후)")
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
