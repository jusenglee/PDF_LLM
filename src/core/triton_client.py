import asyncio
import hashlib
import json
import logging
import aiohttp
from aiohttp import ClientError, ClientResponseError, ClientTimeout
from src.utils.cache import ResponseCache

logger = logging.getLogger(__name__)

class OptimizedTritonClient:
    MAX_SERVER_TOKENS = 2048

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

        async def process_chunk(chunk_prompts: list[str], start_idx: int, log_first_prompt: bool = False):
            tasks = []
            for i, prompt in enumerate(chunk_prompts):
                # 첫 번째 프롬프트는 토큰 정보 로깅 활성화 (선택적)
                log_this_prompt = log_first_prompt and i == 0 and start_idx == 0

                # 각 프롬프트를 독립적으로 처리하는 태스크 생성
                task = asyncio.create_task(
                    self._generate_single_cached(prompt, max_new_tokens, log_tokens=log_this_prompt, log_prompt=log_this_prompt),
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
            chunk_num = idx + 1
            total_chunks = len(chunks)
            print(f"  📊 청크 그룹 {chunk_num}/{total_chunks} 처리 중... ({len(chunk)}/{len(prompts)} 프롬프트)")
            async with sem:
                # 첫 번째 청크의 첫 번째 프롬프트만 로깅 (idx == 0)
                await process_chunk(chunk, idx * effective_batch, log_first_prompt=(idx == 0))
            print(f"  ✓ 청크 그룹 {chunk_num}/{total_chunks} 완료")

        # 모든 청크 병렬 처리 시작 (세마포어로 제한)
        await asyncio.gather(*[
            process_chunk_with_limit(chunk, i) 
            for i, chunk in enumerate(chunks)
        ])

        return [r for r in results if r is not None]

    def _get_cache_key(self, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> str:
        """프롬프트와 파라미터를 기반으로 캐시 키 생성"""
        # 콜백 함수가 전달된 경우 기본값으로 대체
        if callable(max_new_tokens):
            max_new_tokens = 200  # 안전한 기본값

        prompt_hash = hashlib.sha256(prompt.encode('utf-8')).hexdigest()[:16]
        params_str = f"{max_new_tokens}_{temperature:.2f}_{top_p:.2f}"
        return f"{prompt_hash}_{params_str}"

    def _log_prompt_to_file(self, prompt: str, prefix: str = "prompt"):
        """프롬프트 내용을 파일에 로깅"""
        try:
            import os
            import time
            from pathlib import Path
            from config.settings import LOG_DIR

            # 로그 디렉토리 확인
            prompt_log_dir = LOG_DIR / "prompts"
            os.makedirs(prompt_log_dir, exist_ok=True)

            # 타임스탬프로 로그 파일명 생성
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            log_file = prompt_log_dir / f"{prefix}_{timestamp}.log"

            # 프롬프트 저장
            with open(log_file, "w", encoding="utf-8") as f:
                f.write(prompt)

            logger.debug(f"프롬프트 로그 저장됨: {log_file}")
            return str(log_file)
        except Exception as e:
            logger.warning(f"프롬프트 로깅 실패: {e}")
            return None

    async def _generate_single_cached(self, prompt: str, max_new_tokens: int, log_tokens: bool = False, log_prompt: bool = False) -> str:
        """캐시를 활용한 단일 프롬프트 처리 - aiohttp 스트리밍 API 호출 방식"""
        # TensorRT-LLM 최대 입력 제한 처리 (2047 토큰)
        MAX_INPUT_TOKENS = 2048
        TOTAL_MODEL_CONTEXT = 4096  # 모델의 총 컨텍스트 길이

        # 안전 마진 (토큰 계산의 오차를 고려) - 여유 있게 설정
        SAFETY_MARGIN = 100
        # 출력 토큰 최대 한계 (한계 초과 방지)
        MAX_OUTPUT_TOKENS = 900

        # 토크나이저가 있을 경우에만 길이 제한 적용
        if self.tokenizer:
            try:
                # 원본 토큰 수 확인
                original_token_count = len(self.tokenizer(prompt, add_special_tokens=False).input_ids)

                # 프롬프트 축소 전에 로깅 옵션 활성화된 경우 원본 토큰 정보 출력
                if log_tokens and original_token_count > MAX_INPUT_TOKENS:
                    print(f"\n📊 프롬프트 토큰 정보 (축소 전):")
                    print(f"  • 원본 입력 토큰 수: {original_token_count}")
                    print(f"  • 최대 허용 토큰: {MAX_INPUT_TOKENS}")
                    print(f"  ⚠️ 토큰 초과: {original_token_count - MAX_INPUT_TOKENS}토큰 초과로 자동 축소됨")

                # 프롬프트 길이 제한 적용
                prompt = self._trim_for_server(prompt)

                # 축소 후 토큰 수 다시 계산
                token_count = len(self.tokenizer(prompt, add_special_tokens=False).input_ids)

                # 토큰 수 로깅 옵션이 활성화된 경우 토큰 정보 출력
                # 총 토큰이 모델 한계를 넘지 않도록 자동 조정
                total_possible_tokens = TOTAL_MODEL_CONTEXT - SAFETY_MARGIN
                available_tokens = total_possible_tokens - token_count

                # 요청 토큰이 가용 토큰을 초과하면 자동 조정
                # callable 처리 추가
                tokens_to_check = max_new_tokens
                if callable(max_new_tokens):
                    try:
                        # 임의의 인덱스로 평가 (실제 값이 필요한 경우)
                        tokens_to_check = max_new_tokens(0)
                    except Exception as e:
                        logger.warning(f"토큰 길이 계산 중 오류 (무시됨): {e}")
                        tokens_to_check = 200  # 안전한 기본값

                if token_count + tokens_to_check > total_possible_tokens:
                    adjusted_tokens = max(100, min(available_tokens, MAX_OUTPUT_TOKENS))  # 최소 100토큰, 최대 MAX_OUTPUT_TOKENS 보장
                    logger.info(f"토큰 자동 조정: {tokens_to_check} → {adjusted_tokens} (입력: {token_count}, 가용: {available_tokens})")
                    max_new_tokens = adjusted_tokens
                # 요청이 최대 출력 한계를 초과하는 경우에도 제한 적용
                elif not callable(max_new_tokens) and max_new_tokens > MAX_OUTPUT_TOKENS:
                    logger.info(f"최대 토큰 제한 적용: {max_new_tokens} → {MAX_OUTPUT_TOKENS}")
                    max_new_tokens = MAX_OUTPUT_TOKENS

                if log_tokens:
                    print(f"\n📊 프롬프트 토큰 정보:")
                    print(f"  • 입력 토큰 수: {token_count}")
                    print(f"  • 최대 허용 컨텍스트: {TOTAL_MODEL_CONTEXT}")
                    print(f"  • 안전 컨텍스트 한계: {total_possible_tokens}")
                    print(f"  • 응답 가능 토큰: {available_tokens}")
                    print(f"  • 요청 응답 토큰: {max_new_tokens}")

                    # 토큰 자동 조정 정보 표시
                    if token_count + max_new_tokens > TOTAL_MODEL_CONTEXT:
                        print(f"  ⚠️ 주의: 총 토큰({token_count + max_new_tokens})이 모델 한계({TOTAL_MODEL_CONTEXT})를 초과합니다.")
                        print(f"  🔄 응답 토큰이 {max_new_tokens}개로 자동 조정되었습니다.")

                    # 프롬프트 로깅
                    log_file = self._log_prompt_to_file(prompt, f"prompt_{token_count}tokens")
                    if log_file:
                        print(f"  📝 프롬프트 로그: {log_file}")

                if token_count > MAX_INPUT_TOKENS:
                    logger.warning(f"프롬프트 길이 초과: {token_count} 토큰 > {MAX_INPUT_TOKENS} 제한 (자동 축소 실패)")

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

                    # API 호출 전에 모델 최대 컨텍스트 초과 여부 최종 확인
                    if self.tokenizer:
                        try:
                            input_tokens = len(self.tokenizer(prompt, add_special_tokens=False).input_ids)
                            if input_tokens + max_new_tokens > TOTAL_MODEL_CONTEXT - SAFETY_MARGIN:
                                # 컨텍스트 길이를 넘지 않도록 출력 토큰 조정
                                safe_tokens = max(50, min(TOTAL_MODEL_CONTEXT - SAFETY_MARGIN - input_tokens, MAX_OUTPUT_TOKENS))
                                logger.warning(f"API 호출 전 토큰 안전 조정: {max_new_tokens} → {safe_tokens} (입력: {input_tokens})")
                                data["max_tokens"] = safe_tokens  # API 요청 데이터 수정
                        except Exception as e:
                            logger.warning(f"토큰 안전 조정 실패 (무시): {e}")

                    async with self.session.post(
                        self.url,
                        json=data,
                        headers=headers,
                        timeout=self.timeout
                    ) as response:
                        response.raise_for_status()

                        # 스트리밍 응답 처리 개선 - 청크 크기 증가 및 버퍼 보강
                        buffer = ""
                        last_incomplete = ""
                        try:
                            # 청크 크기를 늘려 스트리밍 안정성 향상
                            async for chunk in response.content.iter_chunked(4096):
                                try:
                                    chunk_text = chunk.decode("utf-8")
                                    # 이전에 처리하지 못한 불완전 라인과 현재 청크 결합
                                    buffer = last_incomplete + chunk_text
                                    lines = buffer.split('\n')
                                    # 마지막 라인이 불완전할 수 있으므로 따로 저장
                                    last_incomplete = lines.pop() if lines else ""

                                    # 완전한 라인들 처리
                                    for line in lines:
                                        line = line.strip()
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
                                                # 불완전한 JSON일 가능성 - 로깅만 하고 계속 진행
                                                logger.debug(f"응답 데이터 파싱 실패 (계속 진행): {line!r}")
                                                logger.debug(f"JSONDecodeError: {e}, 위치: {e.pos}")
                                                continue
                                            except KeyError as e:
                                                logger.debug(f"응답 데이터에서 필요한 키를 찾을 수 없음: {e}")
                                                continue
                                            except Exception as e:
                                                logger.debug(f"응답 처리 중 예상치 못한 오류: {type(e).__name__}: {e}")
                                                continue
                                except UnicodeDecodeError as ude:
                                    logger.warning(f"유니코드 디코딩 오류 (무시): {ude}")
                                    continue

                            # 스트림 종료 후 마지막 불완전 라인 처리
                            if last_incomplete.strip():
                                if last_incomplete.startswith("data: "):
                                    try:
                                        json_str = last_incomplete[6:].strip()
                                        data = json.loads(json_str)
                                        text = data.get("text_output", "")
                                        if text:
                                            full_text.append(text)
                                    except Exception:
                                        pass  # 마지막 불완전 데이터는 조용히 무시
                        except Exception as e:
                            logger.error(f"스트리밍 응답 처리 중 오류: {e}")
                            # 에러가 나도 지금까지 받은 데이터는 처리 계속

                    # 응답 결합 - 문자열 처리 개선
                    if full_text:
                        # 문자열 결합 시 strip/rstrip 사용하지 않음 (텍스트 잘림 방지)
                        result = "".join(full_text)
                        # 응답이 있으면 캐시 및 반환
                        if result:
                            await self.cache.put(key, result)
                            return result
                        else:
                            # 빈 응답 처리
                            raise ValueError("Triton 서버가 응답을 생성했으나 텍스트가 비어있습니다")
                    else:
                        # 응답 자체가 없는 경우
                        raise ValueError("Triton 서버가 응답을 반환하지 않았습니다")

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
