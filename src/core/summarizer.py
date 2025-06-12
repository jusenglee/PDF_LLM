import logging
import time
import os
from typing import Dict, Any, List
from src.utils.token_manager import AdaptiveTokenManager
from src.core.triton_client import OptimizedTritonClient

logger = logging.getLogger(__name__)

MAX_SELECTED_CHUNKS = 12  # 임베딩 기반 중요도 선별 시 최대 사용할 청크 수


class HierarchicalSummarizer:
    def __init__(
        self, token_mgr: AdaptiveTokenManager, triton_client: OptimizedTritonClient
    ):
        self.token_mgr = token_mgr
        self.triton = triton_client
        self.semantic_engine = None

        # 의미 검색 엔진 초기화
        try:
            from src.search.semantic_search import SemanticSearchEngine

            self.semantic_engine = SemanticSearchEngine()
            logger.info("✅ 임베딩 모듈(의미 검색 엔진) 초기화 성공")
            print("✅ 임베딩 모듈 로드됨: 문서 핵심 내용 추출 기능 활성화")
        except ImportError as e:
            logger.warning(f"⚠️ 임베딩 모듈 초기화 실패: {e}")
            print("⚠️ 임베딩 모듈 로드 실패: 전체 텍스트 처리 모드로 전환")
            print("💡 임베딩 활성화 방법: pip install sentence-transformers faiss-cpu")

    def _select_top_chunks_by_embedding(
        self, chunks: List, query: str, top_k: int = MAX_SELECTED_CHUNKS
    ) -> (List, List[int]):
        """임베딩 유사도 기준으로 상위 청크 선별"""
        if self.semantic_engine is None:
            return chunks, list(range(len(chunks)))

        try:
            model = self.semantic_engine.model
            query_emb = model.encode([query], normalize_embeddings=True)[0]
            chunk_embs = model.encode(
                [c.text for c in chunks], normalize_embeddings=True
            )
            scores = [float((emb * query_emb).sum()) for emb in chunk_embs]
            ranked = sorted(enumerate(chunks), key=lambda x: scores[x[0]], reverse=True)
            selected = ranked[: min(top_k, len(ranked))]
            selected_indices = sorted([idx for idx, _ in selected])
            selected_chunks = [chunks[i] for i in selected_indices]
            logger.info(
                f"임베딩 중요도 순 청크 선별: {len(selected_chunks)}/{len(chunks)}개 선택"
            )
            print(
                f"💡 중요도 기반 상위 {len(selected_chunks)}개 청크 선택 (총 {len(chunks)}개 중)"
            )
            return selected_chunks, selected_indices
        except Exception as e:
            logger.warning(f"청크 중요도 선별 실패: {e}")
            return chunks, list(range(len(chunks)))

    async def smart_chunking_summary(
        self, text: str, target_length: int = 200
    ) -> Dict[str, Any]:
        """스마트 청킹 기반 계층적 요약"""

        chunks, allocation = self.token_mgr.create_adaptive_chunks(text, target_length)

        original_chunk_count = len(chunks)
        selected_indices = list(range(original_chunk_count))

        # 청크 수가 많으면 임베딩을 활용해 중요도가 높은 청크만 선별
        if (
            self.semantic_engine is not None
            and original_chunk_count > MAX_SELECTED_CHUNKS
        ):
            query = "이 문서의 핵심 내용과 중요 정보를 요약"
            chunks, selected_indices = self._select_top_chunks_by_embedding(
                chunks, query, MAX_SELECTED_CHUNKS
            )

        if len(chunks) == 1:
            # 단일 청크는 직접 요약
            prompt = f"다음 텍스트를 {target_length}자 이내로 최대한 짧게 요약해주세요:\n\n{chunks[0].text}"
            # 토큰 길이 확인 (디버깅용, 토크나이저가 있는 경우만)
            if hasattr(self.triton, "tokenizer") and self.triton.tokenizer:
                try:
                    token_count = len(
                        self.triton.tokenizer(
                            prompt, add_special_tokens=False
                        ).input_ids
                    )
                    if token_count > 2000:
                        logger.warning(
                            f"단일 청크 프롬프트 길이: {token_count} 토큰 (자동 잘림 예정)"
                        )
                except Exception:
                    pass  # 토크나이저 오류는 무시

            summary = await self.triton._generate_single_cached(prompt, target_length)

            return {
                "chunk_summaries": [summary],
                "final_summary": summary,
                "processing_method": "direct",
            }

        # 다중 청크 처리
        chunk_prompts = []
        # 청크별 요약 길이를 동적으로 할당
        base_chars_per_chunk = int(target_length * 0.7)  # 기본 요약 길이 (30% 감소)

        # 청크별 복잡도에 따른 동적 토큰 할당 계산 함수
        def calculate_dynamic_token_allocation(chunks):
            """청크 길이와 복잡도에 따라 동적으로 토큰 할당"""
            # 청크 길이 정규화 (상대적 길이 계산)
            total_tokens = sum(chunk.token_count for chunk in chunks)
            avg_tokens = total_tokens / len(chunks) if chunks else base_chars_per_chunk

            # 각 청크별 상대적 토큰 할당량 계산
            allocations = {}

            # 복잡도 지표: 문장 당 평균 단어 수로 추정
            complexity_scores = {}

            # 청크별 문장 수 계산
            for i, chunk in enumerate(chunks):
                text = chunk.text
                sentences = [s.strip() for s in text.split(".") if s.strip()]
                words = text.split()

                # 문장 수와 단어 수 계산
                sent_count = max(1, len(sentences))
                words_count = len(words)

                # 문장당 평균 단어 수로 복잡도 점수 계산
                complexity = words_count / sent_count
                complexity_scores[i] = complexity

                # 청크 길이 정규화 (상대적 크기)
                size_factor = chunk.token_count / avg_tokens

                # 복잡도와 크기를 고려한 가중치 계산
                allocations[i] = {
                    "size_factor": size_factor,
                    "complexity": complexity,
                    "token_count": chunk.token_count,
                }

            # 복잡도 점수 정규화
            if complexity_scores:
                avg_complexity = sum(complexity_scores.values()) / len(
                    complexity_scores
                )
                max_complexity = max(complexity_scores.values())
                min_complexity = min(complexity_scores.values())
                complexity_range = max(0.5, max_complexity - min_complexity)

                # 정규화된 복잡도 점수 추가
                for i in allocations:
                    norm_complexity = (
                        (complexity_scores[i] - min_complexity) / complexity_range
                        if complexity_range > 0
                        else 0.5
                    )
                    allocations[i]["norm_complexity"] = max(
                        0.7, min(1.3, 0.7 + norm_complexity)
                    )

            # 최종 할당 계산
            final_allocations = {}
            for i, alloc in allocations.items():
                # 기본 할당 (청크 크기 기반)
                base_alloc = base_chars_per_chunk * alloc.get("size_factor", 1.0)

                # 복잡도 기반 조정
                complexity_factor = alloc.get("norm_complexity", 1.0)

                # 최종 할당 (최소/최대 범위 제한)
                final_alloc = int(base_alloc * complexity_factor)

                # 너무 적거나 많은 할당 방지
                final_alloc = max(
                    int(base_chars_per_chunk * 0.7),
                    min(int(base_chars_per_chunk * 1.5), final_alloc),
                )

                final_allocations[i] = final_alloc

            logger.info(f"동적 토큰 할당: {final_allocations}")
            print(f"🔍 청크별 동적 토큰 할당 계산 완료")
            return final_allocations

        # 동적 토큰 할당 계산
        token_allocations = calculate_dynamic_token_allocation(chunks)

        # 청크 요약에 사용할 기본 문자 수
        chars_per_chunk = base_chars_per_chunk

        # 임베딩 기반 문장 추출 사용 (가능한 경우)
        semantic_extraction_used = False
        if self.semantic_engine is not None:
            try:
                logger.info("임베딩 기반 의미 검색 적용 중...")
                filtered_chunks = []
                for i, chunk in enumerate(chunks):
                    # 각 청크에 대해 '청크 요약' 쿼리를 사용하여 핵심 문장만 추출
                    extraction_query = (
                        f"이 문서의 핵심 내용과 중요 정보를 최대한 짧게 요약"
                    )
                    # 관련 문장 추출 수 증가 (더 많은 컨텍스트 제공) 및 중복 제거 임계값 설정
                    # 0.85 임계값으로 유사 문장 중복 제거 활성화
                    filtered_text = self.semantic_engine.extract_relevant_context(
                        extraction_query, chunk.text, top_k=20, dedup_threshold=0.85
                    )
                    # 원본 청크 대신 필터링된 문장들만 사용
                    if filtered_text.strip():
                        filtered_chunks.append(
                            {
                                "index": i,
                                "original_chunk": chunk,
                                "filtered_text": filtered_text,
                                "token_reduction": (
                                    len(chunk.text) / len(filtered_text)
                                    if filtered_text
                                    else 1.0
                                ),
                            }
                        )

                # 필터링된 청크가 있으면 이를 사용
                if filtered_chunks:
                    semantic_extraction_used = True
                    avg_reduction = sum(
                        c["token_reduction"] for c in filtered_chunks
                    ) / len(filtered_chunks)
                    logger.info(
                        f"임베딩 검색으로 텍스트 {avg_reduction:.2f}배 축소 (평균)"
                    )

                    # 필터링된 텍스트로 생성형 요약 프롬프트 생성
                    for fc in filtered_chunks:
                        # 해당 청크에 대한 동적 할당 크기 사용
                        idx = fc["index"]
                        dynamic_length = token_allocations.get(idx, chars_per_chunk)

                        prompt = f"""# 텍스트 요약 작업

        ## 원본 텍스트 (파트 {fc['index']+1}/{len(chunks)}):
        {fc['filtered_text']}

        ## 요약 지침:
        1. 위 텍스트의 핵심 내용만 추출하여 {dynamic_length}자 내외로 요약하세요.
        2. 중요하지 않은 세부사항은 과감히 생략하세요.
        3. 원문의 핵심 개념과 주요 아이디어를 보존하세요.
        4. 요약은 완전한 문장으로 작성하고, 문장 사이에 적절한 연결성을 유지하세요.
        5. 원문에 없는 내용을 추가하지 마세요.
        6. 모든 문장은 완결되게 작성하고, 중간에 끊기지 않도록 하세요.
        7. 요약은 반드시 마침표로 끝나야 합니다.

        ## 요약 결과:"""
                        chunk_prompts.append(prompt)
            except Exception as e:
                logger.error(f"임베딩 기반 검색 처리 중 오류: {e}")
                semantic_extraction_used = False

        # 임베딩 검색 실패하거나 사용할 수 없는 경우 생성형 요약 방식 사용
        if not semantic_extraction_used:
            for i, chunk in enumerate(chunks):
                # 해당 청크에 대한 동적 할당 크기 사용
                dynamic_length = token_allocations.get(i, chars_per_chunk)

                # 동적 할당 크기를 프롬프트에 반영
                prompt = f"""# 텍스트 요약 작업

        ## 원본 텍스트 (파트 {i+1}/{len(chunks)}):
        {chunk.text}

        ## 요약 지침:
        1. 위 텍스트의 핵심 내용만 추출하여 {dynamic_length}자 내외로 요약하세요.
        2. 중요하지 않은 세부사항은 과감히 생략하세요.
        3. 원문의 핵심 개념과 주요 아이디어를 보존하세요.
        4. 요약은 완전한 문장으로 작성하고, 문장 사이에 적절한 연결성을 유지하세요.
        5. 원문에 없는 내용을 추가하지 마세요.
        6. 모든 문장은 완결되게 작성하고, 중간에 끊기지 않도록 하세요.
        7. 요약은 반드시 마침표로 끝나야 합니다.

        ## 요약 결과:"""
                chunk_prompts.append(prompt)

        # 병렬 요약 생성 - 문자 수를 토큰 수로 변환 (한국어의 경우 더 높은 비율)
        # 동적 토큰 할당 정보 계산 (프롬프트 인덱스 → 토큰 수 매핑)
        dynamic_tokens = {}

        prompt_index = 0
        if semantic_extraction_used:
            # 임베딩 사용 시: filtered_chunks의 순서대로 프롬프트가 생성됨
            for fc in filtered_chunks:
                chunk_idx = fc["index"]
                chars = token_allocations.get(chunk_idx, chars_per_chunk)
                # 문자 수를 토큰 수로 변환 (한국어 고려)
                tokens = max(120, int(chars * 1.0))
                dynamic_tokens[prompt_index] = tokens
                prompt_index += 1
        else:
            # 임베딩 미사용 시: chunks 순서대로 프롬프트 생성됨
            for i in range(len(chunks)):
                chars = token_allocations.get(i, chars_per_chunk)
                tokens = max(120, int(chars * 1.0))
                dynamic_tokens[prompt_index] = tokens
                prompt_index += 1

        # 모든 프롬프트에 대한 토큰 할당 정보 로깅
        logger.info(f"청크별 동적 토큰 할당 (최종): {dynamic_tokens}")
        print(f"🔄 동적 토큰 할당으로 청크 요약 생성 중...")

        # 개별 청크마다 다른 토큰 수 할당을 위한 함수
        def get_tokens_for_prompt(idx):
            # 정수 값 반환 보장 (JSON 직렬화 가능)
            return int(max(40, dynamic_tokens.get(idx, 120)))  # 최소 40토큰 보장

        # 청크별로 다른 토큰 수를 적용한 병렬 생성
        chunk_summaries = await self.triton.generate_parallel_optimized(
            chunk_prompts, max_new_tokens=get_tokens_for_prompt
        )

        # 최종 통합 요약
        combined_text = "\n\n".join(
            [s for s in chunk_summaries if s and not s.startswith("[오류")]
        )  # 오류 응답 필터링

        # 요약 블록 및 문장 중복 제거 (순서 유지)
        if combined_text.strip():
            blocks = [b.strip() for b in combined_text.split("\n\n") if b.strip()]
            unique_blocks = list(dict.fromkeys(blocks))
            cleaned_blocks = []
            for block in unique_blocks:
                lines = [ln.strip() for ln in block.split("\n") if ln.strip()]
                dedup_lines = list(dict.fromkeys(lines))
                cleaned_blocks.append("\n".join(dedup_lines))
            combined_text = "\n\n".join(cleaned_blocks)

        # 문장 중복 제거 (순서 유지)
        if combined_text.strip():
            lines = combined_text.split("\n")
            deduped_lines = list(dict.fromkeys(lines))  # 순서를 유지하며 중복 제거
            deduped_lines = [line for line in deduped_lines if line.strip()]
            combined_text = "\n".join(deduped_lines)

        if not combined_text.strip():
            logger.warning("모든 청크 요약이 실패했습니다. 직접 요약을 시도합니다.")
            # 모든 청크 요약이 실패한 경우 원본 텍스트에서 간단한 요약 추출
            # 임베딩 기반 핵심 문장 추출 시도
            if self.semantic_engine is not None:
                try:
                    # 핵심 문장 추출 쿼리
                    extraction_query = "이 문서의 핵심 내용과 중요 정보를 요약"
                    # 상위 20개 관련 문장 추출 (중복 제거 적용)
                    filtered_text = self.semantic_engine.extract_relevant_context(
                        extraction_query, text, top_k=20, dedup_threshold=0.85
                    )
                    if filtered_text.strip():
                        short_text = filtered_text
                        logger.info(
                            f"임베딩 필터링으로 최종 요약용 텍스트 추출 성공: {len(filtered_text)} 문자"
                        )
                    else:
                        short_text = text[
                            : min(len(text), 2000)
                        ]  # 원본 텍스트 앞부분만 사용
                except Exception as e:
                    logger.error(f"임베딩 기반 최종 필터링 중 오류: {e}")
                    short_text = text[
                        : min(len(text), 2000)
                    ]  # 오류 시 원본 텍스트 앞부분만 사용
            else:
                short_text = text[
                    : min(len(text), 2000)
                ]  # 의미 검색 엔진 없을 경우 원본 앞부분만 사용

            final_prompt = f"다음 텍스트를 {target_length}자 이내로 간결하게 요약해주세요:\n\n{short_text}"
        else:
            # 목표 요약 길이 증가 (더 상세한 요약을 위해)
            enhanced_target_length = int(target_length * 1.5)  # 50% 늘린 요약 길이

            final_prompt = f"""# 문서 요약 생성

        ## 요약 작업:
        다음은 긴 문서의 파트별 요약입니다. 이 요약들을 통합하여 전체 내용을 담은 최종 요약을 작성해주세요.

        ## 파트별 요약:
        {combined_text}

        ## 요약 조건:
        - 요약 길이: {enhanced_target_length}자 내외로 작성할 것
        - 문서의 주제, 핵심 논점, 주요 내용을 포함할 것
        - 논리적 흐름을 유지하고 완전한 문장으로 작성할 것
        - 각 파트의 핵심만 추출하여 통합할 것
        - 모든 문장은 완결되게 작성하고, 문장이 중간에 잘리지 않도록 할 것
        - 요약의 마지막 문장은 반드시 마침표로 끝나도록 할 것

        ## 최종 요약:"""

            # 최종 요약 프롬프트 로깅
            try:
                from pathlib import Path
                from config.settings import LOG_DIR

                # 로그 디렉토리 확인
                prompt_log_dir = LOG_DIR / "prompts"
                os.makedirs(prompt_log_dir, exist_ok=True)

                # 타임스탬프로 로그 파일명 생성
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                final_log_file = prompt_log_dir / f"final_prompt_{timestamp}.log"

                # 최종 요약 프롬프트 저장
                with open(final_log_file, "w", encoding="utf-8") as f:
                    f.write("==== 최종 요약 프롬프트 ====\n\n")
                    f.write(final_prompt)

                logger.info(f"최종 요약 프롬프트 로그 저장됨: {final_log_file}")
                print(f"ℹ️ 최종 요약 프롬프트 로그 저장됨: {final_log_file}")
            except Exception as e:
                logger.warning(f"최종 요약 프롬프트 로깅 실패: {e}")

            # 최종 요약이 과도하게 길어지지 않도록 토큰 수 조정
            approx_tokens = max(150, int(enhanced_target_length * 2.5))

            print(f"\n🔄 최종 요약 생성 중... (최대 {approx_tokens} 토큰 할당)")
            final_start_time = time.time()
            # 토큰 정보 로깅 활성화
            final_summary = await self.triton._generate_single_cached(
                final_prompt, approx_tokens, log_tokens=True, log_prompt=True
            )
            final_time = time.time() - final_start_time
            print(f"✅ 최종 요약 완료 (소요시간: {final_time:.2f}초)")

        # 처리 방법 결정
        processing_method = "hierarchical"
        if semantic_extraction_used:
            processing_method = "hierarchical_with_embedding_filter"

        # 임베딩 정보 수집 (사용된 경우)
        embedding_info = None
        if semantic_extraction_used and hasattr(self.semantic_engine, "embedding_info"):
            embedding_info = self.semantic_engine.embedding_info
            logger.info(f"임베딩 정보 수집됨: {embedding_info}")

        return {
            "chunk_summaries": chunk_summaries,
            "final_summary": final_summary,
            "processing_method": processing_method,
            "semantic_extraction_used": semantic_extraction_used,
            "embedding_info": embedding_info,
            "dynamic_token_allocation": token_allocations,
            "original_chunk_count": original_chunk_count,
            "selected_chunk_indices": selected_indices,
        }
