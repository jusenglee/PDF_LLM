import logging
from typing import Dict, Any, List
from src.utils.token_manager import AdaptiveTokenManager
from src.core.triton_client import OptimizedTritonClient

logger = logging.getLogger(__name__)

class HierarchicalSummarizer:
    def __init__(self, token_mgr: AdaptiveTokenManager, triton_client: OptimizedTritonClient):
        self.token_mgr = token_mgr
        self.triton = triton_client
        self.semantic_engine = None

        # 의미 검색 엔진 초기화
        try:
            from src.search.semantic_search import SemanticSearchEngine
            self.semantic_engine = SemanticSearchEngine()
            logger.info("의미 검색 엔진 초기화 성공")
        except ImportError as e:
            logger.warning(f"의미 검색 엔진 초기화 실패: {e}")

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

        # 임베딩 기반 문장 추출 사용 (가능한 경우)
        semantic_extraction_used = False
        if self.semantic_engine is not None:
            try:
                logger.info("임베딩 기반 의미 검색 적용 중...")
                filtered_chunks = []
                for i, chunk in enumerate(chunks):
                    # 각 청크에 대해 '청크 요약' 쿼리를 사용하여 핵심 문장만 추출
                    extraction_query = f"이 문서의 핵심 내용과 중요 정보를 요약"
                    # top-k=15로 관련 문장만 추출
                    filtered_text = self.semantic_engine.extract_relevant_context(
                        extraction_query, chunk.text, top_k=15
                    )
                    # 원본 청크 대신 필터링된 문장들만 사용
                    if filtered_text.strip():
                        filtered_chunks.append({
                            'index': i,
                            'original_chunk': chunk,
                            'filtered_text': filtered_text,
                            'token_reduction': len(chunk.text) / len(filtered_text) if filtered_text else 1.0
                        })

                # 필터링된 청크가 있으면 이를 사용
                if filtered_chunks:
                    semantic_extraction_used = True
                    avg_reduction = sum(c['token_reduction'] for c in filtered_chunks) / len(filtered_chunks)
                    logger.info(f"임베딩 검색으로 텍스트 {avg_reduction:.2f}배 축소 (평균)")

                    # 필터링된 텍스트로 프롬프트 생성
                    for fc in filtered_chunks:
                        prompt = f"다음 텍스트의 핵심 내용을 {chars_per_chunk}자 이내로 요약해주세요 (파트 {fc['index']+1}/{len(chunks)}):\n\n{fc['filtered_text']}"
                        chunk_prompts.append(prompt)
            except Exception as e:
                logger.error(f"임베딩 기반 검색 처리 중 오류: {e}")
                semantic_extraction_used = False

        # 임베딩 검색 실패하거나 사용할 수 없는 경우 기존 방식 사용
        if not semantic_extraction_used:
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
            # 임베딩 기반 핵심 문장 추출 시도
            if self.semantic_engine is not None:
                try:
                    # 핵심 문장 추출 쿼리
                    extraction_query = "이 문서의 핵심 내용과 중요 정보를 요약"
                    # 상위 20개 관련 문장 추출
                    filtered_text = self.semantic_engine.extract_relevant_context(
                        extraction_query, text, top_k=20
                    )
                    if filtered_text.strip():
                        short_text = filtered_text
                        logger.info(f"임베딩 필터링으로 최종 요약용 텍스트 추출 성공: {len(filtered_text)} 문자")
                    else:
                        short_text = text[:min(len(text), 2000)]  # 원본 텍스트 앞부분만 사용
                except Exception as e:
                    logger.error(f"임베딩 기반 최종 필터링 중 오류: {e}")
                    short_text = text[:min(len(text), 2000)]  # 오류 시 원본 텍스트 앞부분만 사용
            else:
                short_text = text[:min(len(text), 2000)]  # 의미 검색 엔진 없을 경우 원본 앞부분만 사용

            final_prompt = f"다음 텍스트를 {target_length}자 이내로 간결하게 요약해주세요:\n\n{short_text}"
        else:
            final_prompt = f"다음 부분별 요약들을 종합하여 {target_length}자 이내의 최종 요약을 작성해주세요:\n\n{combined_text}"

        # 최종 요약에 충분한 토큰 할당 (한글 문자:토큰 비율 고려)
        approx_tokens = max(80, int(target_length * 1.5))  # 토큰 할당량 대폭 증가
        final_summary = await self.triton._generate_single_cached(final_prompt, approx_tokens)

        # 처리 방법 결정
        processing_method = "hierarchical"
        if semantic_extraction_used:
            processing_method = "hierarchical_with_embedding_filter"

        return {
            "chunk_summaries": chunk_summaries,
            "final_summary": final_summary,
            "processing_method": processing_method,
            "semantic_extraction_used": semantic_extraction_used
        }
