import logging
from transformers import AutoTokenizer
from src.utils.token_manager import AdaptiveTokenManager
from src.core.pdf_processor import AsyncPDFProcessor
from src.core.triton_client import OptimizedTritonClient
from src.core.summarizer import HierarchicalSummarizer

logger = logging.getLogger(__name__)

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
                    "compression_ratio": len(result.get("final_summary", "")) / len(clean_text),
                    "semantic_search_used": hasattr(self.summarizer, 'semantic_engine') and 
                                            self.summarizer.semantic_engine is not None
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
