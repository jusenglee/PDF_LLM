import logging
import json
import time
from pathlib import Path
from transformers import AutoTokenizer
from src.utils.token_manager import AdaptiveTokenManager
from src.core.pdf_processor import AsyncPDFProcessor
from src.core.triton_client import OptimizedTritonClient
from src.core.summarizer import HierarchicalSummarizer
from config.settings import ROOT_DIR

logger = logging.getLogger(__name__)

class OptimizedPipeline:
    def __init__(
        self,
        model_path: str = "./",
        triton_url: str = "http://203.250.238.30:8888/v2/models/ensemble/generate_stream",
        ctx_len: int | None = None,
        config_path: str | Path | None = None,
    ):
        if ctx_len is None:
            # config.json에서 max_seq_len 값을 읽어 기본 컨텍스트 길이로 사용
            path = Path(config_path) if config_path else ROOT_DIR / "config.json"
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                ctx_len = int(data.get("build_config", {}).get("max_seq_len", 2048))
            except Exception as e:
                logger.warning(f"config.json 로드 실패: {e}")
                ctx_len = 2048

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

        # 시작 시간 기록
        start_time = time.time()
        process_times = {}

        try:
            # 1. PDF 추출
            extraction_start = time.time()
            try:
                raw_text = await self.pdf_proc.extract(pdf_path)
                process_times['extraction'] = time.time() - extraction_start
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
            chunking_start = time.time()
            try:
                chunks, allocation = self.token_mgr.create_adaptive_chunks(
                    clean_text, target_summary_length
                )
                process_times['chunking'] = time.time() - chunking_start
            except Exception as e:
                logger.error(f"텍스트 청킹 실패: {e}")
                print(f"❌ 청킹 오류: {e}")
                return {
                    "error": f"텍스트 청킹 오류: {e}",
                    "success": False
                }

            # 4. 요약 처리
            summarizing_start = time.time()
            try:
                result = await self.summarizer.smart_chunking_summary(
                    clean_text, target_summary_length
                )
                process_times['summarizing'] = time.time() - summarizing_start
            except Exception as e:
                logger.error(f"요약 처리 실패: {e}")
                print(f"❌ 요약 오류: {e}")
                return {
                    "error": f"요약 처리 오류: {e}",
                    "success": False
                }

            # 5. 성공 결과 반환
            # 임베딩 사용 여부 확인
            semantic_search_used = hasattr(self.summarizer, 'semantic_engine') and self.summarizer.semantic_engine is not None

            # 임베딩 상태 명확히 로깅
            if semantic_search_used:
                logger.info("✅ 임베딩 모듈 활성화 상태로 처리 완료")
                print("✨ 임베딩 모듈 활성화: 문서 핵심 내용 추출 기능 사용 중")
            else:
                logger.warning("⚠️ 임베딩 모듈 비활성화 상태로 처리 완료 (sentence-transformers, faiss-cpu 설치 필요)")
                print("⚠️ 임베딩 모듈 비활성화: 전체 텍스트 처리 모드로 동작 중")
                print("💡 임베딩 활성화 방법: pip install sentence-transformers faiss-cpu")

            # 전체 처리 시간 계산
            total_time = time.time() - start_time

            result.update({
                "token_allocation": allocation,
                "success": True,
                "processing_stats": {
                    "chunks_created": len(chunks),
                    "avg_chunk_size": sum(c.token_count for c in chunks) / len(chunks),
                    "compression_ratio": len(result.get("final_summary", "")) / len(clean_text),
                    "semantic_search_used": semantic_search_used,
                    "process_times": process_times,
                    "total_time": total_time
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
