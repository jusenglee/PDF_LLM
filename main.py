import asyncio
import argparse
import fitz
import logging
import os
import sys
import locale
from pathlib import Path

from src.core.pipeline import OptimizedPipeline
from config.settings import config
from src.utils.logger import setup_logger

# Windows 한글 인코딩 설정
if os.name == "nt":                       # Windows 한정
    locale.setlocale(locale.LC_ALL, "")   # 현재 로캘 유지
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

async def process_pdf(pdf_path: str):
    # 컨텍스트 관리자 패턴으로 파이프라인 사용
    # (리소스는 자동으로 정리됨)
    async with OptimizedPipeline(
        model_path=config.model_path,
        triton_url=config.triton.url,
        ctx_len=config.ctx_len
    ) as pipeline:
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
                    semantic_info = "임베딩 검색 사용" if stats.get('semantic_search_used', False) else "전체 텍스트 사용"
                    print(f"\n📊 처리 통계: 청크 {stats.get('chunks_created', 0)}개, "
                        f"압축률 {stats.get('compression_ratio', 0):.3f}, {semantic_info}")
                    logger.info(f"처리 통계: {stats}")
            else:
                error_msg = result.get('error', '알 수 없는 오류')
                logger.error(f"PDF 처리 실패: {error_msg}")
                print(f"❌ 처리 실패: {error_msg}")
        except Exception as e:
            logger.exception(f"예기치 않은 오류 발생: {e}")
            print(f"❌ 예기치 않은 오류 발생: {e}")

async def main():
    parser = argparse.ArgumentParser(description='PDF 요약 프로그램')
    parser.add_argument('pdf_file', nargs='?', help='처리할 PDF 파일 경로', default='example3.pdf')
    parser.add_argument('--output', '-o', help='결과 저장 파일 경로')

    args = parser.parse_args()

    await process_pdf(args.pdf_file)

def cli_entry():
    """콘솔 스크립트 진입점 - setup.py entry_points에서 사용"""
    asyncio.run(main())

if __name__ == "__main__":
    # 로깅 설정
    logger = setup_logger("__main__", config.log_file, config.log_level)

    # 라이브러리 로거 설정
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.ERROR)  # transformers 경고 억제

    # Windows에서 안전한 이벤트 루프 처리
    if os.name == "nt" and sys.version_info >= (3, 8):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(main())
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
