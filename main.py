import asyncio
import argparse
import fitz
import logging
import os
import sys
import locale

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

                final_summary = result.get("final_summary", "요약 내용이 없습니다.")
                # 최종 요약이 비어있거나 매우 짧은 경우 처리
                if len(final_summary.strip()) < 10:
                    print("\n⚠️ 최종 요약 생성 실패 --------------------")
                    print("파트별 요약 내용을 참고해 주세요.")
                else:
                    print("\n✅ 최종 요약 --------------------")
                    # 줄바꿈으로 요약을 보기 좋게 표시
                    formatted_summary = "\n".join([line.strip() for line in final_summary.split(".") if line.strip()])
                    print(formatted_summary)

                stats = result.get('processing_stats', {})
                if stats:
                    is_semantic = stats.get('semantic_search_used', False)
                    semantic_info = "임베딩 검색 (중복 제거) 사용 ✅" if is_semantic else "전체 텍스트 사용 ⚠️"
                    compression = stats.get('compression_ratio', 0)
                    chunks_count = stats.get('chunks_created', 0)

                    print(f"\n📊 처리 통계:")
                    print(f"  • 문서 분할: {chunks_count}개 청크")

                    # 압축률 표시 개선 (백분율 대신 축소 비율로 표시)
                    if compression > 0:
                        reduction_ratio = 1.0 / compression if compression > 0 else 0
                        print(f"  • 압축률: {compression:.3f} (원본 대비 {int(compression*100)}%, {reduction_ratio:.1f}배 축소)")
                    else:
                        print(f"  • 압축률: 계산 불가 (최종 요약이 없음)")

                    print(f"  • 처리 모드: {semantic_info}")

                    # 소요시간 표시 추가
                    times = stats.get('process_times', {})
                    total_time = stats.get('total_time', 0)

                    print(f"\n⏱️ 처리 소요시간:")
                    if times.get('extraction'):
                        print(f"  • PDF 추출: {times.get('extraction', 0):.2f}초")
                    if times.get('chunking'):
                        print(f"  • 문서 분할: {times.get('chunking', 0):.2f}초")
                    if times.get('summarizing'):
                        print(f"  • 요약 생성: {times.get('summarizing', 0):.2f}초")
                    print(f"  • 총 소요시간: {total_time:.2f}초")

                    if not is_semantic:
                        print("\n💡 임베딩 모듈을 활성화하면 더 정확한 요약이 가능합니다.")
                        print("   설치 명령어: pip install sentence-transformers faiss-cpu")

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
    parser.add_argument('pdf_file', nargs='?', help='처리할 PDF 파일 경로', default='data/example3.pdf')
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
