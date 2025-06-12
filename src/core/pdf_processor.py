import os
import asyncio
import logging

logger = logging.getLogger(__name__)

class AsyncPDFProcessor:
    def __init__(self, concurrency: int = 8):
        self.sem = asyncio.Semaphore(concurrency)

    async def extract(self, pdf_path: str) -> str:
        """PDF 전체를 한 번만 연 뒤, 페이지별 텍스트를 순차-스레드로 추출."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise ImportError("PyMuPDF(fitz)가 설치되지 않았습니다. 'pip install pymupdf'로 설치하세요.")

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
        pass
