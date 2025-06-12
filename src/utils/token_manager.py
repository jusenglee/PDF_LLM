import re
import logging
from typing import List, Dict, Any, Tuple
from src.models.data_models import TextChunk

logger = logging.getLogger(__name__)

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
        target_summary_length: int = 350
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
