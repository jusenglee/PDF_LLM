from dataclasses import dataclass
import numpy as np
from typing import Optional

@dataclass
class TextChunk:
    text: str
    token_count: int
    priority: float = 1.0

    def __post_init__(self):
        if not self.text.strip():
            raise ValueError("빈 텍스트 청크는 허용되지 않습니다")

@dataclass
class SemanticChunk:
    text: str
    embedding: Optional[np.ndarray] = None
    score: float = 0.0
    index: int = -1

    def similarity_to(self, other: 'SemanticChunk') -> float:
        """다른 청크와의 코사인 유사도 계산"""
        return float(np.dot(self.embedding, other.embedding))
