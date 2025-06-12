import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import re
from src.models.data_models import SemanticChunk

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    SEMANTIC_SEARCH_AVAILABLE = True
except ImportError:
    logger.warning("sentence-transformers 또는 faiss-cpu가 설치되지 않았습니다. 의미 검색 기능이 비활성화됩니다.")
    SEMANTIC_SEARCH_AVAILABLE = False

class SemanticSearchEngine:
    def __init__(self, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        """문장 임베딩 및 검색 엔진 초기화"""
        if not SEMANTIC_SEARCH_AVAILABLE:
            raise ImportError("의미 검색을 위해 'sentence-transformers' 및 'faiss-cpu'를 설치하세요.")

        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()
        # FAISS 인덱스 생성 (코사인 유사도를 위한 L2 정규화 벡터로 설정)
        self.index = faiss.IndexFlatIP(self.dimension)  # 내적(IP) 인덱스 - 코사인 유사도 용
        self.chunks = []
        logger.info(f"의미 검색 엔진 초기화 완료: {model_name}, 임베딩 차원: {self.dimension}")

    def split_into_sentences(self, text: str) -> List[str]:
        """텍스트를 문장 단위로 분할"""
        if not text.strip():
            return []

        # 한국어/영어 문장 경계 패턴
        pattern = r'(?<=[.!?])[\s]+'  # 마침표/느낌표/물음표 + 공백 패턴
        sentences = re.split(pattern, text)

        # 빈 문장 제거 및 최소 길이 필터링 (너무 짧은 문장은 제외)
        MIN_SENT_LEN = 10  # 최소 문장 길이
        return [s.strip() for s in sentences if len(s.strip()) > MIN_SENT_LEN]

    def preprocess_chunks(self, text: str) -> List[SemanticChunk]:
        """텍스트를 문장 단위로 분할하고 임베딩"""
        sentences = self.split_into_sentences(text)
        if not sentences:
            return []

        # 문장 임베딩 생성
        embeddings = self.model.encode(sentences, normalize_embeddings=True)

        # 청크 객체 생성
        chunks = []
        for i, (sent, emb) in enumerate(zip(sentences, embeddings)):
            chunks.append(SemanticChunk(text=sent, embedding=emb, index=i))

        # 저장
        self.chunks = chunks

        # FAISS 인덱스 구축
        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(np.vstack([c.embedding for c in chunks]))

        logger.info(f"임베딩 처리 완료: {len(chunks)} 문장")
        return chunks

    def search(self, query: str, top_k: int = 15) -> List[SemanticChunk]:
        """질의와 가장 유사한 top_k개 문장 검색"""
        query_embedding = self.model.encode([query], normalize_embeddings=True)

        # 검색 수행
        scores, indices = self.index.search(query_embedding, min(top_k, len(self.chunks)))

        # 검색 결과 구성
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                chunk.score = float(score)
                results.append(chunk)

        # 점수 기준 정렬
        results.sort(key=lambda x: x.score, reverse=True)
        return results

    def extract_relevant_context(self, query: str, text: str, top_k: int = 15) -> str:
        """원문에서 질의와 관련된 상위 K개 문장만 추출 - MAX_INPUT_TOKENS 제한 해결용"""
        # 원문을 문장으로 분할하고 임베딩
        self.preprocess_chunks(text)

        if not self.chunks:
            logger.warning("추출할 문장이 없습니다. 원본 텍스트 일부를 반환합니다.")
            return text[:min(len(text), 2000)]  # 원본 일부만 반환

        # 질의와 가장 관련성 높은 top_k개 문장 검색
        relevant_chunks = self.search(query, top_k=min(top_k, len(self.chunks)))

        if not relevant_chunks:
            logger.warning("관련 문장을 찾지 못했습니다. 원본 텍스트 일부를 반환합니다.")
            return text[:min(len(text), 2000)]  # 원본 일부만 반환

        # 원래 순서대로 정렬 (문서 흐름 유지)
        relevant_chunks.sort(key=lambda x: x.index)

        # 추가: 높은 점수의 문장 우선 선택 (중요도에 따른 필터링 강화)
        # 스코어가 0.5 이상인 문장만 선택 (임계값 설정)
        high_score_chunks = [chunk for chunk in relevant_chunks if chunk.score >= 0.5]

        # 높은 점수 문장이 너무 적으면 원래 결과 사용
        final_chunks = high_score_chunks if len(high_score_chunks) >= min(5, len(relevant_chunks)//2) else relevant_chunks

        # 연관 컨텍스트 구성
        context = "\n\n".join([chunk.text for chunk in final_chunks])

        extraction_rate = len(final_chunks) / len(self.chunks) * 100 if self.chunks else 0
        logger.info(f"질의 '{query}'에 대해 {len(final_chunks)}개 관련 문장 추출 (총 {len(self.chunks)}개 중, {extraction_rate:.1f}%)")
        return context
