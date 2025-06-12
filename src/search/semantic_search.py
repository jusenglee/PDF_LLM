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

        # 한국어/영어 문장 경계 패턴 개선
        # 마침표/느낌표/물음표 다음에 공백 또는 줄바꿈이 있는 경우 (더 유연한 매칭)
        pattern = r'(?<=[.!?])(?=[\s\n]|$)'  
        sentences = re.split(pattern, text)

        # 빈 문장 제거 및 최소 길이 필터링 (너무 짧은 문장은 제외)
        MIN_SENT_LEN = 10  # 최소 문장 길이
        filtered = [s.strip() for s in sentences if len(s.strip()) > MIN_SENT_LEN]

        # 문장 수가 너무 적으면 줄바꿈 기준으로 다시 분할
        if len(filtered) <= 3 and len(text) > 500:
            paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
            paragraphs = [p for p in paragraphs if len(p) > MIN_SENT_LEN]
            if len(paragraphs) > len(filtered):
                logger.info(f"문장 분할 개선: 문장 경계 {len(filtered)}개 → 단락 기준 {len(paragraphs)}개")
                return paragraphs

        return filtered

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

    def extract_relevant_context(self, query: str, text: str, top_k: int = 15, dedup_threshold: float = 0.85) -> str:
        """원문에서 질의와 관련된 상위 K개 문장만 추출 - MAX_INPUT_TOKENS 제한 해결용

        Args:
            query: 검색 쿼리
            text: 원본 텍스트
            top_k: 추출할 상위 문장 수
            dedup_threshold: 중복 제거를 위한 유사도 임계값 (높을수록 더 엄격한 중복 제거)
        """
        # 원문을 문장으로 분할하고 임베딩
        self.preprocess_chunks(text)

        if not self.chunks:
            logger.warning("추출할 문장이 없습니다. 원본 텍스트 일부를 반환합니다.")
            return text[:min(len(text), 2000)]  # 원본 일부만 반환

        # 질의와 가장 관련성 높은 top_k개 문장 검색 (중복 제거 고려해 2배 더 가져옴)
        relevant_chunks = self.search(query, top_k=min(top_k * 2, len(self.chunks)))

        if not relevant_chunks:
            logger.warning("관련 문장을 찾지 못했습니다. 원본 텍스트 일부를 반환합니다.")
            return text[:min(len(text), 2000)]  # 원본 일부만 반환

        # 유사한 문장 중복 제거
        unique_chunks = []
        duplicate_count = 0

        # 중복 제거 전에 점수순으로 정렬 (점수가 높은 문장 우선 보존)
        relevant_chunks.sort(key=lambda x: x.score, reverse=True)

        # 실제 사용할 문장 수 (원하는 양의 2배까지 최대 확보)
        max_chunks_needed = min(top_k * 2, len(relevant_chunks))

        for chunk in relevant_chunks:
            # 임베딩 없는 청크 건너뛰기
            if chunk.embedding is None or np.isnan(chunk.embedding).any():
                logger.warning("임베딩이 없거나 유효하지 않은 청크 발견, 건너뜀")
                continue

            is_duplicate = False

            # 기존 선택된 청크들과 현재 청크의 유사도 계산
            for selected in unique_chunks:
                try:
                    # 임베딩 유효성 확인
                    if selected.embedding is None or np.isnan(selected.embedding).any():
                        continue

                    similarity = float(np.dot(chunk.embedding, selected.embedding))  # 코사인 유사도

                    # 유사도가 임계값보다 높으면 중복으로 처리
                    if similarity > dedup_threshold:
                        is_duplicate = True
                        duplicate_count += 1
                        break
                except Exception as e:
                    logger.warning(f"유사도 계산 중 오류 발생 (무시): {e}")
                    continue

            # 중복이 아니면 추가
            if not is_duplicate:
                unique_chunks.append(chunk)

            # 충분한 수의 고유 청크를 얻었으면 종료
            if len(unique_chunks) >= max_chunks_needed:
                break

        # 충분한 청크가 선택되지 않았으면 경고
        if not unique_chunks and relevant_chunks:
            logger.warning("중복 제거 후 선택된 청크가 없음. 원본 청크 사용")
            unique_chunks = relevant_chunks[:max_chunks_needed]

        if duplicate_count > 0:
            logger.info(f"중복 제거: {duplicate_count}개 유사 문장 제거됨 (임계값: {dedup_threshold})")

        # 원래 순서대로 정렬 (문서 흐름 유지)
        unique_chunks.sort(key=lambda x: x.index)

        # 추가: 높은 점수의 문장 우선 선택 (중요도에 따른 필터링 강화)
        # 스코어가 0.5 이상인 문장만 선택 (임계값 설정)
        high_score_chunks = [chunk for chunk in unique_chunks if chunk.score >= 0.5]

        # 높은 점수 문장이 너무 적으면 원래 결과 사용
        final_chunks = high_score_chunks if len(high_score_chunks) >= min(5, len(unique_chunks)//2) else unique_chunks

        # 연관 컨텍스트 구성
        context = "\n\n".join([chunk.text for chunk in final_chunks])

        extraction_rate = len(final_chunks) / len(self.chunks) * 100 if self.chunks else 0
        logger.info(f"질의 '{query}'에 대해 {len(final_chunks)}개 관련 문장 추출 (총 {len(self.chunks)}개 중, {extraction_rate:.1f}%)")

        # 콘솔에 중복 제거 및 문장 선택 정보 출력
        if duplicate_count > 0:
            print(f"💡 임베딩 처리: {duplicate_count}개 유사 문장 제거됨, {len(final_chunks)}개 문장 선택")

        # 임베딩 처리 정보 수집
        self.embedding_info = {
            "model_name": getattr(self.model, "name", "sentence-transformers/all-MiniLM-L6-v2"),
            "dimension": self.dimension,
            "sentences_count": len(self.chunks),
            "selected_count": len(final_chunks),
            "deduped_count": duplicate_count,
            "filtering_ratio": extraction_rate,
            "dedup_threshold": dedup_threshold
        }

        # 콘솔에 임베딩 상세 정보 출력
        print(f"📊 임베딩 처리: 총 {len(self.chunks)}개 문장에서 {len(final_chunks)}개 선택 ({extraction_rate:.1f}%)")

        return context
