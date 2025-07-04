# PDF 요약 시스템
# PDF 요약 도구

이 도구는 PDF 문서를 처리하고 계층적 요약을 생성하는 파이프라인을 제공합니다.

## 주요 기능

- PDF 텍스트 추출 (비동기 병렬 처리)
- 의미 검색 기반 텍스트 필터링
- 계층적 요약 생성
- Triton 추론 서버 통합

## 최근 업데이트

- 코드 모듈화 및 구조화 완료
- 불필요한 파일 정리 (pdf_summarizer.py, semantic_search.py, allReadyPromptSummaryLLM.py)
- Git 저장소 정리 및 .gitignore 추가
- 성능 최적화 및 메모리 사용량 개선

## 설치 방법

```bash
# 기본 설치
pip install -e .

# 개발 의존성 포함 설치
pip install -e ".[dev]"
```

## 사용 방법

```bash
# 기본 사용법
python main.py 요약할_PDF_파일.pdf

# 도움말 보기
python main.py --help
```

## 프로젝트 구조

```
/
├── config/             # 설정 관련 모듈
│   ├── __init__.py
│   └── settings.py     # 앱 설정
├── data/               # 데이터 저장소 (캐시, 로그 등)
├── src/                # 소스 코드
│   ├── core/           # 핵심 기능
│   │   ├── pipeline.py     # 메인 파이프라인
│   │   ├── pdf_processor.py# PDF 처리 모듈
│   │   ├── summarizer.py   # 요약 엔진
│   │   └── triton_client.py# Triton 서버 클라이언트
│   ├── models/         # 데이터 모델
│   ├── search/         # 검색 관련 모듈
│   └── utils/          # 유틸리티 함수
├── main.py             # 진입점
├── setup.py            # 설치 스크립트
└── requirements.txt    # 의존성 목록
```

## config.json 파일

모델 빌드 파라미터가 담긴 TensorRT-LLM용 설정 파일입니다. `build_config.max_seq_len` 값이
파이프라인의 기본 컨텍스트 길이로 사용되며, 필요 시 파일을 수정해 최대 입력 길이를 조정할 수 있습니다.

## 의존성

- Python 3.8 이상
- PyMuPDF (fitz)
- aiohttp
- transformers
- sentence-transformers (선택적)
- faiss-cpu (선택적)

## 라이선스

MIT 라이선스
PDF 문서를 처리하고 최신 임베딩 기술과 대규모 언어 모델을 활용하여 계층적으로 요약하는 고성능 파이프라인입니다.

## 프로젝트 개요

이 시스템은 대용량 PDF 문서를 효율적으로 처리하여 핵심 내용만 추출하고 간결한 요약을 생성합니다. 비동기 처리, 캐싱, 의미 기반 문장 필터링을 통해 최적화되었습니다.

## 주요 기능

- **비동기 PDF 처리**: 멀티스레드 기반 페이지별 텍스트 추출
- **적응형 텍스트 청킹**: 문서 길이와 컨텍스트에 맞게 최적화된 분할 알고리즘
- **계층적 요약 시스템**: 각 청크를 개별적으로 요약 후 통합
- **임베딩 기반 핵심 문장 추출**: SBERT로 중요도가 높은 문장만 선별
- **병렬 처리 최적화**: 배치 처리로 대용량 문서 고속 처리
- **캐싱 시스템**: 중복 요청 방지 및 응답 속도 향상

## 기술 아키텍처

### 1. 텍스트 추출 계층
- `AsyncPDFProcessor`: 비동기 방식으로 PDF에서 텍스트 추출
- 멀티스레딩으로 대용량 문서 고속 처리

### 2. 텍스트 청킹 및 토큰 관리
- `AdaptiveTokenManager`: 컨텍스트 길이에 맞게 텍스트 분할
- 문단 및 문장 단위 적응형 청킹 알고리즘

### 3. 의미 검색 엔진
- `SemanticSearchEngine`: SBERT와 FAISS를 활용한 벡터 유사도 검색
- 유사도 점수 기반 핵심 문장 필터링

### 4. 추론 엔진 연동
- `OptimizedTritonClient`: TensorRT-LLM 서버와 통신
- 스트리밍 처리 및 응답 캐싱

### 5. 계층적 요약
- `HierarchicalSummarizer`: 단계별 요약 파이프라인
- 의미 기반 필터링 + 추상적 요약 결합

## 임베딩 기반 필터링의 이점

- **토큰 제한 극복**: LLM이 처리해야 하는 토큰 수를 약 10배 감소
- **정보 밀도 향상**: 의미적으로 중요한 문장만 선별하여 중복 정보 제거
- **요약 품질 개선**: 핵심 정보에 집중하여 더 정확하고 관련성 높은 요약 생성
- **컨텍스트 유지**: 원본 문서의 흐름을 보존하기 위해 선별된 문장들을 원래 순서대로 유지

## 작동 프로세스

1. PDF 문서를 비동기적으로 페이지별 텍스트로 추출
2. 추출된 텍스트를 적응형 알고리즘으로 최적 크기 청크로 분할
3. 각 청크를 문장 단위로 분할하고 SBERT로 임베딩 벡터 생성
4. 요약 목적 쿼리와 문장 간 유사도 계산하여 핵심 문장만 선별
5. 선별된 문장들을 원래 순서대로 재구성하여 LLM에 전달
6. 각 청크별 요약 결과를 다시 종합하여 최종 요약 생성

## 설치 방법

```bash
pip install -r requirements.txt
```

아래 라이브러리가 필요합니다:
- PyMuPDF (fitz)
- aiohttp
- transformers
- sentence-transformers (임베딩 기반 필터링 사용 시)
- faiss-cpu (벡터 유사도 검색용)

## 사용 방법

```bash
python pdf_summarizer.py
```

기본적으로 `example3.pdf` 파일을 처리합니다. 다른 PDF를 처리하려면 코드 내 `sample_pdf` 변수를 수정하세요.

## 성능 최적화 팁

- 대용량 PDF는 `concurrency` 매개변수 조정으로 처리 속도 향상
- `batch` 크기 조정으로 병렬 처리 수준 최적화
- 캐시 TTL 및 크기 조정으로 메모리 사용량 관리
