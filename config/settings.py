from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional

# 프로젝트 루트 디렉토리
ROOT_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))  

# 데이터 및 캐시 경로
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
LOG_DIR = DATA_DIR / "logs"

# 필요한 디렉토리 생성
for directory in [DATA_DIR, CACHE_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

@dataclass
class TritonConfig:
    url: str = "http://203.250.238.30:8888/v2/models/ensemble/generate_stream"
    timeout: int = 300
    max_connections: int = 10
    max_server_tokens: int = 4096

@dataclass
class AppConfig:
    triton: TritonConfig = TritonConfig()
    semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    log_level: str = "INFO"
    log_file: str = str(LOG_DIR / "pdf_pipeline.log")
    cache_ttl: int = 3600
    cache_max_size: int = 1000
    model_path: str = "./"
    ctx_len: int = 2048

# 기본 설정 인스턴스 생성
config = AppConfig()
