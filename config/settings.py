from dataclasses import dataclass
import os
from pathlib import Path
from typing import Optional
import json

# 프로젝트 루트 디렉토리
ROOT_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# 데이터 및 캐시 경로
DATA_DIR = ROOT_DIR / "data"
CACHE_DIR = DATA_DIR / "cache"
LOG_DIR = DATA_DIR / "logs"

# 필요한 디렉토리 생성
for directory in [DATA_DIR, CACHE_DIR, LOG_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


def load_ctx_len_from_json(path: Path) -> int:
    """config.json에서 max_seq_len 값을 읽어 ctx_len 기본값으로 사용"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return int(data.get("build_config", {}).get("max_seq_len", 2048))
    except Exception:
        return 2048

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
    ctx_len: int = load_ctx_len_from_json(ROOT_DIR / "config.json")

# 기본 설정 인스턴스 생성
config = AppConfig()
