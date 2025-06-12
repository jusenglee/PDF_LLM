import logging
from pathlib import Path

def setup_logger(name: str, log_file: str = None, level: str = "INFO"):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 콘솔 핸들러
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 파일 핸들러 (선택적)
    if log_file:
        log_dir = Path(log_file).parent
        if not log_dir.exists():
            log_dir.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
