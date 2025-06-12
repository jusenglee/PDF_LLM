from setuptools import setup, find_packages

setup(
    name="pdf-summarizer",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "aiohttp>=3.8.0",
        "pymupdf>=1.19.0",
        "transformers>=4.20.0",
        "numpy>=1.21.0",
        "sentence-transformers>=2.2.0",
        "faiss-cpu>=1.7.0"
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio",
            "flake8>=5.0.0",
            "black>=22.0.0",
            "mypy>=0.950",
        ],
        "gpu": ["faiss-gpu"],
    },
        python_requires=">=3.8",
        entry_points={
            "console_scripts": [
                "pdf-summarize=main:cli_entry",
            ],
        },
    author="PDF Summarizer Team",
    description="PDF 문서 요약 도구",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
