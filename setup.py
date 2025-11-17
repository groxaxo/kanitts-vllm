"""
KaniTTS-vLLM Setup Configuration
This file enforces Python version requirements during installation.
"""
import sys

if sys.version_info >= (3, 13):
    sys.stderr.write(
        "ERROR: Python 3.13+ is not supported by KaniTTS-vLLM.\n"
        "\n"
        "This project requires Python 3.10, 3.11, or 3.12 due to dependency conflicts:\n"
        "  - nemo-toolkit requires numpy<2.0.0\n"
        "  - ml-dtypes>=0.5.0 requires numpy>=2.1.0 on Python 3.13+\n"
        "\n"
        "Please use Python 3.10, 3.11, or 3.12 to install this project.\n"
        "\n"
        "Example:\n"
        "  python3.12 -m venv venv\n"
        "  source venv/bin/activate\n"
        "  pip install -r requirements.txt\n"
    )
    sys.exit(1)

from setuptools import setup, find_packages

setup(
    name="kanitts-vllm",
    version="1.0.0",
    description="KaniTTS-vLLM: Optimized Text-to-Speech with vLLM",
    author="groxaxo",
    url="https://github.com/groxaxo/kanitts-vllm",
    packages=find_packages(),
    python_requires=">=3.10,<3.13",
    install_requires=[
        "transformers==4.53.2",
        "torch>=2.1.0",
        "numpy>=1.24.0,<2.0.0",
        "scipy>=1.11.0",
        "ml_dtypes>=0.5.0; python_version<'3.13'",
        "bitsandbytes==0.46.0",
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "nemo-toolkit[tts]==2.5.3",
        "vllm==0.10.0",
        "pydantic>=2.0.0",
        "langdetect>=1.0.9",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
