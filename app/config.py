"""
Configuration settings for the multimodal AI agent
"""
import os
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).parent.parent

# Storage directories
STORAGE_DIR = ROOT_DIR / "storage"
PAPERS_DIR = STORAGE_DIR / "papers"
IMAGES_DIR = STORAGE_DIR / "images"

# Data directory for ChromaDB
DATA_DIR = ROOT_DIR / "data"
CHROMA_DIR = DATA_DIR / "chroma"

# Model settings (can be overridden by environment variables)
TEXT_MODEL_NAME = os.environ.get("TEXT_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
IMAGE_MODEL_NAME = os.environ.get("IMAGE_MODEL_NAME", "openai/clip-vit-base-patch32")
DEVICE = os.environ.get("DEVICE", "cpu")  # Options: "cpu", "cuda", "mps"

# Text processing settings
CHUNK_SIZE = 1000  # Characters per chunk
CHUNK_OVERLAP = 200  # Overlap between chunks

# Create directories if they don't exist
PAPERS_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)