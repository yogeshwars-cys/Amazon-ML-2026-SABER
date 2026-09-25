"""Paths. Override with environment variables SABER_DATA (the challenge `dataset/` folder) and SABER_WORK
(scratch space for normalised parquet, embeddings, legs, candidates)."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("SABER_DATA", os.path.join(ROOT, "dataset")).rstrip("/\\") + "/"
WORK = os.environ.get("SABER_WORK", os.path.join(ROOT, "work")).rstrip("/\\") + "/"
os.makedirs(WORK, exist_ok=True)
