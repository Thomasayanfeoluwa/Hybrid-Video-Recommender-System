import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s')

# List of all files and directories to create
list_of_files = [
    # Root files
    "app.py",
    "setup.py",
    "requirements.txt",
    ".env",

    # Source code
    "src/__init__.py",
    "src/config.py",
    "src/logger.py",

    # Data ingestion & preprocessing
    "src/data/__init__.py",
    "src/data/load_data.py",
    "src/data/preprocess.py",

    # Feature engineering
    "src/features/__init__.py",
    "src/features/content_embeddings.py",
    "src/features/interaction_matrix.py",

    # Models
    "src/models/__init__.py",
    "src/models/als_model.py",
    "src/models/hybrid_model.py",
    "src/models/utils_model.py",

    # Evaluation
    "src/evaluation/__init__.py",
    "src/evaluation/ranking_metrics.py",

    # API
    "src/api/__init__.py",
    "src/api/routes.py",

    # Utilities
    "src/utils/__init__.py",
    "src/utils/helpers.py",

    # Data directories
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "data/embeddings/.gitkeep",

    # Notebooks
    "notebooks/01_eda.ipynb",
    "notebooks/02_content_embeddings.ipynb",
    "notebooks/03_als_training.ipynb",
    "notebooks/04_hybrid_fusion.ipynb",

    # Research
    "research/eda.ipynb",
    "research/experiments.ipynb",

    # Templates
    "templates/index.html",

    # Scripts
    "scripts/train_als.py",
    "scripts/build_embeddings.py",
    "scripts/store_index.py",

    # Models directory
    "models/faiss_index/.gitkeep",
    "models/als_model/.gitkeep",
]

# Create directories and files
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir = filepath.parent

    # Create directories if they don't exist
    if filedir != Path("."):
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Ensured directory exists: {filedir}")

    # Create files if they don't exist
    if not filepath.exists():
        with open(filepath, "w") as f:
            pass
        logging.info(f"Created file: {filepath}")
    else:
        logging.info(f"File already exists: {filepath}")

logging.info("Project skeleton creation complete!")
