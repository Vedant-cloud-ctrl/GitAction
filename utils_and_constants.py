import shutil
from pathlib import Path

DATASET_TYPES = ["test", "train"]
DROP_COLNAMES = []
TARGET_COLUMN = "y"
RAW_DATASET = "bank-full.csv"
PROCESSED_DATASET = "processed_dataset/bank-full.csv"


def delete_and_recreate_dir(path):
    try:
        shutil.rmtree(path)
    except:
        pass
    finally:
        Path(path).mkdir(parents=True, exist_ok=True)
