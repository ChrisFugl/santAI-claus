from pathlib import Path

# See model card here: https://huggingface.co/bert-base-uncased
MODEL_NAME = "bert-base-uncased"

LABEL_TO_CLASS = {"Naughty": 0, "Nice": 1}
CLASS_TO_LABEL = {0: "Naughty", 1: "Nice"}

MODELS_DIR = Path(__file__).parent / "models"
