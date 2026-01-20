from transformers import RobertaModel, RobertaTokenizer
import os
from pathlib import Path

ROBERTA_ID = "FacebookAI/roberta-base"
DEFAULT_LOCAL_DIR = Path(__file__).resolve().parent / "local_hf"
DEFAULT_ROBERTA_DIRNAME = "roberta-base"


def get_roberta_local_dir(local_dir: str | os.PathLike = DEFAULT_LOCAL_DIR) -> Path:
    return Path(local_dir) / DEFAULT_ROBERTA_DIRNAME

def download_hf_roberta_model(local_dir: str | os.PathLike = DEFAULT_LOCAL_DIR, *, force: bool = False) -> str:
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    roberta_dir = get_roberta_local_dir(local_dir)

    if force and roberta_dir.exists():
        for p in roberta_dir.glob("*"):
            if p.is_file():
                p.unlink(missing_ok=True)
    roberta_dir.mkdir(parents=True, exist_ok=True)
    RobertaModel.from_pretrained(ROBERTA_ID).save_pretrained(roberta_dir)
    RobertaTokenizer.from_pretrained(ROBERTA_ID).save_pretrained(roberta_dir)

    return str(roberta_dir)

def _hf_path_or_id(local_path: Path, model_id: str) -> tuple[str, bool]:
    """Return (path_or_id, local_only)."""
    if local_path.exists() and any(local_path.iterdir()):
        return str(local_path), True
    return model_id, False

if __name__ == "__main__":
    download_hf_roberta_model()