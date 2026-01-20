from torch.utils.data import Dataset
from PIL import Image
import torch
from transformers import RobertaModel, RobertaTokenizer
from torchvision.models import resnet50, ResNet50_Weights
import os
import pickle
import csv
from tqdm import tqdm


class CrossModalDataset(Dataset):
    def __init__(self, samples, tokenizer, image_processor):
        self.samples = samples
        self.tokenizer = tokenizer
        self.image_processor = image_processor

        weights = ResNet50_Weights.DEFAULT
        self.image_transform = weights.transforms()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Image
        image = Image.open(sample["image_path"]).convert("RGB")
        image = self.image_transform(image)

        # Text
        encoding = self.tokenizer(
            sample["text"],
            padding="max_length",
            truncation=True,
            max_length=40,
            return_tensors="pt",
        )

        return {
            "image": image,
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(sample["label"], dtype=torch.float)
        }

def load_true_false_csv(filepath, images_dir):
    """
    Load CSV with rows: image,caption,is_true(1 or 0)
    Returns a list of dicts: {'image_path': <abs path>, 'text': <caption>, 'label': 0|1}
    - images_dir is used to resolve filenames if the 'image' column contains only a basename.
    """
    samples = []
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
    image_files = {os.path.basename(f): os.path.join(images_dir, f) for f in os.listdir(images_dir) if os.path.splitext(f)[1].lower() in exts} if os.path.isdir(images_dir) else {}

    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in tqdm(reader, desc="Loading true/false CSV", unit="lines"):
            if not row or len(row) < 3:
                continue
            img_col = row[0].strip()
            caption = row[1].strip()
            label_raw = row[2].strip()

            try:
                label = int(label_raw)
                label = 1 if label == 1 else 0
            except Exception:
                # skip malformed label
                continue

            # Try direct path first
            candidate = img_col
            if not os.path.isabs(candidate):
                candidate = os.path.join(images_dir, img_col)
            image_path = candidate if os.path.isfile(candidate) else None

            # Fallback: lookup by basename in images_dir
            if image_path is None:
                base = os.path.basename(img_col)
                if base in image_files:
                    image_path = image_files[base]
                else:
                    # try matching by id (without extension)
                    img_id = os.path.splitext(base)[0]
                    for fname in image_files:
                        if os.path.splitext(fname)[0] == img_id:
                            image_path = image_files[fname]
                            break

            if image_path is None or not os.path.isfile(image_path):
                # skip missing images
                continue

            samples.append({
                "image_path": image_path,
                "text": caption,
                "label": label
            })
    return samples

def save_dataset(dataset, filepath):
    with open(filepath, "wb") as f:
        pickle.dump(dataset, f)

def create_and_save_dataset(
    csv_path: str = "dataset_true_false_improved.csv",
    images_dir: str = os.path.join("archive", "Images"),
    out_filepath: str = "dataset.pkl",
):
    """
    Build CrossModalDataset directly from a CSV (image,caption,is_true)
    and save it to `out_filepath` (default named_datasetV2.pkl).
    Returns: (dataset, samples, out_filepath)
    """
    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"No images directory found at `{images_dir}`")

    samples = load_true_false_csv(csv_path, images_dir)

    if not samples:
        raise FileNotFoundError(f"No valid samples parsed from `{csv_path}`")

    tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")
    image_processor = resnet50(weights=ResNet50_Weights.DEFAULT)

    dataset = CrossModalDataset(samples, tokenizer, image_processor)
    save_dataset(dataset, out_filepath)
    return dataset, samples, out_filepath

if __name__ == "__main__":
    create_and_save_dataset()