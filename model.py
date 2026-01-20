import torch
import torch.nn as nn
from torchvision import transforms
from transformers import RobertaModel, RobertaTokenizer
from torchvision.models import resnet50, ResNet50_Weights
from pathlib import Path
from download_online_models import _hf_path_or_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

threshold: float = 0.5

HF_LOCAL_DIR = Path(__file__).resolve().parent / "local_hf"
HF_ROBERTA_DIR = HF_LOCAL_DIR / "roberta-base"

class ImageEncoder(nn.Module):
    def __init__(self, hidden_dim=768):
        super().__init__()

        backbone = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        # Output: [B, 2048, 7, 7]

        self.proj = nn.Linear(2048, hidden_dim)

    def forward(self, images):
        feats = self.backbone(images)            # [B, 2048, 7, 7]
        B, C, H, W = feats.shape

        feats = feats.flatten(2).transpose(1, 2)  # [B, 49, 2048]
        feats = self.proj(feats)                  # [B, 49, 768]

        return feats


class TextEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        roberta_src, local_only = _hf_path_or_id(HF_ROBERTA_DIR, "FacebookAI/roberta-base")
        self.roberta = RobertaModel.from_pretrained(roberta_src, local_files_only=local_only)
        # self.roberta = RobertaModel.from_pretrained("FacebookAI/roberta-base")

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        return outputs.last_hidden_state  # [B, M, D]


class CrossAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            batch_first=True
        )

    def forward(self, text_feats, image_feats):
        fused, _ = self.attn(
            query=text_feats,
            key=image_feats,
            value=image_feats
        )
        return fused


class ClassificationHead(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.mlp(x).squeeze(-1)


def get_transform():
    weights = ResNet50_Weights.DEFAULT
    return weights.transforms()

class SubmissionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = ImageEncoder()
        self.text_encoder = TextEncoder()
        self.cross_attn = CrossAttention(hidden_dim=768)
        self.classifier = ClassificationHead(hidden_dim=768)

        self._tokenizer = None

        self.to(device)

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            roberta_src, local_only = _hf_path_or_id(HF_ROBERTA_DIR, "FacebookAI/roberta-base")
            self._tokenizer = RobertaTokenizer.from_pretrained(roberta_src, local_files_only=local_only)
            # self._tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")
        return self._tokenizer

    def encode_text(self, text, *, device=None):
        enc = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=40,
            return_tensors="pt",
        )
        if device is not None:
            enc = {k: v.to(device) for k, v in enc.items()}
        return enc

    def prepare_image_tensor(self, image_tensor: torch.Tensor, *, device=None):
        if device is None:
            device = next(self.parameters()).device

        if image_tensor.ndim == 3:
            image_tensor = self._image_transform(image_tensor)

        return image_tensor.unsqueeze(0).to(device)

    def forward(self, images, input_ids, attention_mask):
        img_feats = self.image_encoder(images)
        txt_feats = self.text_encoder(input_ids, attention_mask)

        fused = self.cross_attn(txt_feats, img_feats)
        cls_token = fused[:, 0]
        logits = self.classifier(cls_token)
        return logits

    @torch.no_grad()
    def predict(self, image_tensor, text_string):
        """Return a Python float score in [0,1] as required by the evaluator."""
        # Always use the model's current device.
        dev = next(self.parameters()).device
        score = self.predict_proba(image_tensor, text_string, device=dev)
        return float(score)

    @torch.no_grad()
    def predict_proba(self, image_tensor, text: str, *, device=None) -> float:
        self.eval()
        if device is None:
            device = next(self.parameters()).device

        pixel_values = self.prepare_image_tensor(image_tensor, device=device)
        enc = self.encode_text(text, device=device)

        logits = self(pixel_values, enc["input_ids"], enc["attention_mask"])
        return float(torch.sigmoid(logits).item())