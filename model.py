import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download


MODEL_REPO = "sivachandrakb/malayalam-sarcasm-deberta"


def load_model():

    tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/deberta-v3-base"
    )

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="best_model.pt"
    )

    model = torch.load(model_path, map_location=torch.device("cpu"))
    model.eval()

    return model, tokenizer


def predict_sarcasm(text, model, tokenizer):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    probs = F.softmax(logits, dim=1)

    confidence, pred = torch.max(probs, dim=1)

    confidence = confidence.item()
    pred = pred.item()

    label = "Sarcastic" if pred == 1 else "Non-Sarcastic"

    # Simple uncertainty estimate
    uncertainty = 1 - confidence

    return label, confidence, uncertainty
