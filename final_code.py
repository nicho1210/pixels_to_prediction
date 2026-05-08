# ── 1. Imports & Configuration ───────────────────────────────────────────────
import os
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ── Paths ────────────────────────────────────────────────────────────────────
# Adjust these paths to match your local environment
DATA_DIR   = Path("data")

# ── Model ────────────────────────────────────────────────────────────────────
MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"

# ── Basic Settings ───────────────────────────────────────────────────────────
IMG_SIZE        = 224

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# ── 2a. Load CSVs ─────────────────────────────────────────────────────────────
train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df   = pd.read_csv(DATA_DIR / "val.csv")
test_df  = pd.read_csv(DATA_DIR / "test.csv")

# The 'choices' column is a JSON string, so we parse it
for df in [train_df, val_df, test_df]:
    df["choices"] = df["choices"].apply(json.loads)

print(f"Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
train_df.head(2)


# ── 2b. Improve the Prompt Format ────────────────────────────────────────────
CHOICE_LETTERS = "ABCDEFGHIJ"

def build_prompt(row: pd.Series, include_answer: bool = False) -> str:
    """
    Build a prompt for multiple-choice visual reasoning.
    """
    context_parts = []
    lecture = row.get("lecture", "")
    hint = row.get("hint", "")

    if pd.notna(lecture) and str(lecture).strip():
        context_parts.append(str(lecture).strip())
    if pd.notna(hint) and str(hint).strip():
        context_parts.append(str(hint).strip())

    context_str = "\n".join(context_parts)

    choices = row["choices"]
    choices_str = "\n".join(
        f"{CHOICE_LETTERS[i]}. {c}" for i, c in enumerate(choices)
    )

    prompt = "<image>\n"
    prompt += "You are solving a multiple-choice science question.\n"
    prompt += "Choose the single best answer.\n"
    prompt += "Reply with only one capital letter.\n\n"

    if context_str:
        prompt += f"Context:\n{context_str}\n\n"

    prompt += f"Question: {row['question']}\n"
    prompt += f"Choices:\n{choices_str}\n\n"
    prompt += "Answer"

    if include_answer:
        answer_idx = int(row["answer"])
        prompt += f": {CHOICE_LETTERS[answer_idx]}"
    else:
        prompt += ":"

    return prompt

print(build_prompt(train_df.iloc[0], include_answer=True))


# ── 2c. PyTorch Dataset ───────────────────────────────────────────────────────
class ScienceQADataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_dir: Path, img_size: int = 224, is_train: bool = True):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.img_size = img_size
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, rel_path: str) -> Image.Image:
        img = Image.open(self.data_dir / rel_path).convert("RGB")
        img = img.resize((self.img_size, self.img_size), Image.BICUBIC)
        return img

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        img = self._load_image(row["image_path"])

        if self.is_train:
            return {
                "image":  img,
                "text":   build_prompt(row, include_answer=True),
                "answer": int(row["answer"]),
            }
        else:
            return {
                "image":   img,
                "text":    build_prompt(row, include_answer=False),
                "choices": row["choices"],
                "answer":  int(row["answer"]) if "answer" in row else -1,
            }

train_ds = ScienceQADataset(train_df, DATA_DIR, img_size=IMG_SIZE, is_train=True)
val_ds   = ScienceQADataset(val_df,   DATA_DIR, img_size=IMG_SIZE, is_train=False)
test_ds  = ScienceQADataset(test_df,  DATA_DIR, img_size=IMG_SIZE, is_train=False)

print(f"Datasets created: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")



# ── 3a. Load SmolVLM model + run one inference example ───────────────────────
from transformers import AutoProcessor, AutoModelForVision2Seq

processor = AutoProcessor.from_pretrained(MODEL_ID)
if processor.tokenizer.pad_token is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    device_map="auto" if torch.cuda.is_available() else None,
    low_cpu_mem_usage=True,
 )
if not torch.cuda.is_available():
    model.to(device)
model.eval()

# Pick a sample from validation set
sample = val_df.iloc[0]
sample_image = Image.open(DATA_DIR / sample["image_path"]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
sample_prompt = build_prompt(sample, include_answer=False)

inputs = processor(
    text=[sample_prompt],
    images=[sample_image],
    return_tensors="pt",
)
inputs = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in inputs.items()}

with torch.inference_mode():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
    )

decoded = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("Prompt:")
print(sample_prompt)
print("\nModel output:")
print(decoded)
print(f"\nGround-truth answer index: {sample['answer']}")


# ── 4a. Output Parsing Helpers ───────────────────────────────────────────────
import re

CHOICE_LETTERS = "ABCDEFGHIJ"

def decode_new_tokens(processor, generated_ids, inputs):
    prompt_len = inputs["input_ids"].shape[1]
    generated_only = generated_ids[:, prompt_len:]
    text = processor.batch_decode(generated_only, skip_special_tokens=True)[0].strip()
    return text

def extract_answer_index(text: str, num_choices: int) -> int:
    """
    Convert model outputs like:
    'A'
    'Answer: A'
    'A. the eggs will hatch'
    'A\n\nAnswer: E\n\nExplanation: ...'
    into a 0-indexed integer answer.
    """
    text_up = text.strip().upper()

    patterns = [
        r"ANSWER\s*[:\-]?\s*([A-J])",
        r"FINAL\s+ANSWER\s*[:\-]?\s*([A-J])",
        r"CORRECT\s+ANSWER\s*[:\-]?\s*([A-J])",
    ]

    for pattern in patterns:
        m = re.search(pattern, text_up)
        if m:
            idx = ord(m.group(1)) - ord("A")
            if 0 <= idx < num_choices:
                return idx

    lines = [line.strip() for line in text_up.splitlines() if line.strip()]

    for line in lines:
        m = re.fullmatch(r"([A-J])", line)
        if m:
            idx = ord(m.group(1)) - ord("A")
            if 0 <= idx < num_choices:
                return idx

        m = re.match(r"^([A-J])[\.\)\:\- ]", line)
        if m:
            idx = ord(m.group(1)) - ord("A")
            if 0 <= idx < num_choices:
                return idx

    m = re.search(r"\b([A-J])\b", text_up)
    if m:
        idx = ord(m.group(1)) - ord("A")
        if 0 <= idx < num_choices:
            return idx

    return 0

print("Helpers ready.")


# ── 4b. Test One Validation Prediction ───────────────────────────────────────
sample = val_df.iloc[0]
sample_image = Image.open(DATA_DIR / sample["image_path"]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
sample_prompt = build_prompt(sample, include_answer=False)

inputs = processor(
    text=[sample_prompt],
    images=[sample_image],
    return_tensors="pt",
)
inputs = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in inputs.items()}

with torch.inference_mode():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=10,
        do_sample=False,
    )

decoded = decode_new_tokens(processor, generated_ids, inputs)
pred_idx = extract_answer_index(decoded, int(sample["num_choices"]))
true_idx = int(sample["answer"])

print("Raw model output:", repr(decoded))
print("Predicted index:", pred_idx)
print("Ground-truth index:", true_idx)
print("Correct:", pred_idx == true_idx)


# ── 4b. Test One Validation Prediction ───────────────────────────────────────
def predict_one_row(row):
    img_path = DATA_DIR / row["image_path"]

    if img_path.exists():
        image = Image.open(img_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    else:
        print(f"Warning: missing image -> {img_path}")
        image = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color="white")

    prompt = build_prompt(row, include_answer=False)

    inputs = processor(
        text=[prompt],
        images=[image],
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=1,
            do_sample=False,
            temperature=None,
        )

    decoded = decode_new_tokens(processor, generated_ids, inputs)
    pred_idx = extract_answer_index(decoded, int(row["num_choices"]))
    return pred_idx, decoded


# ── 5a. Run Validation Inference ─────────────────────────────────────────────
def predict_dataframe(df: pd.DataFrame, print_every: int = 50) -> pd.DataFrame:
    records = []

    for i, (_, row) in enumerate(df.iterrows(), start=1):
        pred_idx, decoded = predict_one_row(row)

        records.append({
            "id": row["id"],
            "pred_answer": pred_idx,
            "raw_output": decoded,
        })

        if i % print_every == 0 or i == len(df):
            print(f"Processed {i}/{len(df)}")

    return pd.DataFrame(records)


# ── 6a. Load PEFT and Attach LoRA ────────────────────────────────────────────
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
)

model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    device_map="auto" if torch.cuda.is_available() else None,
    low_cpu_mem_usage=True,
)

if not torch.cuda.is_available():
    model.to(device)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ── 6b. Build a Training Dataset for LoRA ────────────────────────────────────
class ScienceQATrainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, data_dir: Path, img_size: int = 224):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.img_size = img_size

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.data_dir / row["image_path"]

        if img_path.exists():
            image = Image.open(img_path).convert("RGB").resize((self.img_size, self.img_size))
        else:
            image = Image.new("RGB", (self.img_size, self.img_size), color="white")

        prompt_text = build_prompt(row, include_answer=False)
        full_text = build_prompt(row, include_answer=True)

        return {
            "image": image,
            "prompt_text": prompt_text,
            "full_text": full_text,
        }

train_lora_ds = ScienceQATrainDataset(train_df, DATA_DIR, img_size=IMG_SIZE)
val_lora_ds = ScienceQATrainDataset(val_df, DATA_DIR, img_size=IMG_SIZE)

print(len(train_lora_ds), len(val_lora_ds))


# ── 6c. Create the Collate Function ──────────────────────────────────────────
def lora_collate_fn(batch):
    images = [item["image"] for item in batch]
    prompt_texts = [item["prompt_text"] for item in batch]
    full_texts = [item["full_text"] for item in batch]

    prompt_inputs = processor(
        text=prompt_texts,
        images=images,
        return_tensors="pt",
        padding=True,
    )

    full_inputs = processor(
        text=full_texts,
        images=images,
        return_tensors="pt",
        padding=True,
    )

    labels = full_inputs["input_ids"].clone()

    for i in range(labels.size(0)):
        prompt_len = int((prompt_inputs["attention_mask"][i] == 1).sum().item())
        labels[i, :prompt_len] = -100

    if processor.tokenizer.pad_token_id is not None:
        labels[labels == processor.tokenizer.pad_token_id] = -100

    batch_out = {
        "input_ids": full_inputs["input_ids"],
        "attention_mask": full_inputs["attention_mask"],
        "pixel_values": full_inputs["pixel_values"],
        "labels": labels,
    }

    if "pixel_attention_mask" in full_inputs:
        batch_out["pixel_attention_mask"] = full_inputs["pixel_attention_mask"]

    return batch_out


# ── 6d. Create Dataloaders ───────────────────────────────────────────────────
train_loader = DataLoader(
    train_lora_ds,
    batch_size=2,
    shuffle=True,
    collate_fn=lora_collate_fn,
)

val_loader = DataLoader(
    val_lora_ds,
    batch_size=2,
    shuffle=False,
    collate_fn=lora_collate_fn,
)

print("Dataloaders ready.")


# 6d-1
def evaluate_validation_accuracy(df: pd.DataFrame, print_every: int = 100) -> float:
    pred_df = predict_dataframe(df, print_every=print_every)
    eval_df = df[["id", "answer"]].merge(pred_df, on="id", how="left")
    eval_df["correct"] = eval_df["answer"] == eval_df["pred_answer"]
    return float(eval_df["correct"].mean())


# ── 6e. Train for 2 Epochs and Keep the Best Checkpoint ──────────────────────
import copy
from torch.optim import AdamW
from tqdm.auto import tqdm

optimizer = AdamW(model.parameters(), lr=1e-4)

num_epochs = 3

best_val_acc = -1.0
best_model_state = None
history = []

for epoch in range(num_epochs):
    model.train()
    epoch_losses = []

    for step, batch in enumerate(tqdm(train_loader, desc=f"Training Epoch {epoch+1}/{num_epochs}"), start=1):
        batch = {
            k: v.to(model.device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

        outputs = model(**batch)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_losses.append(loss.item())

        if step % 100 == 0:
            print(f"Epoch {epoch+1}, Step {step}, Train loss: {np.mean(epoch_losses[-100:]):.4f}")

    avg_train_loss = float(np.mean(epoch_losses))
    print(f"Epoch {epoch+1} average train loss: {avg_train_loss:.4f}")

    model.eval()
    val_acc = evaluate_validation_accuracy(val_df, print_every=100)
    print(f"Epoch {epoch+1} validation accuracy: {val_acc:.4f}")

    history.append({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_accuracy": val_acc,
    })

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = copy.deepcopy(model.state_dict())
        print(f"New best model found at epoch {epoch+1} with val accuracy {val_acc:.4f}")

print("Training complete.")
print("Best validation accuracy:", best_val_acc)
print(history)


# ── 6f. Load Best Checkpoint and Re-Evaluate ─────────────────────────────────
if best_model_state is not None:
    model.load_state_dict(best_model_state)
    print("Loaded best model state.")

model.eval()
best_val_acc_final = evaluate_validation_accuracy(val_df, print_every=100)
print(f"Best-checkpoint validation accuracy: {best_val_acc_final:.4f}")


# ── 6g. Generate Test Predictions with LoRA ──────────────────────────────────
model.eval()

test_pred_df_lora = predict_dataframe(test_df, print_every=50)

print(test_pred_df_lora.shape)
test_pred_df_lora.head(10)


# ── 6h. Save LoRA Submission ─────────────────────────────────────────────────
submission_df_lora = test_pred_df_lora[["id", "pred_answer"]].copy()
submission_df_lora = submission_df_lora.rename(columns={"pred_answer": "answer"})
submission_df_lora["answer"] = submission_df_lora["answer"].astype(int)

print(submission_df_lora.head())
print(submission_df_lora.shape)
print(submission_df_lora.columns.tolist())

submission_df_lora.to_csv("submission_lora.csv", index=False)
print("Saved submission_lora.csv")


# ── 6i. Check LoRA Submission Format ─────────────────────────────────────────
check_df_lora = pd.read_csv("submission_lora.csv")

print(check_df_lora.head())
print(check_df_lora.shape)
print(check_df_lora.columns.tolist())
print("Unique ids:", check_df_lora["id"].nunique())
print("Total rows:", len(check_df_lora))
print("Answer dtype:", check_df_lora["answer"].dtype)
print("Answer min/max:", check_df_lora["answer"].min(), check_df_lora["answer"].max())