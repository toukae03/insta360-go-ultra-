"""
Stage 3 - User-type classification of product-review comments.

Five user segments:
    daily recorder / travel & outdoor / professional creator / vlogger / first-time trier

Strategy (keyword anchoring first, model second):
  1. Comments containing explicit segment signals are anchored by keyword rules
     (high precision).
  2. Ambiguous comments are classified with a BERT model (``bert-base-chinese``
     with a 5-way classification head); a fine-tuned state dict is loaded when
     available, otherwise the base model runs with keyword anchoring only.

Input : data/comments_labeled_sentiment.csv
Output: data/comments_labeled_full.csv       (adds 用户类型)

Optional:
    insta360_user_type_model.pth  - fine-tuned weights for the 5-way head.

If your network requires a proxy to download the pretrained model, set the
HTTP_PROXY / HTTPS_PROXY environment variables before running.

Usage:
    python code/user_classification.py
"""

import os
import warnings

import pandas as pd
import torch
from transformers import BertForSequenceClassification, BertTokenizer

warnings.filterwarnings("ignore")

INPUT_FILE = os.path.join("data", "comments_labeled_sentiment.csv")
OUTPUT_FILE = os.path.join("data", "comments_labeled_full.csv")
WEIGHTS_FILE = "insta360_user_type_model.pth"
MODEL_NAME = "bert-base-chinese"

TEXT_COLUMN = "评论内容"

USER_TYPES = [
    "日常记录型用户",
    "旅行户外型用户",
    "专业创作型用户",
    "vlog博主型用户",
    "尝鲜体验型用户",
]

USER_TYPE_KEYWORDS = {
    "日常记录型用户": ["带娃", "家庭", "宠物", "日常", "生活碎片", "拍合照", "简单操作", "价格适中"],
    "旅行户外型用户": ["旅行", "徒步", "骑行", "登山", "户外", "西藏", "新疆", "续航够", "防摔", "防水"],
    "专业创作型用户": ["iso", "raw", "动态范围", "噪点", "解析力", "商拍", "赛事", "风光摄影", "色彩还原"],
    "vlog博主型用户": ["vlog", "探店", "穿搭", "手持防抖", "收音", "剪辑", "长时间拍摄", "素材导出"],
    "尝鲜体验型用户": ["第一次用", "试试水", "之前用", "换设备", "从手机换", "旧相机", "好奇", "体验一下"],
}


def load_model():
    """Load tokenizer + model, using fine-tuned weights when available."""
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(USER_TYPES)
    )

    if os.path.exists(WEIGHTS_FILE):
        model.load_state_dict(torch.load(WEIGHTS_FILE, map_location=torch.device("cpu")))
        print(f"Loaded fine-tuned weights: {WEIGHTS_FILE}")
    else:
        print(
            f"Fine-tuned weights not found ({WEIGHTS_FILE}); using the base model "
            "with keyword anchoring only."
        )

    model.eval()
    return tokenizer, model


def classify_user_type(comment, tokenizer, model):
    """Return the user segment for one comment."""
    text = str(comment).strip().lower()

    # Step 1: keyword anchoring.
    for user_type, keywords in USER_TYPE_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return user_type

    # Step 2: BERT for ambiguous comments.
    inputs = tokenizer(
        text, return_tensors="pt", max_length=128, padding="max_length", truncation=True
    )
    with torch.no_grad():
        logits = model(**inputs).logits
    return USER_TYPES[int(torch.argmax(logits, dim=1))]


def main():
    if not os.path.exists(INPUT_FILE):
        raise SystemExit(f"Input file not found: {INPUT_FILE}")

    tokenizer, model = load_model()

    try:
        df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(INPUT_FILE, encoding="gbk")
    df = df.dropna(subset=[TEXT_COLUMN]).copy()
    total = len(df)
    print(f"Loaded {total} valid comments from {INPUT_FILE}")

    user_types = []
    for i, comment in enumerate(df[TEXT_COLUMN], 1):
        user_types.append(classify_user_type(comment, tokenizer, model))
        if i % 200 == 0 or i == total:
            print(f"  {i}/{total} ({i / total * 100:.1f}%)")
    df["用户类型"] = user_types

    print("\nUser-type distribution:")
    for user_type, count in df["用户类型"].value_counts().items():
        print(f"  {user_type:<12}: {count:>5} ({count / total * 100:.1f}%)")

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\nSaved -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
