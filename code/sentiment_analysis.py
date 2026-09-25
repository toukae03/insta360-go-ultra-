"""
Stage 2 - Sentiment and scene labeling of product-review comments.

Two-stage rule system:
  1. Scene detection   : competitor comparison -> "competitor advantage";
                         worry-related terms  -> "usage concerns".
  2. Sentiment grading : SnowNLP score mapped to five tendency levels
                         (core positive / mild positive / purely neutral /
                          mild negative / negative).

Input : data/comments_raw.csv                (column: 评论内容)
Output: data/comments_labeled_sentiment.csv  (adds 分类标签, 情感置信度)

Usage:
    python code/sentiment_analysis.py
"""

import os
import time
import warnings

import pandas as pd
from snownlp import SnowNLP

warnings.filterwarnings("ignore")

INPUT_FILE = os.path.join("data", "comments_raw.csv")
OUTPUT_FILE = os.path.join("data", "comments_labeled_sentiment.csv")

TEXT_COLUMN = "评论内容"

# --- keyword rules -----------------------------------------------------------
COMPETITOR_WORDS = ["大疆", "dji", "action", "执法记录仪", "gopro"]
COMPARISON_WORDS = ["比", "不如", "更", "赢", "好", "强"]
COMPARISON_EXCEPTIONS = ["差", "不如"]
CONCERN_WORDS = ["担心", "怕", "会不会", "不敢", "安全", "续航", "没电", "易坏", "摔"]


def classify(text):
    """Return ``(label, score)`` for a single comment."""
    text = str(text).strip().lower()
    score = round(SnowNLP(text).sentiments, 4)

    # Scene labels take priority over tendency labels.
    if (
        any(w in text for w in COMPETITOR_WORDS)
        and any(w in text for w in COMPARISON_WORDS)
        and not any(bad in text for bad in COMPARISON_EXCEPTIONS)
    ):
        return "竞品优势", score
    if any(w in text for w in CONCERN_WORDS):
        return "使用顾虑", score

    # Five tendency levels (splitting the otherwise fuzzy neutral band).
    if score > 0.7:
        return "核心好评", score
    if score >= 0.55:
        return "轻微好评", score
    if score > 0.45:
        return "纯粹中性", score
    if score >= 0.3:
        return "轻微负面", score
    return "负面评价", score


def load_comments(path):
    """Read the comment CSV, tolerating both utf-8-sig and gbk encodings."""
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="gbk")
    return df.dropna(subset=[TEXT_COLUMN]).copy()


def main():
    if not os.path.exists(INPUT_FILE):
        raise SystemExit(f"Input file not found: {INPUT_FILE}")

    df = load_comments(INPUT_FILE)
    total = len(df)
    print(f"Loaded {total} valid comments from {INPUT_FILE}")

    labels, scores = [], []
    start = time.time()
    for i, comment in enumerate(df[TEXT_COLUMN], 1):
        label, score = classify(comment)
        labels.append(label)
        scores.append(score)
        if i % 300 == 0 or i == total:
            print(f"  {i}/{total} ({i / total * 100:.1f}%), {time.time() - start:.1f}s")

    df["分类标签"] = labels
    df["情感置信度"] = scores

    print("\nLabel distribution:")
    for label, count in df["分类标签"].value_counts().items():
        print(f"  {label:<6}: {count:>5} ({count / total * 100:.1f}%)")

    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\nSaved -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
