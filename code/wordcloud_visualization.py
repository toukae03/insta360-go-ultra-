"""
Stage 4 - Word clouds by dimension (overall / gender / sentiment label / user type).

Comments are tokenized with jieba, filtered through a curated stop-word list,
and rendered with ``wordcloud`` + ``matplotlib``. Only words appearing more than
three times are kept.

Input : data/comments_labeled_full.csv   (columns: 评论内容, 性别, 分类标签, 用户类型)
Output: visualization/wordclouds/*.png

Note: the script expects a Chinese font (SimHei / Microsoft YaHei on Windows);
change FONT_FAMILY when running on another platform.

Usage:
    python code/wordcloud_visualization.py
"""

import os
from collections import Counter

import jieba
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

INPUT_FILE = os.path.join("data", "comments_labeled_full.csv")
OUTPUT_DIR = os.path.join("visualization", "wordclouds")

TEXT_COLUMN = "评论内容"
FONT_FAMILY = "SimHei"
MIN_WORD_LENGTH = 2

plt.rcParams["font.sans-serif"] = [FONT_FAMILY, "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# Stop words: function words, platform/social noise, and generic descriptors
# that carry no product-specific signal. Product model names are preserved.
STOP_WORDS = set([
    # basic function words
    "的", "了", "是", "我", "很", "也", "都", "还", "比较", "挺", "非常",
    "一下", "一些", "有点", "感觉", "觉得", "用于", "用来", "可以", "能够",
    "会", "不会", "有", "没有", "在", "到", "就是", "还是", "但是", "不过",
    "所以", "因为", "如果", "虽然", "既然", "而且", "或者", "并且", "于是",
    "因此", "否则", "除非", "只要", "只有", "tim",
    # questions and interjections
    "有没有", "只是", "之类", "怎么", "为什么", "怎么样", "哪个", "哪里",
    "什么", "是不是", "要不要", "会不会", "能不能", "可不可以", "对吧",
    "呢", "啊", "呀", "嘛", "呗", "哦", "嗯", "哈", "嘿嘿", "哈哈", "吃瓜",
    "笑哭", "喜极而泣", "大哭", "爆笑",
    # pronouns, connectives, adverbs
    "这个", "那个", "这样", "那样", "这么", "那么", "这里", "那里", "时候",
    "之前", "之后", "现在", "已经", "其实", "当然", "毕竟", "反正", "总之",
    "各位", "大家", "朋友", "同学", "学生", "兄弟", "我们", "你们", "他们",
    "自己", "人家", "别人", "任何", "所有", "全部", "整个",
    # platform and e-commerce noise
    "回复", "评论", "留言", "点赞", "收藏", "转发", "关注", "私信", "@",
    "沙发", "板凳", "地板", "围观", "路过", "打卡", "红包", "抽奖", "京东",
    "淘宝", "拼多多", "天猫", "抖音", "B站", "小红书", "优惠", "券", "补贴",
    # generic descriptors without product signal
    "不错", "很好", "非常好", "很棒", "厉害", "牛", "强", "可以", "还行",
    "一般", "不好", "不行", "差", "烂", "垃圾", "问题", "毛病", "缺点", "优点",
    "体验", "感受", "使用", "用", "买", "入手", "购买", "测评", "评测", "上手",
    "相机", "设备", "工具", "机器", "产品", "东西", "物品", "宝贝", "商品",
])


def tokenize(texts):
    """Tokenize a series of comments and return a frequency counter."""
    words = []
    for text in texts:
        for word in jieba.cut(str(text)):
            word = word.strip()
            if len(word) >= MIN_WORD_LENGTH and word not in STOP_WORDS and not word.isdigit():
                words.append(word)
    return Counter(words)


def create_wordcloud(freq, save_filename, title):
    """Render and save a single word cloud from a word-frequency counter."""
    if not freq:
        print(f"Skipped (no words): {title}")
        return

    wc = WordCloud(
        width=1600,
        height=1000,
        background_color="white",
        max_words=150,
        colormap="viridis",
        collocations=False,
    ).generate_from_frequencies(freq)

    plt.figure(figsize=(12, 7.5))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=18)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, save_filename), dpi=200)
    plt.close()
    print(f"Saved: {save_filename}")


def main():
    if not os.path.exists(INPUT_FILE):
        raise SystemExit(f"Input file not found: {INPUT_FILE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")
    print(f"Loaded {len(df)} comments from {INPUT_FILE}")

    # 1. Overall word cloud.
    create_wordcloud(
        tokenize(df[TEXT_COLUMN]),
        "01_overall_comments.png",
        f"Overall comment keywords (n={len(df)})",
    )

    # 2. By gender.
    for gender in df["性别"].dropna().unique():
        subset = df[df["性别"] == gender]
        create_wordcloud(
            tokenize(subset[TEXT_COLUMN]),
            f"02_gender_{gender}.png",
            f"Gender: {gender} (n={len(subset)})",
        )

    # 3. By sentiment label.
    for label in df["分类标签"].dropna().unique():
        subset = df[df["分类标签"] == label]
        create_wordcloud(
            tokenize(subset[TEXT_COLUMN]),
            f"03_label_{label}.png",
            f"Sentiment label: {label} (n={len(subset)})",
        )

    # 4. By user type.
    for user_type in df["用户类型"].dropna().unique():
        subset = df[df["用户类型"] == user_type]
        create_wordcloud(
            tokenize(subset[TEXT_COLUMN]),
            f"04_usertype_{user_type}.png",
            f"User type: {user_type} (n={len(subset)})",
        )

    print(f"\nAll word clouds saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
