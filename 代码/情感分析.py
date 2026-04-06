import pandas as pd
import warnings
import os
import time
from snownlp import SnowNLP

warnings.filterwarnings("ignore")

# 1. 数据加载（和之前一致）
csv_file = r'D:\Tomoyo\变大变重？有变强吗？Insta360 GO Ultra上手_评论.csv'
if not os.path.exists(csv_file):
    print(f"❌ 错误：找不到文件 '{csv_file}'")
    exit()

print(f"📂 正在读取文件...")
try:
    df = pd.read_csv(csv_file, encoding='utf-8-sig')
except UnicodeDecodeError:
    df = pd.read_csv(csv_file, encoding='gbk')
df_clean = df.dropna(subset=['评论内容']).copy()
total_valid = len(df_clean)
print(f"✅ 读取成功：有效评论数 {total_valid} 条")

# 2. 调整后的分类函数（拆分轻微倾向，缩小纯粹中性范围）
def classify_with_tendency(text):
    text = str(text).strip().lower()
    snlp = SnowNLP(text)
    score = round(snlp.sentiments, 4)

    # 关键关键词定义（和你的评论匹配）
    core_pos_words = ["防抖", "画质", "清晰", "便携", "小巧", "轻", "好用", "值", "牛", "绝了"]
    competitor_words = ["大疆", "dji", "action", "执法记录仪", "gopro"]
    compare_adv_words = ["比", "不如", "更", "赢", "好", "强"]
    negative_words = ["贵", "割韭菜", "智商税", "不值", "发热", "卡顿", "难用", "鸡肋", "垃圾", "坑", "劝退"]
    worry_words = ["担心", "怕", "会不会", "不敢", "安全", "续航", "没电", "易坏", "摔"]

    # 优先判断“场景类标签”（竞品优势、使用顾虑）
    if any(cw in text for cw in competitor_words) and any(aw in text for aw in compare_adv_words) and not any(bad in text for bad in ["差", "不如"]):
        return "竞品优势", score
    if any(ww in text for ww in worry_words):
        return "使用顾虑", score

    # 再判断“态度倾向类标签”（拆分轻微倾向）
    if score > 0.7:  # 明确正面
        return "核心好评", score
    elif 0.55 <= score <= 0.7:  # 轻微正面（有小缺点但整体满意）
        return "轻微好评", score
    elif 0.45 < score < 0.55:  # 纯粹中性（完全无倾向）
        return "纯粹中性", score
    elif 0.3 <= score <= 0.45:  # 轻微负面（有小优点但整体不满）
        return "轻微负面", score
    else:  # 明确负面（<0.3）
        return "负面评价", score

# 3. 批量分类（和之前一致，只改了分类函数）
print("\n🔍 基于拆分标签开始分类...")
start_time = time.time()
results = []
for idx, row in df_clean.iterrows():
    comment = row['评论内容']
    label, score = classify_with_tendency(comment)
    results.append({
        "原Excel行号": idx + 2,
        "评论内容": comment,
        "分类标签": label,
        "情感置信度": score
    })
    # 进度显示
    if (len(results) % 300 == 0) or (len(results) == total_valid):
        elapsed = time.time() - start_time
        print(f"   进度：{len(results)}/{total_valid} 条（{len(results)/total_valid*100:.1f}%），耗时 {elapsed:.1f} 秒")

# 4. 合并结果并统计
result_df = pd.DataFrame(results)
df['分类标签'] = None
df['情感置信度'] = None
for _, res_row in result_df.iterrows():
    df.loc[res_row['原Excel行号'] - 2, '分类标签'] = res_row['分类标签']
    df.loc[res_row['原Excel行号'] - 2, '情感置信度'] = res_row['情感置信度']

# 显示调整后的结果（重点看中性占比下降）
print("\n" + "="*60)
print("📊 调整后分类结果统计（拆分轻微倾向）：")
label_count = df['分类标签'].value_counts()
for label, count in label_count.items():
    percentage = count / total_valid * 100
    print(f"   {label:<8}：{count:<4} 条（{percentage:.1f}%）")
print("="*60)

# 保存调整后的结果
output_file = r'D:\Tomoyo\Insta360_评论_拆分倾向分类结果.csv'
df.to_csv(output_file, index=False, encoding='utf-8-sig')
print(f"\n✅ 调整完成！文件已保存至：{output_file}")
print("💡 分析建议：重点看“核心好评”（用户明确满意的点）和“轻微负面+负面评价”（用户主要不满的点），更有针对性～")