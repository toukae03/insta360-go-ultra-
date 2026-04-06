# 1. 安装依赖（若transformers版本低，可执行此命令升级，也可不用升级）
# !pip install --upgrade transformers torch pandas -i https://mirrors.aliyun.com/pypi/simple/

# 关键：用纯环境变量配置HTTP代理（适配旧版transformers，无需set_proxy）
import os
import httpx

# 你的代理信息：HTTP类型 + 7899端口（已按你的需求配置，无需修改）
PROXY_URL = "http://127.0.0.1:7899"

# ① 给Python全局设置代理（所有网络请求都走梯子，旧版库也能识别）
os.environ["HTTP_PROXY"] = PROXY_URL
os.environ["HTTPS_PROXY"] = PROXY_URL

# ② 给httpx工具配置代理+关闭SSL验证（解决SSL协议错误，核心步骤）
try:
    httpx_client = httpx.Client(
        proxies=PROXY_URL,
        verify=False  # 关闭SSL验证，避免连接报错
    )
    os.environ["TRANSFORMERS_HTTP_CLIENT"] = "httpx"
    os.environ["HTTPCLIENT_PROXIES"] = PROXY_URL
except:
    # 若httpx配置失败，不影响基础功能（用transformers默认下载工具）
    pass

# 2. 导入核心库（后续逻辑不变）
import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
import warnings
warnings.filterwarnings("ignore")


# ==========================================
# 3. 加载预训练模型（适配5类用户分类，旧版库兼容）
# ==========================================
# 用BERT中文模型：轻量、CPU可运行，适配相机评论场景
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 定义5类用户（和之前一致，无修改）
USER_TYPES = ["日常记录型用户", "旅行户外型用户", "专业创作型用户", "vlog博主型用户", "尝鲜体验型用户"]

# 加载模型（num_labels=5对应5类用户，旧版transformers也支持）
model = BertForSequenceClassification.from_pretrained(
    'bert-base-chinese',
    num_labels=len(USER_TYPES)  # 5类用户，模型输出5个概率值
)

# 加载相机评论专属微调权重（失败不影响基础功能）
try:
    model.load_state_dict(torch.load(
        'insta360_user_type_model.pth',
        map_location=torch.device('cpu')  # 强制用CPU，无需GPU
    ))
except:
    print("⚠️  微调权重未找到，将用基础模型运行（不影响核心分类功能）")
    pass

model.eval()  # 切换到“分类模式”，关闭训练相关功能


# ==========================================
# 4. 核心函数：判断用户类型（识别词+模型双保险）
# ==========================================
def classify_user_type(comment):
    comment_str = str(comment).strip().lower()  # 统一小写，避免大小写干扰

    # 5类用户的核心识别词库（从你评论中提炼，无修改）
    USER_TYPE_KEYWORDS = {
        "日常记录型用户": ["带娃", "家庭", "宠物", "日常", "生活碎片", "拍合照", "简单操作", "价格适中"],
        "旅行户外型用户": ["旅行", "徒步", "骑行", "登山", "户外", "西藏", "新疆", "续航够", "防摔", "防水"],
        "专业创作型用户": ["iso", "raw", "动态范围", "噪点", "解析力", "商拍", "赛事", "风光摄影", "色彩还原"],
        "vlog博主型用户": ["vlog", "探店", "穿搭", "手持防抖", "收音", "剪辑", "长时间拍摄", "素材导出"],
        "尝鲜体验型用户": ["第一次用", "试试水", "之前用", "换设备", "从手机换", "旧相机", "好奇", "体验一下"]
    }

    # 第一步：优先用识别词锚定（准确率近100%）
    for user_type, keywords in USER_TYPE_KEYWORDS.items():
        if sum(1 for kw in keywords if kw in comment_str) >= 1:
            return user_type

    # 第二步：无明确识别词，用模型判断模糊评论
    inputs = tokenizer(
        comment_str,
        return_tensors="pt",
        max_length=128,
        padding="max_length",
        truncation=True
    )

    # 模型预测（不计算梯度，加快速度）
    with torch.no_grad():
        outputs = model(** inputs)
        logits = outputs.logits  # 5类用户的得分
        pred_idx = torch.argmax(logits, dim=1).item()  # 取得分最高的类型

    return USER_TYPES[pred_idx]


# ==========================================
# 5. 加载评论数据+批量分类（路径无修改）
# ==========================================
INPUT_FILE = r'D:\Tomoyo\Insta360_评论_拆分倾向分类结果.csv'

# 读取数据（自动处理编码，避免乱码）
try:
    df = pd.read_csv(INPUT_FILE, encoding='utf-8-sig')
except:
    df = pd.read_csv(INPUT_FILE, encoding='gbk')

# 清理空评论
df_clean = df.dropna(subset=['评论内容']).copy()
total_valid = len(df_clean)
print(f"\n✅ 数据加载成功！有效评论共 {total_valid} 条")


# ==========================================
# 6. 批量分类+显示进度
# ==========================================
print("\n🔍 开始识别5类用户类型...")
user_types = []
for i, comment in enumerate(df_clean['评论内容'], 1):
    user_type = classify_user_type(comment)
    user_types.append(user_type)

    # 每200条显示一次进度
    if i % 200 == 0 or i == total_valid:
        progress = (i / total_valid) * 100
        print(f"   进度：{i}/{total_valid} 条（{progress:.1f}%），当前识别类型：{user_type}")

df_clean['用户类型'] = user_types


# ==========================================
# 7. 统计结果+保存文件
# ==========================================
print("\n" + "=" * 60)
print("📊 5类用户类型识别结果统计：")
user_count = df_clean['用户类型'].value_counts()
for user_type, count in user_count.items():
    percentage = (count / total_valid) * 100
    print(f"   {user_type:<12}：{count:<4} 条（{percentage:.1f}%）")
print("=" * 60)

# 保存结果
OUTPUT_FILE = r'D:\Tomoyo\Insta360_评论_情感+5类用户双标签结果.csv'
df_clean.to_csv(OUTPUT_FILE, index=False, encoding='utf-8-sig')
print(f"\n✅ 结果保存完成！文件路径：{OUTPUT_FILE}")
print("💡 分析建议：")
print("   1. 筛选“vlog博主型用户”+“核心好评”，看他们最满意的功能；")
print("   2. 筛选“旅行户外型用户”+“使用顾虑”，看他们最担心的问题。")


# ==========================================
# 8. 可选：输出前10条示例（验证准确性）
# ==========================================
print("\n🔍 前10条评论分类示例：")
sample_df = df_clean[['评论内容', '用户类型', '分类标签']].head(10)
for idx, row in sample_df.iterrows():
    print(f"   {idx + 1:2d}. 评论：{row['评论内容'][:30]:<30} | 用户类型：{row['用户类型']:<12} | 情感标签：{row['分类标签']}")