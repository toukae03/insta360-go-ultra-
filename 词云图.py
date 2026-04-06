# 1. 安装依赖（首次运行执行，后续可注释；若已安装过直接跳过）
# !pip install jieba wordcloud matplotlib pandas -i https://mirrors.aliyun.com/pypi/simple/

# 2. 导入核心库
import jieba
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import pandas as pd
import os
from collections import Counter

# --------------------------
# 基础配置（避免中文乱码、统一保存路径）
# --------------------------
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 定义词云图保存目录（和旧版隔离，方便对比）
output_dir = r'D:\Tomoyo\insta360\所有维度词云图_最终优化版'
os.makedirs(output_dir, exist_ok=True)

# --------------------------
# 最终版精准停用词表（只删无效词，保留所有产品型号）
# --------------------------
STOP_WORDS = set([
    # 1. 基础无意义词（保留产品型号相关，只删纯冗余）
    "的", "了", "是", "我", "很", "也", "都", "还", "比较", "挺", "非常",
    "一下", "一些", "有点", "感觉", "觉得", "用于", "用来", "可以", "能够",
    "会", "不会", "有", "没有", "在", "到", "就是", "还是", "但是", "不过",
    "所以", "因为", "如果", "虽然", "既然", "而且", "或者", "并且", "于是",
    "因此", "否则", "除非", "只要", "只有", "tim",

    # 2. 疑问词/语气词（核心优化：彻底删除你说的"有没有""只是""之类"等）
    "有没有", "只是", "之类", "怎么", "为什么", "怎么样", "哪个", "哪里",
    "什么", "是不是", "要不要", "会不会", "能不能", "可不可以", "对吧",
    "呢", "啊", "呀", "嘛", "呗", "哦", "嗯", "哈", "嘿嘿", "哈哈", "吃瓜",
    "笑哭", "喜极而泣", "大哭", "爆笑",

    # 3. 代词/连词/副词（纯冗余，无业务价值）
    "这个", "那个", "这样", "那样", "这么", "那么", "这里", "那里", "时候",
    "之前", "之后", "现在", "已经", "其实", "当然", "毕竟", "反正", "总之",
    "各位", "大家", "朋友", "同学", "学生", "兄弟", "我们", "你们", "他们",
    "自己", "人家", "别人", "任何", "所有", "全部", "整个",

    # 4. 互动/平台/电商相关词
    "回复", "评论", "留言", "点赞", "收藏", "转发", "关注", "私信", "@",
    "沙发", "板凳", "地板", "围观", "路过", "打卡", "红包", "抽奖", "京东",
    "淘宝", "拼多多", "天猫", "抖音", "B站", "小红书", "优惠", "券", "补贴",

    # 5. 通用描述词（无业务指向性，纯情绪词）
    "不错", "很好", "非常好", "很棒", "厉害", "牛", "强", "可以", "还行",
    "一般", "不好", "不行", "差", "烂", "垃圾", "问题", "毛病", "缺点", "优点",
    "体验", "感受", "使用", "用", "买", "入手", "购买", "测评", "评测", "上手",
    "相机", "设备", "工具", "机器", "产品", "东西", "物品", "宝贝", "商品"
])


# --------------------------
# 通用函数：生成词云图（双重过滤，保留产品型号）
# --------------------------
def create_wordcloud(text, save_filename, title):
    # 步骤1：中文精确分词（避免拆分产品型号，如"go3s"完整保留）
    words = jieba.lcut(text, cut_all=False)

    # 步骤2：第一轮过滤（长度≥2 + 不在停用词表，产品型号自动保留）
    initial_filtered = [
        word for word in words
        if len(word) >= 2 and word not in STOP_WORDS
    ]
    if not initial_filtered:
        print(f"⚠️ 【{title}】无有效关键词，跳过生成")
        return

    # 步骤3：第二轮过滤（词频≥3，彻底剔除低频无效词，只保留高频核心词）
    word_freq = Counter(initial_filtered)
    final_filtered = [word for word, count in word_freq.items() if count >= 3]
    if not final_filtered:
        print(f"⚠️ 【{title}】有效关键词均为低频词（<3次），跳过生成")
        return

    # 步骤4：配置词云样式（强化核心词视觉占比）
    wc = WordCloud(
        width=1000,
        height=700,
        background_color="white",
        font_path="C:/Windows/Fonts/simhei.ttf",
        max_words=120,  # 压缩显示数量，只保留TOP120核心词
        collocations=False,
        random_state=42,
        relative_scaling=0.7  # 强化词频差异，高频核心词更突出
    ).generate(" ".join(final_filtered))

    # 步骤5：绘制并保存
    plt.figure(figsize=(12, 8))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=18, pad=20)
    plt.savefig(
        os.path.join(output_dir, save_filename),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
    print(f"✅ 【{title}】已保存至：{os.path.join(output_dir, save_filename)}")


# --------------------------
# 步骤1：加载数据（路径不变，确保正确）
# --------------------------
try:
    df = pd.read_csv(
        r'D:\Tomoyo\insta360\Insta360_评论_情感+5类用户双标签结果.csv',
        encoding='utf-8-sig'
    )
    # 清理空值（保留评论和性别都非空的数据）
    df = df.dropna(subset=['评论内容', '性别'])
    print(f"✅ 数据加载成功！有效评论共 {len(df)} 条")
    print(f"📊 性别分布：{df['性别'].value_counts().to_dict()}")
    print(f"📊 分类标签分布：{df['分类标签'].value_counts().to_dict()}")
    print(f"📊 用户类型分布：{df['用户类型'].value_counts().to_dict()}")
except Exception as e:
    print(f"❌ 数据加载失败！错误原因：{str(e)}")
    print("请检查数据文件路径是否正确、文件是否损坏")
    exit()

# --------------------------
# 步骤2-5：生成所有维度词云图（逻辑不变，过滤更精准）
# --------------------------
# 2. 整体评论词云
all_comments = " ".join(str(comment) for comment in df['评论内容'])
create_wordcloud(
    text=all_comments,
    save_filename="1_整体评论词云_最终优化版.png",
    title=f"整体评论关键词云图（共{len(df)}条评论）"
)

# 3. 性别分类词云
for gender in df['性别'].unique():
    gender_comments = df[df['性别'] == gender]['评论内容']
    gender_text = " ".join(str(comment) for comment in gender_comments)
    create_wordcloud(
        text=gender_text,
        save_filename=f"2_性别_{gender}_词云_最终优化版.png",
        title=f"性别：{gender} 关键词云图（共{len(gender_comments)}条评论）"
    )

# 4. 分类标签词云
for label in df['分类标签'].unique():
    label_comments = df[df['分类标签'] == label]['评论内容']
    label_text = " ".join(str(comment) for comment in label_comments)
    create_wordcloud(
        text=label_text,
        save_filename=f"3_分类标签_{label}_词云_最终优化版.png",
        title=f"分类标签：{label} 关键词云图（共{len(label_comments)}条评论）"
    )

# 5. 用户类型词云
for user_type in df['用户类型'].unique():
    user_comments = df[df['用户类型'] == user_type]['评论内容']
    user_text = " ".join(str(comment) for comment in user_comments)
    create_wordcloud(
        text=user_text,
        save_filename=f"4_用户类型_{user_type}_词云_最终优化版.png",
        title=f"用户类型：{user_type} 关键词云图（共{len(user_comments)}条评论）"
    )

# --------------------------
# 完成提示
# --------------------------
print("\n🎉 所有最终优化版词云图生成完成！")
print(f"📁 所有图片保存目录：{output_dir}")
print("\n💡 若仍有个别无效词，直接在STOP_WORDS中添加，重新运行即可过滤")