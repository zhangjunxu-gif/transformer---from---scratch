# Transformer 从零复现：nanoGPT 逐行跟写

> 张俊旭 · AI 转行能力建设项目 · 仓库起点 2026-09-25
> 目标：不是「跑通 nanoGPT」，而是**逐行读懂并亲手重写**每一个张量变换，
> 最终能向任何一位计算机系教授解释清楚：为什么 `q @ k.T / sqrt(d)` 这一行就是 Transformer 的全部核心。

---

## 0. 为什么是 nanoGPT，而不是别的

| 候选 | 代码量 | 问题 |
|---|---|---|
| HuggingFace Transformers | 数万行 | 抽象层太厚，读不到数学 |
| 原始 `Attention Is All You Need` 复现 | 中等 | 多为编码器-解码器，与现代 LLM 不同 |
| **karpathy/nanoGPT** | **约 600 行核心** | 单一文件定义完整 GPT，无抽象遮掩，且可直接训出能说话的模型 |

nanoGPT 的 `model.py` 只有 331 行，却完整包含：
因果自注意力、多头拆分、位置嵌入、残差 + LayerNorm 前置、MLP、权重绑定、AdamW 参数分组、MFU 估算、采样生成。
**这是目前世界上「性价比最高的 Transformer 教材」。**

---

## 1. 仓库结构（跟写路线图）

```
01_nanoGPT跟写/
├── README.md                     ← 本文件：总纲与验收标准
├── requirements.txt
├── 00_笔记/
│   ├── 01_张量形状速查.md          ← 每次卡住先查这里
│   ├── 02_注意力数学推导.md         ← 从 (B,T,C) 到 (B,nh,T,hs) 的每一步
│   └── 03_每周跟写日志模板.md
├── 01_词元化/
│   └── tokenizer.py              ← 字符级 + 简易 BPE，自己实现
├── 02_模型/
│   ├── attention_from_scratch.py  ★ 第一周写这个：纯手写注意力 + 自测
│   └── model.py                  ★ 第二至四周：逐行重写 nanoGPT，中文注释
├── 03_训练/
│   └── train.py                  ★ 第五至六周：训练循环、余弦退火、梯度裁剪
├── 04_生成/
│   └── sample.py                 ★ 第七周：温度 / top-k 采样
├── verify_all.py                 ★ W1–W5 一键验收（不训练，CPU 30 秒跑完）
├── data/                          ← 语料（.gitignore 排除）
├── ckpt/                          ← 权重（.gitignore 排除）
└── _reference_nanoGPT/            ← 官方源码（已在 .gitignore 中排除）
```

---

## 2. 八周跟写计划（每周 6 小时，与 CS229 / 凸优化并行）

| 周次 | 主题 | 你要亲手写出什么 | 验收标准（可对外展示） |
|---|---|---|---|
| W1 | 注意力的数学 | `attention_from_scratch.py`：不用 `nn.MultiheadAttention`，纯矩阵手写 | 自测脚本全绿；能口述 QKV 三行公式 |
| W2 | 多头与因果掩码 | 手写 `split_heads` / `merge_heads` / 下三角掩码 | 证明第 t 个位置看不到 t+1 之后 |
| W3 | 位置编码 + 残差 + LN | `model.py` 的 `Block` 部分 | 解释 Pre-LN 与 Post-LN 的梯度差异 |
| W4 | 完整 GPT 类 | 装配 `GPT` + 权重绑定 + 初始化 | 打印参数量 ≈ 124M（GPT-2 small 配置） |
| W5 | 数据管线与分词 | `tokenizer.py` + 数据集切分 | 用莎翁数据集训出可读文本 |
| W6 | 训练循环 | `train.py`：AdamW 分组、余弦退火、Grad Clip | loss 曲线正常下降并保存 checkpoint |
| W7 | 采样与评估 | `sample.py`：温度 + top-k | 生成 200 token 通顺样本 |
| W8 | 复现实验与写作 | README 补齐 + 一篇中文技术博客 | 博客可作为套磁附件 |

### 当前进度（2026-09-27 实测）

| 周次 | 状态 | 实测证据 |
|---|---|---|
| W1 | ✅ | `attention_from_scratch.py` 自测 4 项全绿 |
| W2 | ✅ | `verify_all.py`：扰动未来 token，输出偏差 0.00e+00 |
| W3 | ✅ | Pre-LN 生效；与 Post-LN 的首层梯度范数对比 1.22e-01 vs 3.84e-07 |
| W4 | ✅ | GPT-2 small **123,689,472** 参数；`wte` 与 `lm_head` 权重绑定 |
| W5 | ✅ | BPE / 字符级分词 encode-decode 往返一致 |
| W6 | ✅ | CPU 小模型 loss 4.12 → 2.10（500 iters） |
| W7 | ✅ | 采样 200 token；T=0.3/top_k=1 出现预期的「复读机」现象 |
| W8 | ⏳ | 待写 |

一键复验：

```bash
python verify_all.py                      # W1–W5，约 30 秒
python -u 03_训练/train.py --device cpu --n_layer 4 --n_head 4 --n_embd 128 \
    --block_size 64 --batch_size 16 --max_iters 500 --eval_interval 100   # W6
python 04_生成/sample.py --device cpu --max_new_tokens 200 --temperature 0.8 --top_k 20
```

> 环境提示：`raw.githubusercontent.com` 在国内常超时，语料下载已改为
> **jsdelivr CDN 优先 + 原站兜底 + 内置语料保底**（见 `train.py` 的 `DATA_URLS`）。

---

## 3. 环境

```bash
pip install -r requirements.txt
```

本机为 Apple Silicon（arm64），PyTorch 走 MPS 后端；无 GPU 时用 CPU 也能跑通 W1-W4（把 `n_layer=4, n_embd=128` 调小即可）。

```python
device = "mps" if torch.backends.mps.is_available() else "cpu"
```

---

## 4. 铁律（跟写期间不可违反）

1. **先自己写，再看官方答案。** 每个文件先凭记忆/推导写一遍，卡住超过 20 分钟才准看 `_reference_nanoGPT/`。
2. **每一个张量都要写形状注释。** 例如 `q = ...  # (B, nh, T, hs)`。写不出形状注释 = 没读懂。
3. **每天至少一个 git commit。** 仓库的 commit 密度本身就是申请材料的证据链。
4. **不允许 `pip install transformers` 走捷径。** 分词器自己写，注意力自己写。

---

## 5. 这个仓库在申请中的用法

- 套磁信附件：链接到本仓库 README + W8 的技术博客。
- 面试/复试被问「你转 AI 做了什么」时：**从零手写过完整 GPT 并能解释每一行** —— 这是最有说服力的回答。
- RP 方向对接：W8 之后可把「Agent 的市场机制设计」的实验代码直接放在这个仓库里继续长。
