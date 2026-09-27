"""
W6 任务：训练循环（简化可跑版，Apple Silicon 用 MPS，无 GPU 也能跑）

跟写要点：
  1. 数据如何切 batch（随机起点 + 固定 block_size）
  2. 学习率为什么必须 warmup + 余弦退火
  3. 梯度裁剪为什么放在 unscale 之后
  4. 什么时候算 loss（每 N 步估一次，避免频繁同步拖慢训练）

最小可跑（CPU，约 1 分钟出结果）：
    python train.py --device cpu --n_layer 4 --n_head 4 --n_embd 128 --max_iters 200
"""

from __future__ import annotations

import argparse
import math
import os
import time
import urllib.request

import torch
import torch.nn as nn

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "02_模型"))
from model import GPT, GPTConfig   # noqa: E402


# ---------------------------------------------------------------------------
# 1. 数据：默认用 nanoGPT 同款莎士比亚数据集
# ---------------------------------------------------------------------------
# 多个镜像按顺序尝试：raw.githubusercontent.com 在国内经常直接超时，
# jsdelivr（GitHub 文件的 CDN）通常可达。全部失败时用内置兜底语料，
# 保证「训练管线本身」永远能在离线环境下被验证。
DATA_URLS = [
    "https://cdn.jsdelivr.net/gh/karpathy/char-rnn@master/data/tinyshakespeare/input.txt",
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
]
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "tinyshakespeare.txt")
DOWNLOAD_TIMEOUT = 20

# 离线兜底语料：结构极简单但「可被学习」（有重复的模式与词法），
# 足以证明 loss 在下降、采样能出词。注意它不能替代真实语料做效果评估。
FALLBACK_CORPUS = (
    "the king and the queen are in the hall . "
    "the king loves the queen and the queen loves the king . "
    "to be or not to be , that is the question . "
    "romeo loves juliet and juliet loves romeo . "
    "the fox runs and the dog barks and the bird sings . "
) * 400


def get_data() -> str:
    if os.path.exists(DATA_PATH) and os.path.getsize(DATA_PATH) > 1000:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            return f.read()

    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    for url in DATA_URLS:
        try:
            print(f"下载语料：{url}")
            with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT) as r:
                text = r.read().decode("utf-8")
            if len(text) < 1000:
                raise ValueError(f"内容过短（{len(text)} 字节），疑似无效响应")
            with open(DATA_PATH, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"已保存 {len(text):,} 字符 → {DATA_PATH}")
            return text
        except Exception as e:                       # 超时 / DNS / 代理失败都算
            print(f"  失败（{type(e).__name__}: {e}），换下一个镜像")

    print("[警告] 所有镜像均不可达，改用内置兜底语料（仅用于验证管线，不代表真实效果）")
    return FALLBACK_CORPUS


class CharDataset:
    """字符级数据集：最简单，但足以验证训练管线是否真的在学习"""

    def __init__(self, text: str, block_size: int):
        chars = sorted(list(set(text)))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars)
        data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(0.9 * len(data))
        self.train = data[:n]
        self.val = data[n:]
        self.block_size = block_size

    def get_batch(self, split: str, batch_size: int, device: str):
        src = self.train if split == "train" else self.val
        # 随机起点：每个样本是长度 block_size 的连续片段
        ix = torch.randint(len(src) - self.block_size, (batch_size,))
        x = torch.stack([src[i:i + self.block_size] for i in ix])
        # 目标就是 x 向右平移一位 —— 这就是「预测下一个 token」的全部定义
        y = torch.stack([src[i + 1:i + 1 + self.block_size] for i in ix])
        return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# 2. 学习率调度：warmup + 余弦退火
# ---------------------------------------------------------------------------
def get_lr(it: int, warmup_iters: int, lr_decay_iters: int, learning_rate: float, min_lr: float) -> float:
    """
    1) 线性 warmup：训练初期 loss 曲面陡峭，大 lr 会直接发散
    2) 余弦退火到 min_lr：后期需要小步长收敛到更平坦的极小值
    """
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


@torch.no_grad()
def estimate_loss(model, ds: CharDataset, batch_size: int, device: str, eval_iters: int = 20):
    """在 train/val 上各估若干次 loss，取平均（必须开 eval 模式关掉 dropout）"""
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x, y = ds.get_batch(split, batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ---------------------------------------------------------------------------
# 3. 主训练循环
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="mps")
    p.add_argument("--batch_size", type=int, default=12)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--n_layer", type=int, default=6)
    p.add_argument("--n_head", type=int, default=6)
    p.add_argument("--n_embd", type=int, default=192)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--max_iters", type=int, default=2000)
    p.add_argument("--learning_rate", type=float, default=3e-3)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_iters", type=int, default=100)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--out_dir", default=os.path.join(os.path.dirname(__file__), "..", "ckpt"))
    args = p.parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        args.device = "cpu"
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
    print(f"device = {args.device}")

    torch.manual_seed(1337)
    text = get_data()
    ds = CharDataset(text, args.block_size)
    print(f"数据集：{len(text):,} 字符，词表 {ds.vocab_size}")

    config = GPTConfig(
        block_size=args.block_size,
        vocab_size=ds.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
        bias=False,          # 用 bias=False：略快略好（现代实现默认）
    )
    model = GPT(config).to(args.device)

    optimizer = model.configure_optimizers(
        weight_decay=args.weight_decay,
        learning_rate=args.learning_rate,
        betas=(0.9, 0.95),
        device_type=args.device,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    t0 = time.time()

    for it in range(args.max_iters):
        lr = get_lr(it, args.warmup_iters, args.max_iters, args.learning_rate, args.learning_rate / 10)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = ds.get_batch("train", args.batch_size, args.device)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # 梯度裁剪：RNN 时代必备；Transformer 里主要是防止训练初期 loss 尖峰把参数打飞
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if it % args.eval_interval == 0 or it == args.max_iters - 1:
            losses = estimate_loss(model, ds, args.batch_size, args.device)
            print(
                f"iter {it:5d} | lr {lr:.2e} | "
                f"train {losses['train']:.4f} | val {losses['val']:.4f} | "
                f"{time.time() - t0:.1f}s"
            )

    ckpt = os.path.join(args.out_dir, "ckpt.pt")
    torch.save({"model": model.state_dict(), "config": config, "stoi": ds.stoi}, ckpt)
    print(f"已保存 checkpoint → {ckpt}")

    # 训完立刻生成一段，肉眼验证模型确实学到了东西
    model.eval()
    ctx = torch.zeros((1, 1), dtype=torch.long, device=args.device)
    out = model.generate(ctx, max_new_tokens=200, temperature=0.8, top_k=20)
    itos = ds.itos
    print("\n--- 生成样本 ---")
    print("".join(itos[int(i)] for i in out[0].tolist()))


if __name__ == "__main__":
    main()
