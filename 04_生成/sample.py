"""
W7 任务：采样生成

理解三个旋钮：
    temperature → 缩放 logits：T<1 更确定，T>1 更随机
    top_k       → 只保留概率最高的 k 个候选（硬截断）
    top_p       → 核采样：保留累积概率达到 p 的最小候选集（自适应，更自然）

对比实验建议（同一 prompt，各生成 5 次，记录重复率与通顺度）：
    T=0.3 / T=0.8 / T=1.2
    top_k=1（贪心）/ top_k=20 / top_k=50
    观察：T 过低 → 复读机；T 过高 → 语无伦次。这个手感对后面做 Agent 很关键。
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "02_模型"))
from model import GPT, GPTConfig   # noqa: E402


@torch.no_grad()
def generate(model: GPT, idx: torch.Tensor, max_new_tokens: int,
             temperature: float = 1.0, top_k: int | None = None,
             top_p: float | None = None) -> torch.Tensor:
    """
    idx: (B, T) 起始 token 序列
    """
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / temperature          # (B, vocab)

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        if top_p is not None and 0 < top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # 移除累积概率已超过 top_p 的 token（保留第一个越界的位置以保底）
            remove = cumprobs - F.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[remove] = -float("Inf")
            logits.scatter_(1, sorted_idx, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(os.path.dirname(__file__), "..", "ckpt", "ckpt.pt"))
    p.add_argument("--device", default="mps")
    p.add_argument("--max_new_tokens", type=int, default=300)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--top_p", type=float, default=None)
    p.add_argument("--prompt", default="ROMEO:")
    args = p.parse_args()

    if args.device == "mps" and not torch.backends.mps.is_available():
        args.device = "cpu"

    # PyTorch ≥2.6 起 torch.load 默认 weights_only=True，自定义 dataclass（GPTConfig）
    # 不在默认白名单里，会抛 UnpicklingError。正确做法是把它「加进安全白名单」，
    # 而不是图省事设置 weights_only=False（那等于允许 checkpoint 执行任意代码）。
    torch.serialization.add_safe_globals([GPTConfig])
    ck = torch.load(args.ckpt, map_location=args.device, weights_only=True)
    model = GPT(ck["config"]).to(args.device)
    model.load_state_dict(ck["model"])
    stoi, itos = ck["stoi"], {i: c for c, i in ck["stoi"].items()}

    ids = torch.tensor([[stoi.get(c, 0) for c in args.prompt]], dtype=torch.long, device=args.device)
    out = generate(model, ids, args.max_new_tokens, args.temperature, args.top_k, args.top_p)
    print("".join(itos[int(i)] for i in out[0].tolist()))


if __name__ == "__main__":
    main()
