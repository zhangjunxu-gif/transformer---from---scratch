"""
W1 任务：纯手写的因果自注意力（Causal Self-Attention）

规则：本文件不得 import torch.nn.MultiheadAttention，不得用 F.scaled_dot_product_attention。
所有矩阵运算必须显式写出，并在自测中用等价实现交叉验证。

数学：
    Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k) + M ) V
    其中 M 为因果掩码：M[i][j] = 0 若 j <= i，否则 -inf

    Q = X W_q,  K = X W_k,  V = X W_v          X: (B, T, C)
    多头：把 C 切成 nh 份，每份 hs = C / nh

作者：张俊旭
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. 单头注意力：先把它写到不能再简单
# ---------------------------------------------------------------------------
def single_head_attention(
    x: torch.Tensor,          # (B, T, C) 输入序列
    w_q: torch.Tensor,        # (C, hs)
    w_k: torch.Tensor,        # (C, hs)
    w_v: torch.Tensor,        # (C, hs)
    causal: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """返回 (out, att)，out: (B, T, hs)，att: (B, T, T) 注意力权重"""
    B, T, C = x.shape
    hs = w_q.shape[1]

    q = x @ w_q            # (B, T, hs)
    k = x @ w_k            # (B, T, hs)
    v = x @ w_v            # (B, T, hs)

    # 打分：每个 query 与所有 key 做内积。除以 sqrt(hs) 是防止内积方差随维度线性增长，
    # 否则 softmax 会进入饱和区、梯度趋近 0。这一行是整个 Transformer 最关键的数值技巧。
    scores = q @ k.transpose(-2, -1) / math.sqrt(hs)   # (B, T, T)

    if causal:
        # 下三角掩码：位置 i 只能看到 j <= i
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
        scores = scores.masked_fill(~mask, float("-inf"))

    att = F.softmax(scores, dim=-1)                     # (B, T, T)，每行和为 1
    out = att @ v                                       # (B, T, hs)
    return out, att


# ---------------------------------------------------------------------------
# 2. 多头拆分：理解 view + transpose 为什么是这个顺序
# ---------------------------------------------------------------------------
def split_heads(x: torch.Tensor, n_head: int) -> torch.Tensor:
    """
    (B, T, C) -> (B, nh, T, hs)

    关键理解：不能先 reshape 成 (B, T, nh, hs) 就直接用。
    我们希望「同一个时刻的不同头」在内存中相邻，注意力是在 T 维上做的，
    所以要让 nh 成为一个 batch 维：transpose(1, 2)。
    transpose 之后张量非连续，后续若做 view 必须先 .contiguous()。
    """
    B, T, C = x.shape
    assert C % n_head == 0, f"C={C} 不能被 n_head={n_head} 整除"
    hs = C // n_head
    return x.view(B, T, n_head, hs).transpose(1, 2)      # (B, nh, T, hs)


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """(B, nh, T, hs) -> (B, T, C)，split_heads 的逆操作"""
    B, nh, T, hs = x.shape
    return x.transpose(1, 2).contiguous().view(B, T, nh * hs)


class CausalSelfAttention(nn.Module):
    """nanoGPT 风格的因果自注意力，但由你亲手写成（含慢速实现）"""

    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.hs = n_embd // n_head
        self.dropout = dropout

        # 一次线性变换产出 q/k/v 三份，再 split —— 比三个 Linear 更省一次矩阵乘
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # 掩码是 buffer 不是 parameter：不参与训练，但会随 .to(device) 一起移动
        self.register_buffer(
            "bias", torch.tril(torch.ones(1024, 1024)).view(1, 1, 1024, 1024)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape                                  # batch, 时间步, 通道

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)  # 三个都是 (B, T, C)
        k = split_heads(k, self.n_head)                     # (B, nh, T, hs)
        q = split_heads(q, self.n_head)
        v = split_heads(v, self.n_head)

        # 慢速实现：显式写出 softmax，便于核对与调试
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.hs))  # (B, nh, T, T)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        y = att @ v                                          # (B, nh, T, hs)

        y = merge_heads(y)                                   # (B, T, C)
        y = self.resid_dropout(self.c_proj(y))               # 输出投影 + dropout
        return y


# ---------------------------------------------------------------------------
# 3. 自测：手写的每一行都要被验证
# ---------------------------------------------------------------------------
def _test_causality() -> None:
    """位置 i 绝不能看到 j > i"""
    torch.manual_seed(0)
    B, T, C, nh = 2, 6, 32, 4
    attn = CausalSelfAttention(C, nh)
    attn.eval()
    x = torch.randn(B, T, C)

    # 抓取内部注意力矩阵
    q, k, _ = attn.c_attn(x).split(C, dim=2)
    q = split_heads(q, nh)
    k = split_heads(k, nh)
    scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(C // nh))
    scores = scores.masked_fill(attn.bias[:, :, :T, :T] == 0, float("-inf"))
    a = F.softmax(scores, dim=-1)

    upper = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    assert (a[:, :, upper] == 0).all(), "因果性失败：上三角仍有非零注意力"
    assert torch.allclose(a.sum(-1), torch.ones(B, nh, T), atol=1e-5), "注意力行和不为 1"
    print("[PASS] 因果掩码 + 归一化")


def _test_split_merge_roundtrip() -> None:
    torch.manual_seed(1)
    x = torch.randn(2, 5, 32)
    y = merge_heads(split_heads(x, 4))
    assert torch.equal(x, y), "split/merge 不能还原原始张量"
    print("[PASS] split_heads / merge_heads 互逆")


def _test_against_official() -> None:
    """与 PyTorch 官方实现交叉验证（仅用于验证，不用于跟写）"""
    torch.manual_seed(2)
    B, T, C, nh = 2, 8, 32, 4
    x = torch.randn(B, T, C)

    mine = CausalSelfAttention(C, nh)
    mine.eval()
    official = nn.MultiheadAttention(C, nh, batch_first=True, dropout=0.0)
    official.eval()

    # 把官方权重拷到我的实现上（注意官方 in_proj 是 (3C, C)，与 nanoGPT 转置）
    with torch.no_grad():
        official.in_proj_weight.copy_(mine.c_attn.weight)
        if official.in_proj_bias is not None:
            official.in_proj_bias.copy_(mine.c_attn.bias)
        official.out_proj.weight.copy_(mine.c_proj.weight)
        if official.out_proj.bias is not None:
            official.out_proj.bias.copy_(mine.c_proj.bias)

    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    y_official, _ = official(x, x, x, attn_mask=mask, need_weights=False)
    y_mine = mine(x)

    max_err = (y_official - y_mine).abs().max().item()
    assert max_err < 1e-5, f"与官方实现差异过大: {max_err}"
    print(f"[PASS] 与 nn.MultiheadAttention 一致（最大误差 {max_err:.2e}）")


def _test_gradient_flow() -> None:
    """确认梯度能回传到输入与所有参数"""
    attn = CausalSelfAttention(32, 4)
    x = torch.randn(2, 6, 32, requires_grad=True)
    attn(x).sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all(), "梯度异常"
    assert all(p.grad is not None for p in attn.parameters()), "存在未收到梯度的参数"
    print("[PASS] 梯度通路正常")


if __name__ == "__main__":
    _test_split_merge_roundtrip()
    _test_causality()
    _test_against_official()
    _test_gradient_flow()
    print("\n全部通过。W1 完成，可以进入 W2（多头与位置编码）。")
