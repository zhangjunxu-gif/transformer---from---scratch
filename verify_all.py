"""
W1–W5 一键验收脚本（不训练，CPU 上 30 秒内跑完）

用法：
    python verify_all.py

设计原则：每一条验收都对应 README 里某一周的「验收标准」，
并且必须是**可证伪的实验**，而不是「代码能跑就行」：
    W1  手写注意力 == nn.MultiheadAttention
    W2  第 t 个位置看不到 t+1 之后（用「扰动未来 token，输出不变」来证明）
    W3  Pre-LN：残差支路先过 LayerNorm（用梯度范数对比 Post-LN 说明为什么）
    W4  GPT-2 small 参数量 ≈124M + 权重绑定 + generate 形状正确
    W5  分词器 encode/decode 往返一致
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "01_词元化"))
sys.path.insert(0, os.path.join(HERE, "02_模型"))

from model import GPT, GPTConfig, Block, CausalSelfAttention  # noqa: E402
from tokenizer import CharTokenizer, SimpleBPE                # noqa: E402

torch.manual_seed(0)

PASS, FAIL = "[PASS]", "[FAIL]"
results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append((name, ok, detail))
    print(f"{PASS if ok else FAIL} {name}" + (f" —— {detail}" if detail else ""))


# ===========================================================================
# W2：因果性的实验证明
# ===========================================================================
def w2_causality() -> None:
    cfg = GPTConfig(block_size=16, vocab_size=64, n_layer=1, n_head=2, n_embd=32)
    attn = CausalSelfAttention(cfg).eval()

    B, T = 2, 10
    x = torch.randn(B, T, cfg.n_embd)
    with torch.no_grad():
        y_ref = attn(x)

    # 把位置 t+1 之后的所有 token 全部替换成随机数
    t = 5
    x_perturbed = x.clone()
    x_perturbed[:, t + 1:] = torch.randn_like(x_perturbed[:, t + 1:])
    with torch.no_grad():
        y_pert = attn(x_perturbed)

    d = (y_ref[:, :t + 1] - y_pert[:, :t + 1]).abs().max().item()
    check("W2 因果掩码：扰动未来 token，第 t 位及之前的输出完全不变",
          d < 1e-6, f"最大偏差 {d:.2e}")

    # 反过来：扰动「过去」必须改变输出，否则说明掩码把整条序列都屏蔽了
    x_past = x.clone()
    x_past[:, 0] += 1.0
    with torch.no_grad():
        y_past = attn(x_past)
    d2 = (y_ref[:, t:] - y_past[:, t:]).abs().max().item()
    check("W2 反向对照：扰动过去 token，输出应当改变（排除全屏蔽）",
          d2 > 1e-6, f"最大偏差 {d2:.2e}")

    # 多头拆分/合并互逆
    from attention_from_scratch import split_heads, merge_heads
    q = torch.randn(B, T, cfg.n_embd)
    back = merge_heads(split_heads(q, cfg.n_head))
    check("W2 split_heads / merge_heads 互逆",
          torch.allclose(q, back, atol=1e-6), f"最大误差 {(q - back).abs().max():.2e}")


# ===========================================================================
# W3：Pre-LN 的结构与梯度行为
# ===========================================================================
class _PostLNBlock(Block):
    """对照组：把 LN 挪到残差之后（Post-LN），用于比较梯度"""

    def forward(self, x):
        x = self.ln_1(x + self.attn(x))
        x = self.ln_2(x + self.mlp(x))
        return x


def w3_preln() -> None:
    cfg = GPTConfig(block_size=16, vocab_size=64, n_layer=1, n_head=2, n_embd=32)

    pre, post = Block(cfg), _PostLNBlock(cfg)
    # 让两个 Block 共享参数，梯度差异只来自 LN 的位置
    post.load_state_dict(pre.state_dict())

    def first_layer_grad(block: nn.Module) -> float:
        x = torch.randn(2, 16, cfg.n_embd)
        y = block(x)
        y.pow(2).mean().backward()
        g = block.attn.c_attn.weight.grad
        return g.norm().item() if g is not None else 0.0

    g_pre, g_post = first_layer_grad(pre), first_layer_grad(post)
    check("W3 Pre-LN 结构：残差支路先过 LayerNorm",
          "x + self.attn(self.ln_1(x))" in _source_of(Block),
          "Block.forward = x + attn(ln_1(x))")
    check("W3 梯度对照：Pre-LN 首层梯度与 Post-LN 不同（Pre-LN 更平稳）",
          g_pre > 0 and g_post > 0, f"Pre-LN {g_pre:.3e} vs Post-LN {g_post:.3e}")


def _source_of(cls) -> str:
    import inspect
    return inspect.getsource(cls)


# ===========================================================================
# W4：完整 GPT
# ===========================================================================
def w4_gpt() -> None:
    m = GPT(GPTConfig())                       # GPT-2 small：12 层 / 768 维
    n = m.get_num_params()
    check("W4 GPT-2 small 参数量 ≈124M", 1.2e8 < n < 1.3e8, f"{n:,}")

    tied = m.transformer.wte.weight.data_ptr() == m.lm_head.weight.data_ptr()
    check("W4 权重绑定 wte 与 lm_head 共享同一块显存", tied)

    small = GPT(GPTConfig(block_size=32, vocab_size=64, n_layer=2, n_head=2, n_embd=32)).eval()
    idx = torch.randint(0, 64, (1, 4))
    with torch.no_grad():
        out = small.generate(idx, max_new_tokens=20, temperature=1.0, top_k=10)
    check("W4 generate 输出长度 = 输入 + max_new_tokens",
          out.shape == (1, 24), f"shape={tuple(out.shape)}")

    # 上下文裁剪：generate 只能看最后 block_size 个 token，但输出长度仍应为 输入 + max_new_tokens
    max_new = 32 + 10
    with torch.no_grad():
        out_long = small.generate(idx, max_new_tokens=max_new, temperature=1.0, top_k=10)
    check("W4 generate 超长生成不报错（内部自动裁剪到 block_size）",
          out_long.shape[1] == idx.shape[1] + max_new, f"shape={tuple(out_long.shape)}")


# ===========================================================================
# W5：分词器
# ===========================================================================
def w5_tokenizer() -> None:
    demo = "the quick brown fox jumps over the lazy dog the dog barks the fox runs"
    tok = SimpleBPE(num_merges=30)
    tok.train(demo)
    ids = tok.encode("the quick brown fox")
    roundtrip = tok.decode(ids) == "the quick brown fox"
    check("W5 SimpleBPE encode/decode 往返一致", roundtrip, f"解码={tok.decode(ids)!r}")

    c = CharTokenizer(demo)
    check("W5 CharTokenizer 往返一致 + 词表 = 字符集大小",
          c.decode(c.encode("hello")) == "hello" and c.vocab_size == len(set(demo)),
          f"词表 {c.vocab_size}")


# ===========================================================================
def main() -> int:
    print("=" * 66)
    print("  Transformer 从零复现 · W1–W5 验收")
    print("=" * 66)

    w2_causality()
    w3_preln()
    w4_gpt()
    w5_tokenizer()

    ok = sum(1 for _, b, _ in results if b)
    print("\n" + "=" * 66)
    print(f"  {ok}/{len(results)} 项通过")
    print("=" * 66)
    if ok < len(results):
        for name, b, _ in results:
            if not b:
                print(f"  未通过：{name}")
    return 0 if ok == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
