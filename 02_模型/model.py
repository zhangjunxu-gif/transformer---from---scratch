"""
完整的 GPT 语言模型定义 —— 逐行中文跟写版
（结构对齐 karpathy/nanoGPT，注释与形状标注为本人重写）

跟写顺序建议：
    W2  CausalSelfAttention
    W3  LayerNorm / MLP / Block（Pre-LN 残差）
    W4  GPT 装配 / 权重绑定 / 初始化 / generate

阅读方法：每读一个 forward，先在纸上写出输入输出形状，再对照代码里的注释。
"""

from __future__ import annotations

import inspect
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


# ===========================================================================
# 组件一：LayerNorm（自己实现，因为 PyTorch 的 LayerNorm 不支持 bias=False）
# ===========================================================================
class LayerNorm(nn.Module):
    """带可选偏置的 LayerNorm。GPT-2 用的是有 bias 的版本。"""

    def __init__(self, ndim: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))    # 缩放 γ，初始化为 1
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None   # 平移 β

    def forward(self, x):
        # 对最后一个维度做归一化：y = (x - E[x]) / sqrt(Var[x] + eps) * γ + β
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


# ===========================================================================
# 组件二：因果自注意力（详见 attention_from_scratch.py，此处为装配版）
# ===========================================================================
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd 必须能被 n_head 整除"

        # Q/K/V 三份投影合并成一个 Linear：一次矩阵乘产出 3 * n_embd，再 split
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # 注意力输出投影回 n_embd
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)     # 作用在注意力权重上
        self.resid_dropout = nn.Dropout(config.dropout)    # 作用在残差分支输出上

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # PyTorch >= 2.0 有 FlashAttention（scaled_dot_product_attention），
        # 它把 softmax 融合进 CUDA kernel，省掉 O(T^2) 显存。有就用，没有就退回手写。
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # 因果掩码 buffer：下三角为 1，上三角为 0
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x):
        B, T, C = x.size()   # batch 大小, 序列长度, 嵌入维度

        # --- 1) 一次算出 Q/K/V -------------------------------
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)     # 各 (B, T, C)

        # --- 2) 拆成多头：把 C 维切成 (nh, hs)，并把 nh 提到 batch 维 ---
        hs = C // self.n_head
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)      # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        # --- 3) 打分 -> 掩码 -> softmax -> 加权求和 -------------
        if self.flash:
            # is_causal=True 让底层直接生成因果掩码，不必传 attn_mask
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # 手写版：(B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            # 除以 sqrt(hs) 的必要性：q·k 的方差随 hs 线性增长，不缩放会让 softmax 饱和、梯度消失
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v                                        # (B, nh, T, hs)

        # --- 4) 合并多头：先 transpose 回 (B, T, nh, hs)，再 contiguous 才能 view ---
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # --- 5) 输出投影 + dropout -----------------------------
        y = self.resid_dropout(self.c_proj(y))
        return y


# ===========================================================================
# 组件三：MLP（前馈网络，4 倍扩张 + GELU）
# ===========================================================================
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)  # 升维
        self.gelu = nn.GELU()                                                      # GPT-2 用 GELU（平滑版 ReLU）
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)  # 降维
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)     # (B, T, 4C)
        x = self.gelu(x)
        x = self.c_proj(x)   # (B, T, C)
        x = self.dropout(x)
        return x


# ===========================================================================
# 组件四：Transformer Block（Pre-LN 结构）
# ===========================================================================
class Block(nn.Module):
    """
         x ──► LayerNorm ──► Attention ──► + ──► LayerNorm ──► MLP ──► + ──► out
         └────────────────────────────────┘   └──────────────────────┘

    注意这是 **Pre-LN**（归一化在子层之前），与原始论文 Post-LN 不同：
    Pre-LN 的残差通路是恒等映射，深层训练不需要 warmup 也能稳定，是现代 LLM 的默认选择。
    """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))   # 残差分支一：注意力
        x = x + self.mlp(self.ln_2(x))    # 残差分支二：前馈
        return x


# ===========================================================================
# 配置
# ===========================================================================
@dataclass
class GPTConfig:
    block_size: int = 1024     # 最大上下文长度
    vocab_size: int = 50304    # GPT-2 词表 50257，向上补齐到 64 的倍数以利显存对齐
    n_layer: int = 12          # Transformer 层数
    n_head: int = 12           # 注意力头数
    n_embd: int = 768          # 嵌入维度
    dropout: float = 0.0       # 预训练时通常 0.0
    bias: bool = True          # True 对齐 GPT-2；False 略快且略好


# ===========================================================================
# 主体：GPT
# ===========================================================================
class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),        # token 嵌入
            wpe=nn.Embedding(config.block_size, config.n_embd),        # 位置嵌入（可学习，非正弦）
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),           # 最后的 LayerNorm
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # 权重绑定：输出头与输入嵌入共享同一份权重。
        # 直觉：词向量既要「被读」也要「被预测」，共享能让两者语义对齐，同时省下 38M 参数。
        self.transformer.wte.weight = self.lm_head.weight

        # 参数初始化
        self.apply(self._init_weights)
        # 残差分支的输出投影用特殊缩放初始化：层数越深，残差累加的方差越大，
        # 因此标准差随 sqrt(2 * n_layer) 衰减，防止深层激活值爆炸（GPT-2 论文的做法）
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))

    # ---------------------------------------------------------------
    def get_num_params(self, non_embedding: bool = True) -> int:
        """参数量。默认不计位置嵌入（它不参与梯度更新的语义部分）"""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    # ---------------------------------------------------------------
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ---------------------------------------------------------------
    def forward(self, idx, targets=None):
        """
        idx:      (B, T) 整数 token id
        targets:  (B, T) 目标 token id，训练时传入
        返回:     logits (B, T, vocab_size)，loss（或 None）
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"序列长度 {t} 超过 block_size {self.config.block_size}"
        )
        pos = torch.arange(0, t, dtype=torch.long, device=device)   # (t)

        tok_emb = self.transformer.wte(idx)                          # (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)                          # (t, n_embd) 会广播
        x = self.transformer.drop(tok_emb + pos_emb)                 # 嵌入相加即「位置感知的 token」

        for block in self.transformer.h:                             # 逐层前向
            x = block(x)
        x = self.transformer.ln_f(x)                                 # 最终归一化

        if targets is not None:
            # 训练：对全部位置算交叉熵（next-token prediction）
            logits = self.lm_head(x)                                 # (b, t, vocab_size)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # 推理优化：只需要最后一个位置的 logits，省掉 t-1 次无用的 lm_head 计算
            logits = self.lm_head(x[:, [-1], :])                     # 注意 [-1] 保留时间维
            loss = None

        return logits, loss

    # ---------------------------------------------------------------
    def crop_block_size(self, block_size: int):
        """模型手术：缩小上下文长度（加载预训练权重但想用更短窗口时）"""
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    # ---------------------------------------------------------------
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        自回归采样。
          temperature < 1 → 分布更尖锐（保守）；> 1 → 更平坦（发散）
          top_k：只在概率最高的 k 个词里采样，避免长尾垃圾 token
        """
        for _ in range(max_new_tokens):
            # 上下文超过 block_size 时截断，只保留最后 block_size 个 token
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)                 # (B, T, vocab)
            logits = logits[:, -1, :] / temperature    # 只取最后一步并缩放
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")   # 低于第 k 名的一律屏蔽
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # 按概率抽样
            idx = torch.cat((idx, idx_next), dim=1)             # 拼回序列继续
        return idx

    # ---------------------------------------------------------------
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        AdamW 的参数分组：
          所有 2D 及以上（矩阵权重、嵌入）→ 施加 weight decay
          所有 1D（bias、LayerNorm 的 γ/β）→ 不加
        理由：对 bias / 归一化参数做 L2 惩罚会损害模型表达，这是 GPT 训练的标准做法。
        """
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay = sum(p.numel() for p in decay_params)
        num_nodecay = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay:,} parameters")

        # fused AdamW 把整个优化步骤融合进一个 kernel，GPU 上明显更快
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

    # ---------------------------------------------------------------
    def estimate_mfu(self, fwdbwd_per_iter: float, dt: float) -> float:
        """模型算力利用率（MFU），以 A100 bfloat16 峰值 312 TFLOPS 为基准"""
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T        # 6N 前向+反向，加上注意力的 12LHQT
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0 / dt)        # 每秒浮点运算数
        flops_promised = 312e12
        return flops_achieved / flops_promised
