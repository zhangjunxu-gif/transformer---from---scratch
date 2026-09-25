"""
W5 任务：自己写分词器

两个层级，从易到难：
  1) CharTokenizer   —— 字符级，10 分钟写完，用于先把训练管线跑通
  2) SimpleBPE       —— 简易字节对编码，理解 GPT-2 真正的分词原理

为什么要自己写：
    tokenization 决定了「模型看到的世界有多碎」。中英文切分差异直接影响
    token 数与训练成本，也决定你对 context length 的直觉。
"""

from __future__ import annotations

from collections import Counter
from typing import Iterable


class CharTokenizer:
    """字符级分词器：最小可用实现"""

    def __init__(self, text: str):
        chars = sorted(list(set(text)))
        self.stoi = {c: i for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos[i] for i in ids)


class SimpleBPE:
    """
    BPE 的核心一句话：**反复把出现频次最高的相邻 token 对合并成新 token**。

    训练循环：
        for _ in range(num_merges):
            统计所有相邻对的出现次数
            把次数最多的那对 (a, b) 合并成新符号 ab
            在所有序列中替换

    GPT-2 实际用 byte-level BPE（先把 UTF-8 转成字节，保证任何字符都能表示），
    本实现为教学版，直接在字符上做。
    """

    def __init__(self, num_merges: int = 200):
        self.num_merges = num_merges
        self.merges: dict[tuple[str, str], str] = {}
        self.vocab: list[str] = []

    def train(self, text: str) -> None:
        words = text.split()
        seqs = [list(w) + ["</w>"] for w in words]   # </w> 防止跨词合并
        self.vocab = sorted(set("".join("".join(s) for s in seqs)))

        for _ in range(self.num_merges):
            pairs: Counter = Counter()
            for seq in seqs:
                for a, b in zip(seq, seq[1:]):
                    pairs[(a, b)] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            if pairs[best] < 2:
                break
            merged = best[0] + best[1]
            self.merges[best] = merged
            self.vocab.append(merged)

            new_seqs = []
            for seq in seqs:
                out, i = [], 0
                while i < len(seq):
                    if i < len(seq) - 1 and (seq[i], seq[i + 1]) == best:
                        out.append(merged)
                        i += 2
                    else:
                        out.append(seq[i])
                        i += 1
                new_seqs.append(out)
            seqs = new_seqs

        self.vocab = sorted(set(self.vocab))
        self.stoi = {t: i for i, t in enumerate(self.vocab)}
        self.itos = {i: t for t, i in self.stoi.items()}

    def encode_word(self, word: str) -> list[str]:
        """按合并规则的习得顺序（rank）贪心应用"""
        merge_order = list(self.merges.keys())
        seq = list(word) + ["</w>"]
        while True:
            best_pair, best_rank = None, None
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                if pair in self.merges:
                    rank = merge_order.index(pair)
                    if best_rank is None or rank < best_rank:
                        best_pair, best_rank = pair, rank
            if best_pair is None:
                break
            merged = self.merges[best_pair]
            out, i = [], 0
            while i < len(seq):
                if i < len(seq) - 1 and (seq[i], seq[i + 1]) == best_pair:
                    out.append(merged)
                    i += 2
                else:
                    out.append(seq[i])
                    i += 1
            seq = out
        return seq

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for w in text.split():
            for tok in self.encode_word(w):
                ids.append(self.stoi.get(tok, 0))
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        toks = [self.itos[i] for i in ids]
        return "".join(toks).replace("</w>", " ").strip()


if __name__ == "__main__":
    demo = "the quick brown fox jumps over the lazy dog the dog barks the fox runs"
    tok = SimpleBPE(num_merges=30)
    tok.train(demo)
    print("词表大小:", len(tok.vocab))
    print("学到的合并（前 10）:", list(tok.merges.items())[:10])
    ids = tok.encode("the quick brown fox")
    print("编码:", ids)
    print("解码:", tok.decode(ids))

    c = CharTokenizer(demo)
    print("\n字符级词表:", c.vocab_size, "| roundtrip:", c.decode(c.encode("hello")) == "hello")
