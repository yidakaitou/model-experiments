# -*- coding: utf-8 -*-
"""P2 实验 v2：标签必须由「文本 × 状态」共同决定（XOR）。

- 文本：50% 正面 / 50% 负面（平衡）
- 状态 p：U(0,1)，50% > 0.5（平衡）
- 标签 = (文本是正面) XOR (p > 0.5)     ← 单看任一方都只有 50%

这样：
  text   只用文本 → 天花板 50%
  state  只用状态 → 天花板 50%
  concat 拼在一起 → 要学交互，看能做到多少
  film   状态调制每层 → 若 > concat，说明"参与计算"有独特价值

指标用平衡准确率（正负类各 50%，直接准确率也能用）。
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

MODEL_DIR = os.environ.get("MODEL_DIR", r"D:\dsh-lab\models\bert-base-uncased")
DATA = Path(os.environ.get("DATA", r"D:\dsh-lab\emotion-dl\goemo_train.parquet"))
OUT = Path(os.environ.get("OUT", r"D:\dsh-lab\data\state_film"))
MAX_LEN = 48
BATCH = 32
LR = 2e-5
EPOCHS = int(os.environ.get("EPOCHS", 8))
GATE_LR = float(os.environ.get("GATE_LR", 1e-2))
ONLY = os.environ.get("ONLY", "")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
POS_LABELS = {0, 1, 17}


def make_data(n=12000, seed=0):
    tr = pd.read_parquet(DATA)
    rng = np.random.RandomState(seed)
    pos = tr[tr["label"].isin(POS_LABELS)]["text"].dropna().tolist()
    neg = tr[~tr["label"].isin(POS_LABELS)]["text"].dropna().tolist()
    rng.shuffle(pos)
    rng.shuffle(neg)
    half = n // 2
    X = pos[:half] + neg[:half]
    is_pos = np.array([1] * len(pos[:half]) + [0] * len(neg[:half]))
    p = rng.uniform(0, 1, size=len(X)).astype(np.float32)
    y = ((is_pos == 1) ^ (p > 0.5)).astype(np.int64)          # XOR
    idx = rng.permutation(len(X))
    X = [X[i] for i in idx]
    p, y = p[idx], y[idx]
    cut = int(0.8 * len(X))
    return (X[:cut], p[:cut], y[:cut]), (X[cut:], p[cut:], y[cut:])


class DS(Dataset):
    def __init__(self, texts, ps, ys, tok):
        self.t, self.p, self.y, self.tok = texts, ps, ys, tok

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        e = self.tok(self.t[i], truncation=True, max_length=MAX_LEN,
                     padding="max_length", return_tensors="pt")
        return ({k: v.squeeze(0) for k, v in e.items()},
                torch.tensor(float(self.p[i])), torch.tensor(int(self.y[i])))


class TextOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(MODEL_DIR)
        self.head = nn.Linear(self.enc.config.hidden_size, 2)

    def forward(self, ids, mask, state):
        return self.head(self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0])


class StateOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(self, ids, mask, state):
        return self.net(state.unsqueeze(1))


class Concat(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(MODEL_DIR)
        self.head = nn.Linear(self.enc.config.hidden_size + 1, 2)

    def forward(self, ids, mask, state):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        return self.head(torch.cat([h, state.unsqueeze(1)], dim=1))


class FiLM(nn.Module):
    def __init__(self, n_mod=4):
        super().__init__()
        base = AutoModel.from_pretrained(MODEL_DIR)
        self.enc = base
        d = base.config.hidden_size
        self.n_mod = n_mod
        self.to_gb = nn.ModuleList([nn.Linear(1, 2 * d) for _ in range(n_mod)])
        for m in self.to_gb:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        self.head = nn.Linear(d, 2)
        self._state = None
        self._layer = 0
        for lyr in list(self.enc.encoder.layer)[-n_mod:]:
            lyr.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        h = out[0]
        g, b = self.to_gb[self._layer](self._state.unsqueeze(1)).chunk(2, dim=1)
        g, b = g.unsqueeze(1), b.unsqueeze(1)
        self._layer = (self._layer + 1) % self.n_mod
        return ((1 + g) * h + b,) + out[1:]

    def forward(self, ids, mask, state):
        self._state = state
        self._layer = 0
        return self.head(self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0])


def run(kind, tr, te, tok):
    model = {"text": TextOnly, "state": StateOnly, "concat": Concat, "film": FiLM}[kind]().to(DEVICE)
    dl = DataLoader(DS(*tr, tok), batch_size=BATCH, shuffle=True)
    enc_p, gate_p, oth_p = [], [], []
    for n_, q in model.named_parameters():
        if n_.startswith("enc."):
            enc_p.append(q)
        elif n_.startswith("to_gb"):
            gate_p.append(q)
        else:
            oth_p.append(q)
    groups = ([{"params": enc_p, "lr": 2e-5}] if enc_p else [])
    if gate_p:
        groups.append({"params": gate_p, "lr": GATE_LR})   # 门控单独给大学习率
    groups.append({"params": oth_p, "lr": 1e-3})
    opt = torch.optim.AdamW(groups, lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    for ep in range(EPOCHS):
        model.train()
        tot = 0.0
        for batch, st, y in dl:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            st, y = st.to(DEVICE), y.to(DEVICE)
            loss = lossf(model(batch["input_ids"], batch["attention_mask"], st), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
        print(f"  [{kind}] ep{ep+1} loss {tot/len(dl.dataset):.4f}", flush=True)

    model.eval()
    dl2 = DataLoader(DS(*te, tok), batch_size=64)
    c = t = 0
    tp = fp = fn = 0
    with torch.no_grad():
        for batch, st, y in dl2:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            pred = model(batch["input_ids"], batch["attention_mask"], st.to(DEVICE)).argmax(-1).cpu()
            c += (pred == y).sum().item()
            t += len(y)
            tp += int(((pred == 1) & (y == 1)).sum())
            fp += int(((pred == 1) & (y == 0)).sum())
            fn += int(((pred == 0) & (y == 1)).sum())
    acc = c / t
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return round(acc, 4), round(f1, 4)


def main():
    seed = int(os.environ.get("SEED", 0))
    torch.manual_seed(seed)
    np.random.seed(seed)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    tr, te = make_data(seed=seed)
    print(f"训练 {len(tr[0])} / 测试 {len(te[0])} | 正例比例 {tr[2].mean():.2f}", flush=True)

    res = {}
    t0 = time.time()
    conds = ONLY.split(",") if ONLY else ["text", "state", "concat", "film"]
    for kind in conds:
        print(f"\n=== {kind} ===", flush=True)
        acc, f1 = run(kind, tr, te, tok)
        res[kind] = {"acc": acc, "f1": f1}
        print(f"  {kind}: acc {acc}  f1 {f1}", flush=True)

    res["minutes"] = round((time.time() - t0) / 60, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"state_xor_s{seed}.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n" + json.dumps(res, ensure_ascii=False, indent=1))
    print("saved", p)


if __name__ == "__main__":
    main()
