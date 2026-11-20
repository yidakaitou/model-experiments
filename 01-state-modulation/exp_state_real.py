# -*- coding: utf-8 -*-
"""P2 真实任务：状态 = 用户偏好，文本单独看不出来。

任务：个性化情绪判定。
  - 三个用户，每人有一套「哪些情绪算好消息」的私人偏好（规则不同、互不重叠）
  - 输入：一句话 + 这句话是谁说的（用户身份，模型看不到内容之外的任何东西）
  - 标签：这句话的情绪是否属于这个用户的「好消息」集合

关键：**同一句话，不同用户标签可能相反**。所以只看文本无法判断，
必须把「是谁说的」这个状态用起来。

三个条件：
  text     只用文本（状态丢弃）
  concat   状态拼到 [CLS] 上
  film     状态调制每一层
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
OUT = Path(os.environ.get("OUT", r"D:\dsh-lab\data\state_real"))
MAX_LEN = 48
BATCH = 32
EPOCHS = int(os.environ.get("EPOCHS", 6))
GATE_LR = float(os.environ.get("GATE_LR", 1e-2))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_USER = 3


def make_data(n_per_user=4000, seed=0):
    tr = pd.read_parquet(DATA)
    tr = tr.dropna(subset=["text"])
    labels = sorted(tr["label"].unique())
    rng = np.random.RandomState(seed)
    rng.shuffle(labels)
    # 三个用户，各自偏好 5 个情绪类别（互不重叠，尽量）
    per = max(1, len(labels) // (N_USER + 2))
    prefs = [set(labels[i * per:(i + 1) * per]) for i in range(N_USER)]

    X, S, Y = [], [], []
    for u in range(N_USER):
        sub = tr.sample(n=min(n_per_user, len(tr)), random_state=seed + u)
        for txt, lab in zip(sub["text"], sub["label"]):
            X.append(txt)
            S.append(u)
            Y.append(1 if lab in prefs[u] else 0)
    X, S, Y = np.array(X, dtype=object), np.array(S), np.array(Y, dtype=np.int64)
    idx = rng.permutation(len(X))
    X, S, Y = X[idx], S[idx], Y[idx]
    cut = int(0.8 * len(X))
    return (X[:cut], S[:cut], Y[:cut]), (X[cut:], S[cut:], Y[cut:]), prefs


class DS(Dataset):
    def __init__(self, texts, states, ys, tok):
        self.t, self.s, self.y, self.tok = texts, states, ys, tok

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        e = self.tok(str(self.t[i]), truncation=True, max_length=MAX_LEN,
                     padding="max_length", return_tensors="pt")
        oh = torch.zeros(N_USER)
        oh[int(self.s[i])] = 1.0
        return ({k: v.squeeze(0) for k, v in e.items()}, oh, torch.tensor(int(self.y[i])))


class TextOnly(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(MODEL_DIR)
        self.head = nn.Linear(self.enc.config.hidden_size, 2)

    def forward(self, ids, mask, st):
        return self.head(self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0])


class Concat(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc = AutoModel.from_pretrained(MODEL_DIR)
        self.head = nn.Linear(self.enc.config.hidden_size + N_USER, 2)

    def forward(self, ids, mask, st):
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        return self.head(torch.cat([h, st], dim=1))


class FiLM(nn.Module):
    def __init__(self, n_mod=4):
        super().__init__()
        self.enc = AutoModel.from_pretrained(MODEL_DIR)
        d = self.enc.config.hidden_size
        self.n_mod = n_mod
        self.to_gb = nn.ModuleList([nn.Linear(N_USER, 2 * d) for _ in range(n_mod)])
        for m in self.to_gb:
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)
        self.head = nn.Linear(d, 2)
        self._st = None
        self._i = 0
        for lyr in list(self.enc.encoder.layer)[-n_mod:]:
            lyr.register_forward_hook(self._hook)

    def _hook(self, module, inp, out):
        h = out[0]
        g, b = self.to_gb[self._i](self._st).chunk(2, dim=1)
        g, b = g.unsqueeze(1), b.unsqueeze(1)
        self._i = (self._i + 1) % self.n_mod
        return ((1 + g) * h + b,) + out[1:]

    def forward(self, ids, mask, st):
        self._st = st
        self._i = 0
        return self.head(self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0])


def run(kind, tr, te, tok):
    model = {"text": TextOnly, "concat": Concat, "film": FiLM}[kind]().to(DEVICE)
    enc_p, gate_p, oth_p = [], [], []
    for n_, q in model.named_parameters():
        (enc_p if n_.startswith("enc.") else (gate_p if n_.startswith("to_gb") else oth_p)).append(q)
    groups = ([{"params": enc_p, "lr": 2e-5}] if enc_p else [])
    if gate_p:
        groups.append({"params": gate_p, "lr": GATE_LR})
    groups.append({"params": oth_p, "lr": 1e-3})
    opt = torch.optim.AdamW(groups, lr=1e-3)
    lossf = nn.CrossEntropyLoss()
    dl = DataLoader(DS(*tr, tok), batch_size=BATCH, shuffle=True)
    for ep in range(EPOCHS):
        model.train()
        tot = 0.0
        for batch, st, y in dl:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            loss = lossf(model(batch["input_ids"], batch["attention_mask"], st.to(DEVICE)), y.to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(y)
        print(f"  [{kind}] ep{ep+1} loss {tot/len(dl.dataset):.4f}", flush=True)

    model.eval()
    dl2 = DataLoader(DS(*te, tok), batch_size=64)
    per_user = {u: [0, 0] for u in range(N_USER)}
    c = t = 0
    with torch.no_grad():
        for batch, st, y in dl2:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            pred = model(batch["input_ids"], batch["attention_mask"], st.to(DEVICE)).argmax(-1).cpu()
            c += (pred == y).sum().item()
            t += len(y)
            for u in range(N_USER):
                m = (st.argmax(-1) == u)
                per_user[u][0] += int((pred[m] == y[m]).sum())
                per_user[u][1] += int(m.sum())
    return round(c / t, 4), {u: round(v[0] / max(1, v[1]), 3) for u, v in per_user.items()}


def main():
    seed = int(os.environ.get("SEED", 0))
    only = os.environ.get("ONLY", "")
    torch.manual_seed(seed)
    np.random.seed(seed)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    tr, te, prefs = make_data(seed=seed)
    print(f"训练 {len(tr[0])} / 测试 {len(te[0])} | 正例比例 {tr[2].mean():.2f}", flush=True)
    print("三个用户的偏好类别:", [sorted(x) for x in prefs], flush=True)

    res = {}
    t0 = time.time()
    for kind in (only.split(",") if only else ["text", "concat", "film"]):
        print(f"\n=== {kind} ===", flush=True)
        acc, per = run(kind, tr, te, tok)
        res[kind] = {"acc": acc, "per_user": per}
        print(f"  {kind}: acc {acc}  每用户 {per}", flush=True)
    res["minutes"] = round((time.time() - t0) / 60, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"state_real_s{seed}.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n" + json.dumps(res, ensure_ascii=False, indent=1))
    print("saved", p)


if __name__ == "__main__":
    main()
