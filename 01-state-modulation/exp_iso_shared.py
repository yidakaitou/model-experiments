# -*- coding: utf-8 -*-
"""P1 补丁实验：隔离会丢迁移，能不能用「共享适配器」补回来？

对照：
  A 纯隔离（已有结果 0.88 / 0.85 / 0.83，遗忘 0）
  B 隔离 + 共享轻量适配器（适配器在全部任务上学，带 20% 回放）+ 每任务独立头

思路来自 CoDe-LoRA：把「共享知识」和「任务特定知识」分成两支。
这里用最便宜的方式实现：编码器冻结，共享 adapter 学跨任务，head 各自独立。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, r"D:\dsh-lab\scripts")
from exp_continual import DS, build_tasks, MODEL_DIR  # noqa: E402
from transformers import AutoModel, AutoTokenizer

OUT = Path(r"D:\dsh-lab\data\continual")
ADAPTER_DIM = 64
EPOCHS = 6
BATCH = 32
LR_HEAD = 1e-3
LR_ADAPTER = 3e-4
REPLAY = 0.2


class Adapter(nn.Module):
    """瓶颈适配器：h + W2(gelu(W1(h)))，零初始化 W2 → 起点等价于原模型。"""
    def __init__(self, d):
        super().__init__()
        self.down = nn.Linear(d, ADAPTER_DIM)
        self.up = nn.Linear(ADAPTER_DIM, d)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, h):
        return h + self.up(torch.nn.functional.gelu(self.down(h)))


class IsoPlusShared(nn.Module):
    def __init__(self, n_mod=4):
        super().__init__()
        self.enc = AutoModel.from_pretrained(MODEL_DIR)
        for p in self.enc.parameters():
            p.requires_grad = False
        d = self.enc.config.hidden_size
        self.adapters = nn.ModuleList([Adapter(d) for _ in range(n_mod)])
        self.n_mod = n_mod
        self._i = 0
        for lyr in list(self.enc.encoder.layer)[-n_mod:]:
            lyr.register_forward_hook(self._hook)
        self.heads = nn.ModuleDict()

    def _hook(self, module, inp, out):
        h = out[0]
        h = self.adapters[self._i](h)
        self._i = (self._i + 1) % self.n_mod
        return (h,) + out[1:]

    def forward(self, ids, mask, task):
        self._i = 0
        h = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        return self.heads[task](h)


@torch.no_grad()
def evaluate(model, tok, texts, labels, task, device):
    model.eval()
    dl = DataLoader(DS(texts, labels, tok), batch_size=64)
    c = t = 0
    for batch, y in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        p = model(batch["input_ids"], batch["attention_mask"], task).argmax(-1).cpu()
        c += (p == y).sum().item()
        t += len(y)
    return c / max(1, t)


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    tasks = build_tasks(seed)

    model = IsoPlusShared().to(device)
    d = model.enc.config.hidden_size
    for t in tasks:
        model.heads[t["name"]] = nn.Linear(d, 9).to(device)

    t0 = time.time()
    acc = {}
    for si, task in enumerate(tasks):
        nm = task["name"]
        print(f"\n=== 阶段 {si+1}: {nm}（隔离+共享适配器）===", flush=True)
        # 本任务样本 + 20% 回放的旧任务样本（各自带自己的 task 名）
        samples = [(task["train"][0][i], task["train"][1][i], nm) for i in range(len(task["train"][0]))]
        if si > 0:
            for prev in tasks[:si]:
                ptx, pty = prev["train"]
                k = int(REPLAY * len(ptx))
                idx = np.random.choice(len(ptx), k, replace=False)
                samples += [(ptx[i], pty[i], prev["name"]) for i in idx]

        params = list(model.adapters.parameters()) + list(model.heads[nm].parameters())
        opt = torch.optim.AdamW([
            {"params": model.adapters.parameters(), "lr": LR_ADAPTER},
            {"params": model.heads[nm].parameters(), "lr": LR_HEAD},
        ])
        lossf = nn.CrossEntropyLoss()
        rng = np.random.RandomState(seed + si)
        for ep in range(EPOCHS):
            model.train()
            perm = rng.permutation(len(samples))
            tot = 0.0
            for i in range(0, len(perm), BATCH):
                chunk = [samples[j] for j in perm[i:i + BATCH]]
                texts = [c[0] for c in chunk]
                ys = torch.tensor([c[1] for c in chunk]).to(device)
                names = [c[2] for c in chunk]
                e = tok(texts, truncation=True, max_length=48, padding=True, return_tensors="pt")
                e = {k: v.to(device) for k, v in e.items()}
                # 同一个 batch 里可能有不同 task 的样本 → 分组合并 logits
                outs = []
                for nmm in set(names):
                    idxs = [j for j, x in enumerate(names) if x == nmm]
                    sub = {k: v[idxs] for k, v in e.items()}
                    outs.append((idxs, model(sub["input_ids"], sub["attention_mask"], nmm)))
                logits = torch.zeros(len(chunk), 9, device=device)
                for idxs, o in outs:
                    logits[idxs] = o
                loss = lossf(logits, ys)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(chunk)
            print(f"    ep{ep+1} loss {tot/len(samples):.4f}", flush=True)

        for t2 in tasks:
            a = evaluate(model, tok, t2["test"][0], t2["test"][1], t2["name"], device)
            acc[(t2["name"], si)] = round(a, 4)
            print(f"    {t2['name']} {a:.4f}", flush=True)

    names = [t["name"] for t in tasks]
    drops = []
    for tj, nm2 in enumerate(names[:-1]):
        peak = max(acc[(nm2, s)] for s in range(tj, len(tasks)))
        drops.append(max(0.0, peak - acc[(nm2, len(tasks) - 1)]))
    res = {"mode": "iso+shared", "seed": seed,
           "matrix": {f"after_{names[si]}": {n2: acc[(n2, si)] for n2 in names} for si in range(len(tasks))},
           "final": {n2: acc[(n2, len(tasks) - 1)] for n2 in names},
           "forgetting": round(float(np.mean(drops)) if drops else 0.0, 4),
           "minutes": round((time.time() - t0) / 60, 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"iso_shared_s{seed}.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n" + json.dumps({k: res[k] for k in ("final", "forgetting", "minutes")}, ensure_ascii=False, indent=1))
    print("saved", p)


if __name__ == "__main__":
    main()
