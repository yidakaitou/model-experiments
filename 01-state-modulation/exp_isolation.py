# -*- coding: utf-8 -*-
"""P1 对照实验：参数隔离能到什么程度，代价多大。

做法：冻结 BERT 编码器，每个任务只训一个独立的小头（linear probe）。
      因为旧任务的参数从不被改动，遗忘在构造上必然是 0。
      要测的是代价：新任务精度比全参微调低多少。

对比基线（全参微调峰值）：A 0.9391 / B 0.9142 / C 0.9184
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, r"D:\dsh-lab\scripts")
from exp_continual import DS, build_tasks, MODEL_DIR  # noqa: E402

OUT = Path(r"D:\dsh-lab\data\continual")
EPOCHS = 6
BATCH = 32
LR = 1e-3          # 只训头，学习率可以大


@torch.no_grad()
def encode_all(enc, tok, texts, device):
    enc.eval()
    dl = DataLoader(DS(texts, [0] * len(texts), tok), batch_size=64)
    out = []
    for batch, _ in dl:
        batch = {k: v.to(device) for k, v in batch.items()}
        out.append(enc(**batch).last_hidden_state[:, 0].cpu())
    return torch.cat(out)


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    tasks = build_tasks(seed)

    enc = AutoModel.from_pretrained(MODEL_DIR).to(device)
    for p in enc.parameters():
        p.requires_grad = False

    t0 = time.time()
    res = {}
    for task in tasks:
        print(f"\n=== {task['name']}：冻结编码器 + 独立头 ===", flush=True)
        Xtr = encode_all(enc, tok, task["train"][0], device)
        ytr = torch.tensor(task["train"][1])
        Xte = encode_all(enc, tok, task["test"][0], device)
        yte = torch.tensor(task["test"][1])

        head = nn.Linear(Xtr.shape[1], 9).to(device)   # 统一 9 类空间，但只训本任务 3 类的 logit
        opt = torch.optim.AdamW(head.parameters(), lr=LR)
        lossf = nn.CrossEntropyLoss()
        for ep in range(EPOCHS):
            perm = torch.randperm(len(Xtr))
            tot = 0.0
            for i in range(0, len(perm), BATCH):
                idx = perm[i:i + BATCH]
                x, y = Xtr[idx].to(device), ytr[idx].to(device)
                loss = lossf(head(x), y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += loss.item() * len(idx)
            print(f"    epoch {ep+1}/{EPOCHS} loss {tot/len(Xtr):.4f}", flush=True)

        with torch.no_grad():
            pred = head(Xte.to(device)).argmax(-1).cpu()
            acc = (pred == yte).float().mean().item()
        res[task["name"]] = round(acc, 4)
        print(f"    {task['name']} 准确率 {acc:.4f}（全参微调峰值参考见脚本头）", flush=True)

    res["minutes"] = round((time.time() - t0) / 60, 1)
    res["note"] = "隔离：遗忘=0（构造上），代价是精度"
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / f"isolation_s{seed}.json"
    p.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n" + json.dumps(res, ensure_ascii=False, indent=1))
    print("saved", p)


if __name__ == "__main__":
    main()
