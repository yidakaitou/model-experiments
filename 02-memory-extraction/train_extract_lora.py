# -*- coding: utf-8 -*-
"""把「对话→结构化结论」的抽取能力训进权重，并和零样本比较。

流程：加载 Qwen3-1.7B → 零样本评测 → LoRA 微调 → 再评测 → 出对比
指标：JSON 合法率 / 事实条目的 (slot,value) 完全正确率 / 行为条目召回
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("MODEL", r"D:\dsh-lab\models\Qwen3-1.7B")
DATA = Path(os.environ.get("DATA", r"D:\dsh-lab\data\extract_sft"))
OUT = Path(os.environ.get("OUT", r"D:\dsh-lab\data\extract_sft\result.json"))
DEVICE = "cuda"
_cap = torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0, 0)
_AUTO = "1" if _cap[0] < 8 else "0"      # V100/更老 → fp16；Ampere 及以后 → bf16
DTYPE = torch.float16 if os.environ.get("FP16", _AUTO) == "1" else torch.bfloat16
MAXLEN = 512
BATCH = int(os.environ.get('BATCH', 1))
ACCUM = int(os.environ.get('ACCUM', 8))
EPOCHS = int(os.environ.get("EPOCHS", 1))
LR = 1e-4
N_EVAL = int(os.environ.get("N_EVAL", 120))


class DS(Dataset):
    def __init__(self, path, tok):
        self.rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        self.tok = tok

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        m = self.rows[i]["messages"]
        prompt = self.tok.apply_chat_template(m[:2], tokenize=False, add_generation_prompt=True)
        full = prompt + m[2]["content"] + self.tok.eos_token
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        ids = self.tok(full, add_special_tokens=False, truncation=True, max_length=MAXLEN)["input_ids"]
        labels = list(ids)
        for j in range(min(len(p_ids), len(labels))):
            labels[j] = -100
        return {"input_ids": torch.tensor(ids), "labels": torch.tensor(labels),
                "attention_mask": torch.ones(len(ids), dtype=torch.long)}


def collate(batch):
    ml = max(len(b["input_ids"]) for b in batch)
    out = {"input_ids": [], "labels": [], "attention_mask": []}
    for b in batch:
        pad = ml - len(b["input_ids"])
        out["input_ids"].append(torch.cat([b["input_ids"], torch.zeros(pad, dtype=torch.long)]))
        out["labels"].append(torch.cat([b["labels"], torch.full((pad,), -100)]))
        out["attention_mask"].append(torch.cat([b["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
    return {k: torch.stack(v).to(DEVICE) for k, v in out.items()}


def parse_json(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def score(pred_obj, gold_obj):
    """返回 (json_ok, fact_exact, behavior_recall)"""
    if pred_obj is None:
        return 0, 0, 0
    pf = {(b.get("slot"), b.get("value")) for b in pred_obj.get("beliefs", []) if b.get("type") == "fact"}
    gf = {(b.get("slot"), b.get("value")) for b in gold_obj.get("beliefs", []) if b.get("type") == "fact"}
    pb = {b.get("claim") for b in pred_obj.get("beliefs", []) if b.get("type") == "behavior"}
    gb = {b.get("claim") for b in gold_obj.get("beliefs", []) if b.get("type") == "behavior"}
    fact_ok = 1 if pf == gf else 0
    beh = 1 if (not gb) or (pb & gb) else 0
    return 1, fact_ok, beh


@torch.no_grad()
def evaluate(model, tok, rows, tag="eval"):
    model.eval()
    ok = fact = beh = 0
    dump = []
    for r in rows[:N_EVAL]:
        m = r["messages"]
        prompt = tok.apply_chat_template(m[:2], tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=MAXLEN - 200).to(DEVICE)
        out = model.generate(**ids, max_new_tokens=220, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        txt = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        obj = parse_json(txt)
        gold = json.loads(m[2]["content"])
        a, b, c = score(obj, gold)
        ok += a
        fact += b
        beh += c
        dump.append({"input": m[1]["content"], "gold": gold, "pred_raw": txt,
                     "pred": obj, "json_ok": a, "fact_ok": b, "behavior_ok": c})
    n = min(N_EVAL, len(rows))
    try:
        (DATA / "out" / f"dump_{tag}.jsonl").parent.mkdir(parents=True, exist_ok=True)
        with open(DATA / "out" / f"dump_{tag}.jsonl", "w", encoding="utf-8") as f:
            for d in dump:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
    except Exception as e:
        print("dump 失败", e)
    return {"json_ok": round(ok / n, 3), "fact_exact": round(fact / n, 3), "behavior": round(beh / n, 3), "n": n}


def main():
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    rows_te = [json.loads(l) for l in (DATA / "test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]

    print("== 加载基座 ==", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE, device_map=DEVICE)
    if os.environ.get('GC', '1') == '1':
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    print("== 零样本评测 ==", flush=True)
    base = evaluate(model, tok, rows_te, tag="base")
    print("   ", base, flush=True)

    _r = int(os.environ.get("LORA_R", "16"))
    cfg = LoraConfig(r=_r, lora_alpha=_r*2, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    _init = os.environ.get("LORA_INIT", "").strip()
    if _init and Path(_init).exists():
        # 续训：从上一个适配器继续，而不是从基座重来
        model = PeftModel.from_pretrained(model, _init, is_trainable=True)
        print(f"== 续训：从 {_init} 初始化 ==", flush=True)
    else:
        model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    ds = DS(DATA / "train.jsonl", tok)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=collate)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    print(f"== 训练 {len(ds)} 条 × {EPOCHS} epoch ==", flush=True)
    model.train()
    step = 0
    scaler = torch.amp.GradScaler("cuda", enabled=(DTYPE == torch.float16))
    for ep in range(EPOCHS):
        for i, batch in enumerate(dl):
            with torch.amp.autocast("cuda", dtype=DTYPE):
                out = model(**batch)
                loss = out.loss / ACCUM
            scaler.scale(loss).backward()
            if (i + 1) % ACCUM == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                step += 1
                if step % 20 == 0:
                    print(f"  ep{ep+1} step{step} loss {out.loss.item():.4f}  {time.time()-t0:.0f}s", flush=True)

    try:
        save_dir = str(OUT.parent / "lora")
        model.save_pretrained(save_dir)
        tok.save_pretrained(save_dir)
        print("== 适配器已保存 ==", save_dir, flush=True)
    except Exception as e:
        print("保存失败", e, flush=True)

    print("== 微调后评测 ==", flush=True)
    tuned = evaluate(model, tok, rows_te, tag="tuned")
    print("   ", tuned, flush=True)

    res = {"base": base, "tuned": tuned, "minutes": round((time.time() - t0) / 60, 1)}
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))
    print("saved", OUT)


if __name__ == "__main__":
    main()
