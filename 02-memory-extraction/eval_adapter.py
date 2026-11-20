# -*- coding: utf-8 -*-
"""评测已保存的 LoRA 适配器在 5 个留出集上的表现。

区分「学会规则」vs「背下模板」：
  - paraphrase / noise / novel / long：意思相同、字面不同 → 会规则才对
  - multispeaker：不能把别人的事算成她的

用法（云端）:
  python eval_adapter.py --model /root/p2/models/Qwen3-1.7B --lora /root/p2/out/lora \
                         --data /root/p2/heldout --out /root/p2/out/heldout.json
"""
import argparse
import json
import re
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SYS = ("你是记忆整理器。从对话里提炼结论，只输出 JSON。\n"
       "规则：同一时间只有一个值的事（工作/住处/称呼/恋爱状态）记成 fact，必须给 slot 和 value；"
       "同一件事提到多次时，以最后发生的那次为准。她的反应规律记成 behavior，slot 和 value 留空。\n"
       "只输出 {\"beliefs\":[...]}，不要别的文字。")


def parse_json(txt):
    m = re.search(r"\{.*\}", txt, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def score(pred, gold):
    if pred is None:
        return {"json_ok": 0, "fact_exact": 0, "false_facts": 0}
    pf = {(b.get("slot"), b.get("value")) for b in pred.get("beliefs", []) if b.get("type") == "fact"}
    gf = {(b.get("slot"), b.get("value")) for b in gold.get("beliefs", []) if b.get("type") == "fact"}
    return {
        "json_ok": 1,
        "fact_exact": 1 if pf == gf else 0,
        "false_facts": len(pf - gf),     # 多记了别人的事/错误值
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora", default="")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()

    device = "cuda"
    cap = torch.cuda.get_device_capability(0)
    dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
    tok = AutoTokenizer.from_pretrained(a.model)
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype, device_map=device)
    if a.lora:
        model = PeftModel.from_pretrained(base, a.lora)
    else:
        model = base
    model.eval()
    print("mode:", "lora" if a.lora else "zero-shot", flush=True)

    data = Path(a.data)
    result = {}
    for f in sorted(data.glob("*.jsonl")):
        rows = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()][:a.n]
        ok = fact = false = 0
        for r in rows:
            prompt = tok.apply_chat_template(
                [{"role": "system", "content": SYS},
                 {"role": "user", "content": r["input"]}],
                tokenize=False, add_generation_prompt=True)
            ids = tok(prompt, return_tensors="pt", truncation=True, max_length=600).to(device)
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=200, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            txt = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            s = score(parse_json(txt), r["gold"])
            ok += s["json_ok"]
            fact += s["fact_exact"]
            false += s["false_facts"]
        n = len(rows)
        result[f.stem] = {"json_ok": round(ok / n, 3), "fact_exact": round(fact / n, 3),
                          "false_facts_total": false, "n": n}
        print(f"{f.stem:14s} json {ok/n:.2f}  fact {fact/n:.2f}  多记 {false}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print("saved", a.out)


if __name__ == "__main__":
    main()
