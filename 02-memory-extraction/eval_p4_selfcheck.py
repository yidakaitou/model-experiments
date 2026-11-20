# -*- coding: utf-8 -*-
"""P4 评测：模型能不能判断「这条结论有没有依据」。

输出：有据组准确率、编造组准确率、整体准确率、误报/漏报率。
判据来自北极星 P4：自评准确率 >80% 才算成立。
"""
import argparse
import json
import re
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

SYS = ("你是审稿人。判断一条结论能不能从对话里得到支持。只输出 JSON。")

TPL = """对话：
{inp}

结论：{claim}

这条结论能否直接从上面的对话里得到支持？
只输出：{{"supported": true}} 或 {{"supported": false}}"""


def parse(t):
    m = re.search(r"\{.*?\}", t, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        v = d.get("supported")
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("true", "是", "yes", "对")
    except Exception:
        pass
    # 兜底：看关键词
    low = t.lower()
    if "false" in low or "不支持" in t:
        return False
    if "true" in low or "支持" in t:
        return True
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--lora", default="")
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    dev = "cuda"
    cap = torch.cuda.get_device_capability(0)
    dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
    tok = AutoTokenizer.from_pretrained(a.model)
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=dtype, device_map=dev)
    model = PeftModel.from_pretrained(base, a.lora) if a.lora else base
    model.eval()

    rows = [json.loads(l) for l in Path(a.data).read_text(encoding="utf-8").splitlines() if l.strip()]
    hit_yes = tot_yes = hit_no = tot_no = bad = 0
    wrong = []

    for r in rows:
        prompt = tok.apply_chat_template(
            [{"role": "system", "content": SYS},
             {"role": "user", "content": TPL.format(inp=r["input"], claim=r["claim"])}],
            tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt", truncation=True, max_length=800).to(dev)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=24, do_sample=False, pad_token_id=tok.eos_token_id)
        txt = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
        pred = parse(txt)
        if pred is None:
            bad += 1
            continue
        if r["supported"]:
            tot_yes += 1
            if pred:
                hit_yes += 1
            elif len(wrong) < 4:
                wrong.append({"claim": r["claim"], "truth": True, "pred": False})
        else:
            tot_no += 1
            if not pred:
                hit_no += 1
            elif len(wrong) < 4:
                wrong.append({"claim": r["claim"], "truth": False, "pred": True})

    res = {
        "n": len(rows), "parse_fail": bad,
        "supported_acc": round(hit_yes / max(1, tot_yes), 3),
        "unsupported_acc": round(hit_no / max(1, tot_no), 3),
        "overall_acc": round((hit_yes + hit_no) / max(1, tot_yes + tot_no), 3),
        "false_alarm": tot_yes - hit_yes,   # 把对的判成错的
        "miss": tot_no - hit_no,            # 把编造的判成对的
        "wrong_examples": wrong,
    }
    res["P4_pass"] = res["overall_acc"] >= 0.8
    Path(a.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
