# -*- coding: utf-8 -*-
"""集成测试 v2：结构化提问 + 系统层兜底规则。

问题：模型对省略回答的泛化极窄（训练有「嗯，对」→ 测试「嗯对」就失效）。

修法：模型提问时带 slot+guess，系统收到回答后**先用规则判断**：
  确认类（对/是/嗯/没错/对啊/嗯对/对对对...）→ 直接写 slot=guess，不依赖模型
  否认类（不是/不对...）→ 交给模型重新抽取（可能带来新值）
  其他 → 交给模型正常抽取

这样模型的泛化缺口被 20 行规则补掉，而且规则可审计。
"""
import json
import os
import re
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/root/p2")
from beliefs import BeliefStore

SYS = ("你是记忆整理器。从对话里提炼结论，只输出 JSON。\n"
       "规则：只记「用户说」的话。别人说的「我」指的是他们自己。\n"
       "四种输出，按情况选一种：\n"
       "1) 她明确说了 → {\"beliefs\": [{\"claim\":\"她的X是Y\",\"type\":\"fact\",\"slot\":\"X\",\"value\":\"Y\"}]}\n"
       "2) 她后面说骗你的/收回 → {\"beliefs\": [], \"retracted\": [\"X=Y\"]}\n"
       "3) 她说得含糊或前后矛盾 → 别猜，带着候选问一句："
       "{\"beliefs\": [], \"ask\": \"一句具体的澄清问句\", \"slot\": \"X\", \"guess\": \"候选值\"}\n"
       "4) 纯闲聊，没有可记的 → {\"beliefs\": []}\n"
       "只输出 JSON。")

# ---------- 系统层兜底规则 ----------
CONFIRM = {"对", "对的", "对啊", "对呀", "对的对的", "对对", "对对对", "嗯", "嗯嗯", "嗯对", "嗯恩",
           "是", "是的", "是的呀", "是呀", "没错", "没错啊", "行", "可以", "是的对", "恩", "yes", "y"}
DENY = ("不是", "不对", "不", "错", "没", "别")


def norm(s):
    s = s.replace("用户说：", "").strip()
    s = re.sub(r"[，。！？、\s\"'「」『』]", "", s)
    return s.lower()


def judge(answer):
    """规则判断用户回答：confirm / deny / other"""
    a = norm(answer)
    if a in CONFIRM:
        return "confirm"
    for d in DENY:
        if a.startswith(d) and a not in CONFIRM:
            return "deny"
    return "other"


CASES = [
    ("用户说：我好像住在杭州？\n用户说：嗯……不确定\n用户说：今天风好大",
     "对", "含糊→问 | 极简确认"),
    ("用户说：我在银行上班\n用户说：地铁上好挤\n用户说：中午吃了拉面",
     None, "明确→直接记"),
    ("用户说：我搬家了，现在住在成都\n用户说：路上堵车",
     None, "明确变化→更新"),
    ("用户说：可能在设计公司吧，记不清了\n用户说：你猜",
     "嗯对", "含糊→问 | 确认变体（上次失败的那个）"),
    ("用户说：刚洗完衣服\n用户说：猫又把杯子碰倒了\n助手说：注意休息",
     None, "纯闲聊→闭嘴"),
    ("用户说：我是火星人，我住在环形山\n用户说：哈哈骗你的",
     None, "玩笑撤回→丢弃"),
    ("用户说：叫我苑\n用户说：今天食堂菜好咸",
     None, "明确→记称呼"),
    ("用户说：我好像在有对象了？\n用户说：记不清",
     "不是，我单身", "含糊→问 | 否认+纠正（要落成 单身）"),
]


def parse(t):
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def gen(model, tok, text, max_new=200, dev="cuda"):
    prompt = tok.apply_chat_template([{"role": "system", "content": SYS},
                                      {"role": "user", "content": text}],
                                     tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt", truncation=True, max_length=800).to(dev)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    dev = "cuda"
    cap = torch.cuda.get_device_capability(0)
    dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
    tok = AutoTokenizer.from_pretrained("/root/p2/models/Qwen3-8B")
    base = AutoModelForCausalLM.from_pretrained("/root/p2/models/Qwen3-8B", dtype=dtype, device_map=dev)
    model = PeftModel.from_pretrained(base, os.environ.get("LORA", "/root/p2/m10c/merged"))
    model.eval()

    store = BeliefStore("/root/p2/ask2_beliefs.jsonl")
    store.nodes.clear()

    stats = {"windows": len(CASES), "asks": 0, "rule_confirm": 0, "rule_deny": 0,
             "model_path": 0, "facts": 0, "retracted": 0, "silent": 0}
    trace = []

    for i, (win, my_answer, note) in enumerate(CASES, 1):
        rec = {"w": i, "note": note, "path": "", "turns": []}
        raw = gen(model, tok, win)
        obj = parse(raw) or {}
        rec["turns"].append({"in": win[:50], "out": raw[:130]})

        facts_to_store = []

        if obj.get("ask"):
            stats["asks"] += 1
            ask_q, slot, guess = obj.get("ask"), obj.get("slot"), obj.get("guess")
            verdict = judge(my_answer) if my_answer else "other"

            if (my_answer and slot and guess and verdict == "confirm"
                and os.environ.get("RULE_MODE", "on") == "on"):
                # ---- 兜底规则：确认 → 直接落位，不经模型 ----
                stats["rule_confirm"] += 1
                stats["model_path"] += 0
                facts_to_store = [(slot, guess)]
                rec["path"] = f"RULE(confirm) → {slot}={guess}"
                rec["turns"].append({"answer": my_answer, "verdict": verdict})
            else:
                # ---- 交给模型重新抽取 ----
                if os.environ.get("STRUCT_MODE", "off") == "on":
                    _q = obj.get("ask", "")
                    _a = str(my_answer).replace("用户说：", "")
                    convo = f"{win}\n[你的提问] {_q}\n[她的回答] {_a}"
                else:
                    convo = f"{win}\n用户说：{my_answer}" if my_answer else win
                raw2 = gen(model, tok, convo)
                obj2 = parse(raw2) or {}
                stats["model_path"] += 1
                if verdict == "deny":
                    stats["rule_deny"] += 1
                rec["path"] = f"MODEL(verdict={verdict})"
                rec["turns"].append({"answer": my_answer, "verdict": verdict, "out": raw2[:130]})
                obj = obj2

        f = [b for b in obj.get("beliefs", []) if b.get("type") == "fact"]
        if f and not facts_to_store:
            facts_to_store = [(b.get("slot"), b.get("value")) for b in f]

        if facts_to_store:
            for s, v in facts_to_store:
                stats["facts"] += 1
                store.add_or_merge("belief", f"她的{s}是{v}", [{"ep": f"W{i}"}], 0.8, slot=s, value=v)
            rec["stored"] = facts_to_store
        elif obj.get("retracted"):
            stats["retracted"] += 1
            rec["retracted"] = obj["retracted"]
        else:
            stats["silent"] += 1
        trace.append(rec)

    store.save()
    active = {(n.slot, n.value) for n in store.active("belief") if n.slot}
    sup = {(n.slot, n.value) for n in store.superseded() if n.slot}

    expect = {("住处", "成都"), ("工作", "设计公司"), ("称呼", "苑"), ("恋爱状态", "单身")}
    checks = {
        "最终记忆正确": active == expect,
        "没混入环形山": not any("环形山" in str(v) for _, v in active | sup),
        "旧住处已失效": ("住处", "杭州") in sup,
        "旧工作已失效": ("工作", "银行") in sup,
        "兜底规则被用上": stats["rule_confirm"] + stats["rule_deny"] > 0,
    }
    result = {"stats": stats, "final_active": sorted(active), "final_superseded": sorted(sup),
              "checks": checks, "all_pass": all(checks.values()), "trace": trace}
    Path(os.environ.get("OUT", "/root/p2/ask2.json")).write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("stats", "final_active", "final_superseded", "checks", "all_pass")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
