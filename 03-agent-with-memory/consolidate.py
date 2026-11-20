# -*- coding: utf-8 -*-
"""巩固器：把一堆 Episode 压成 Belief / Question。

python consolidate.py [--mem memory.jsonl] [--beliefs beliefs.jsonl] [--n 24] [--dry]

要点：
1. 给模型的输入是编号后的原话，输出必须用编号引用证据
2. 每条证据都回查原话，编造直接丢弃
3. 与已有结论相似 → 合并；相反 → 标 conflict
"""
import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, r"D:\dsh-lab\human-memory")
from beliefs import BeliefStore, quality_gate  # noqa: E402

ENDPOINT = "http://127.0.0.1:8080/v1/chat/completions"

SYSP = ("你是一个记忆整理器。只输出 JSON，不要解释，不要 Markdown 代码块。"
        "你的任务：从多段对话记录里，提炼出关于「她」的稳定结论，以及你还没弄清的问题。")

PROMPT = """下面是你和她的对话记录（带编号）：

{lines}

请提炼，输出这个 JSON（不要别的文字）：
{{
 "beliefs": [
   {{"claim": "一句关于她的行为规律，25字内", "confidence": 0.0-1.0, "evidence": [编号], "slot": "工作|住处|称呼|关系|空", "value": "这个槽当前的值，如 银行；slot 为空时留空"}}
 ],
 "questions": [
   {{"claim": "你还没弄清的事，20字内", "evidence": [编号]}}
 ]
}}

claim 分两类，都要写：

【事实型】她的处境里「同一时间只有一个值」的事，必须带 slot：
- 「她的工作是在银行」→ slot 填「工作」
- 「她现在住在城东」→ slot 填「住处」
- 「她让我们叫她苑」→ slot 填「称呼」
事实型用陈述句，不要套「当她说…时」，因为你是在记事实，不是在解释她。

【行为型】她对什么有反应、想要什么，slot 留空：
- 「当她说『算了』时，其实是想有人接着问」
- 「她要的是 X，不是 Y」
行为型会累积；事实型被新事实取代时旧条自动失效。

好例子：
- 当她说「算了」时，其实是想有人接着问
- 她要的是有人站她这边，不是分析对错
- 她累的时候不想听建议，只想被陪着

坏例子（禁止）：
- 她有焦虑倾向（这是心理诊断）
- 她情绪低落（这是复述，不是规律）
- 她需要关心（太泛，无法指导说话）

规则：
1. 禁止出现「焦虑/抑郁/倾向/心理/障碍」这类诊断词
1.4 事实型必须同时给 value（槽的当前值）；slot 为空时 value 也留空
1.5 slot 只填「同一时间只有一个值」的事：工作/住处/称呼/关系；习惯、喜好、性格这类会叠加的，slot 写「空」
2. evidence 只填上面出现过的编号，最多 3 个最相关的
3. 没把握就不要写，宁可少写
4. 最多 4 条 beliefs、3 条 questions"""


def call(prompt, max_tokens=700):
    body = {"messages": [{"role": "system", "content": SYSP},
                         {"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": 0.0, "top_p": 0.9,
            "chat_template_kwargs": {"enable_thinking": False}, "stream": False}
    req = urllib.request.Request(ENDPOINT, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read().decode())["choices"][0]["message"]["content"].strip()


def parse_json(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mem", default=r"D:\dsh-lab\youwen\local_run\v42\memory.jsonl")
    ap.add_argument("--beliefs", default=r"D:\dsh-lab\youwen\local_run\v42\beliefs.jsonl")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    eps = []
    for line in Path(a.mem).read_text(encoding="utf-8").splitlines():
        try:
            eps.append(json.loads(line))
        except Exception:
            pass
    eps = eps[-a.n:]
    if not eps:
        print("没有记忆可巩固")
        return

    # 数据质量闸：他的话够不够、是不是全在自言自语
    ok, reason = quality_gate([e.get("text", "") for e in eps])
    print(f"[质量闸] {reason}")
    if not ok:
        print("→ 本轮不巩固（避免在污染数据上学生东西）")
        return

    lines = "\n".join(f"[{i+1}] {e.get('text','')[:60]}" for i, e in enumerate(eps))
    raw = call(PROMPT.format(lines=lines))
    data = parse_json(raw)
    if not data:
        print("第一次解析失败，重试一次……")
        raw = call(PROMPT.format(lines=lines) + "\n\n只输出 JSON，不要任何其他字符。")
        data = parse_json(raw)
    if not data:
        print("解析失败，原始输出：\n", raw[:600])
        return

    store = BeliefStore(a.beliefs)
    before = store.stats()
    added = {"new": 0, "merged": 0, "conflict": 0, "superseded": 0}
    dropped = 0

    for kind, key, conf in (("belief", "beliefs", None), ("question", "questions", None)):
        for item in data.get(key, []):
            claim = str(item.get("claim", "")).strip()
            idxs = item.get("evidence") or []
            if re.search(r"(是否在|刻意|是不是在|意味着)", claim):   # 过度解读，丢弃
                dropped += 1
                continue
            if not claim or not idxs:
                dropped += 1
                continue
            ev = []
            for i in idxs:
                try:
                    e = eps[int(i) - 1]
                    ev.append({"ep": e.get("id", ""), "quote": e.get("text", "")[:50]})
                except Exception:
                    continue
            if not ev:                       # 证据无效 → 丢弃（防编造）
                dropped += 1
                continue
            c = float(item.get("confidence", 0.6)) if kind == "belief" else 0.0
            _why = "证据：" + "；".join((x.get("quote") or "")[:16] for x in ev[:2])
            _slot = str(item.get("slot") or "").strip()[:8] if kind == "belief" else ""
            if _slot in ("空", "无", "none", "null"):
                _slot = ""
            _val = str(item.get("value") or "").strip()[:12] if kind == "belief" else ""
            if _val in ("空", "无", "none", "null"):
                _val = ""
            action, _nid = store.add_or_merge(kind, claim, ev, c, why=_why, slot=_slot, value=_val)
            added[action] += 1

    print(f"巩固 {len(eps)} 条记忆 → 新增/合并/冲突 {added}，丢弃(证据无效) {dropped}")
    print("之前:", before)
    print("之后:", store.stats())
    for n in store.nodes[-6:]:
        print(f"  [{n.kind}/{n.status}] {n.claim}  conf={n.confidence:.2f}  证据{len(n.evidence)}条")
    if store.conflicts():
        print("冲突待裁决:")
        for n in store.conflicts():
            print(f"  ! {n.claim}  ← {n.note}")
    if not a.dry:
        store.save()
        print("saved", a.beliefs)
    else:
        print("(dry run，未写盘)")


if __name__ == "__main__":
    main()
