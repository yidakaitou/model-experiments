# -*- coding: utf-8 -*-
"""data4：修 data3 的两个 bug
1) 别人也用第一人称说话（"朋友说：我换工作了"），必须靠说话人前缀做归属
2) 对话长度分布拉宽（2–10 行闲聊），避免 long 留出集上 JSON 退化
"""
import json
import random
from pathlib import Path

OUT = Path(r"D:\dsh-lab\data\extract_sft4")
OUT.mkdir(parents=True, exist_ok=True)

COMPANIES = ["银行", "字节跳动", "一家小公司", "国企", "设计公司", "外卖平台"]
CITIES = ["城东", "城西", "乌鲁木齐", "成都", "杭州", "老家"]
NAMES = ["苑", "用户", "小紫", "阿苑"]
REL = ["单身", "有对象了", "分手了", "订婚了"]
OTHER_COMPANIES = ["腾讯", "华为", "小米", "联通", "一家外贸公司"]
OTHER_CITIES = ["深圳", "南京", "武汉", "西安"]
OTHER_REL = ["结婚了", "异地恋中", "和好了", "离婚了"]

CHAT = ["今天食堂的菜好咸", "地铁上好挤", "外面下雨了", "刚洗完衣服", "手机快没电了",
        "楼下新开了家面馆", "今天风好大", "猫又把杯子碰倒了", "睡到十点才醒", "看完了那本书",
        "昨晚没睡好", "想去剪头发", "路上堵车", "中午吃了拉面", "忘记带伞了", "在整理房间"]
EMO = ["算了，不说了", "有点累", "我是不是很没用", "你别管我", "困了", "今天挺开心的"]
SPEAKERS = ["妈妈", "朋友", "同事", "表姐"]
SLOTS = {"工作": COMPANIES, "住处": CITIES, "称呼": NAMES, "恋爱状态": REL}


def her_stmt(slot, val):
    if slot == "工作":
        return f"我在{val}上班"
    if slot == "住处":
        return f"我住在{val}"
    if slot == "称呼":
        return f"叫我{val}"
    return f"我{val}"


def her_change(slot, val):
    if slot == "工作":
        return f"我换工作了，现在在{val}"
    if slot == "住处":
        return f"我搬家了，现在住在{val}"
    if slot == "称呼":
        return f"以后叫我{val}"
    return f"我现在{val}"


def other_stmt(slot, val):
    """别人说自己的事，用第一人称——和真实对话一致。"""
    if slot == "工作":
        return f"我换工作了，现在在{val}"
    if slot == "住处":
        return f"我搬家了，现在住在{val}"
    if slot == "称呼":
        return f"你们叫我{val}"
    return f"我{val}"


def make_one(rng):
    facts = {}
    lines = []

    slots = rng.sample(list(SLOTS.keys()), rng.randint(2, 4))
    for s in slots:
        v = rng.choice(SLOTS[s])
        facts[s] = v
        lines.append("用户说：" + her_stmt(s, v))
        if rng.random() < 0.4:
            v2 = rng.choice([x for x in SLOTS[s] if x != v])
            facts[s] = v2
            lines.append("用户说：" + her_change(s, v2))

    # 别人说同一槽位的事（第一人称，最容易混淆）
    for _ in range(rng.randint(1, 3)):
        who = rng.choice(SPEAKERS)
        s = rng.choice(list(SLOTS.keys()))
        if s == "工作":
            v = rng.choice(OTHER_COMPANIES)
        elif s == "住处":
            v = rng.choice(OTHER_CITIES)
        elif s == "称呼":
            v = rng.choice(NAMES)
        else:
            v = rng.choice(OTHER_REL)
        lines.append(f"{who}说：" + other_stmt(s, v))

    # 闲聊 2–10 行（拉宽长度分布）
    for _ in range(rng.randint(2, 10)):
        lines.append("用户说：" + rng.choice(CHAT))

    behaviors = []
    if rng.random() < 0.5:
        e = rng.choice(EMO)
        lines.append("用户说：" + e)
        behaviors.append(e)

    rng.shuffle(lines)

    beliefs = [{"claim": f"她的{s}是{v}", "type": "fact", "slot": s, "value": v}
               for s, v in facts.items()]
    beliefs += [{"claim": f"当她说「{e}」时，其实是想被接住", "type": "behavior", "slot": "", "value": ""}
                for e in behaviors]
    return "\n".join(lines), {"beliefs": beliefs}


SYS = ("你是记忆整理器。从对话里提炼结论，只输出 JSON。\n"
       "规则：只记「用户说」的话。别人（妈妈/朋友/同事/表姐）说的「我」，指的是他们自己，不是用户。\n"
       "同一时间只有一个值的事（工作/住处/称呼/恋爱状态）记成 fact，必须给 slot 和 value；"
       "同一件事提到多次时，以用户最后说的那次为准。她的反应规律记成 behavior，slot 和 value 留空。\n"
       "只输出 {\"beliefs\":[...]}，不要别的文字。")


def main():
    rng = random.Random(20261121)
    for split, n in (("train", 3500), ("test", 300)):
        rows = []
        for _ in range(n):
            inp, out = make_one(rng)
            rows.append({"messages": [
                {"role": "system", "content": SYS},
                {"role": "user", "content": inp},
                {"role": "assistant", "content": json.dumps(out, ensure_ascii=False)},
            ]})
        p = OUT / f"{split}.jsonl"
        p.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")
        print(f"{split}: {len(rows)} 条 -> {p}")


if __name__ == "__main__":
    main()
