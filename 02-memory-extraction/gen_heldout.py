# -*- coding: utf-8 -*-
"""生成 5 个「泛化」留出集，用于区分「学会了规则」vs「背下了模板」。

训练集只见过这些句式：
  工作: 我在{公司}上班 / 我换工作了，现在在{公司}
  住处: 我住在{城市} / 我搬家了，现在住在{城市}
  称呼: 叫我{名} / 以后叫我{名}
  恋爱: 我{状态}

留出集换一种说法（语义相同、字面不同）+ 加噪声 + 多人 + 新值 + 长对话。
会规则的应该仍对，背模板的会掉。
"""
import json
import random
from pathlib import Path

OUT = Path(r"D:\dsh-lab\data\heldout")
OUT.mkdir(parents=True, exist_ok=True)

# 训练集见过的值
TRAIN_COMPANIES = ["银行", "字节跳动", "一家小公司", "国企", "设计公司", "外卖平台"]
TRAIN_CITIES = ["城东", "城西", "乌鲁木齐", "成都", "杭州", "老家"]
TRAIN_NAMES = ["苑", "用户", "小紫", "阿苑"]
TRAIN_REL = ["单身", "有对象了", "分手了", "订婚了"]

# 留出集的新值（训练集没见过）
NEW_COMPANIES = ["腾讯", "华为", "小米", "哔哩哔哩"]
NEW_CITIES = ["深圳", "苏州", "南京", "厦门"]
NEW_NAMES = ["小满", "小禾", "南南", "团团"]
NEW_REL = ["结婚了", "异地恋中", "和好了"]

SYS = ("你是记忆整理器。从对话里提炼结论，只输出 JSON。\n"
       "规则：同一时间只有一个值的事（工作/住处/称呼/恋爱状态）记成 fact，必须给 slot 和 value；"
       "同一件事提到多次时，以最后发生的那次为准。她的反应规律记成 behavior，slot 和 value 留空。\n"
       "只输出 {\"beliefs\":[...]}，不要别的文字。")


def fact(slot, value):
    name = {"工作": "工作", "住处": "住处", "称呼": "称呼", "恋爱状态": "恋爱状态"}[slot]
    return {"claim": f"她的{name}是{value}", "type": "fact", "slot": slot, "value": value}


def mk(input_lines, facts):
    return {"input": "\n".join("用户说：" + x if not x.startswith(("用户说", "妈妈说", "朋友说")) else x
                               for x in input_lines),
            "gold": {"beliefs": [fact(s, v) for s, v in facts.items()]}}


def main():
    rng = random.Random(7)

    # 1) 改写：同值，不同句式
    rows = []
    P = {
        "工作": ["我的单位是{v}", "我目前在{v}工作", "{v}是我上班的地方", "我跳槽去了{v}", "我入职了{v}"],
        "住处": ["我现在安家在{v}", "我搬去{v}住了", "我定居在{v}", "我如今住在{v}"],
        "称呼": ["你喊我{v}", "我的名字是{v}", "以后你叫我{v}就行"],
        "恋爱状态": ["我目前{v}", "我最近{v}", "我现在的状态是{v}"],
    }
    for _ in range(100):
        slots = rng.sample(list(P), rng.randint(1, 3))
        facts, lines = {}, []
        for s in slots:
            v = rng.choice(TRAIN_COMPANIES if s == "工作" else TRAIN_CITIES if s == "住处"
                           else TRAIN_NAMES if s == "称呼" else TRAIN_REL)
            facts[s] = v
            lines.append(rng.choice(P[s]).format(v=v))
        rows.append(mk(lines, facts))
    (OUT / "paraphrase.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

    # 2) 噪声：句子里塞口语填充
    rows = []
    FILL = ["嗯，", "那个，", "就是，", "对了，", "其实，"]
    for _ in range(100):
        s = rng.choice(["工作", "住处", "称呼", "恋爱状态"])
        v = rng.choice(TRAIN_COMPANIES if s == "工作" else TRAIN_CITIES if s == "住处"
                       else TRAIN_NAMES if s == "称呼" else TRAIN_REL)
        base = {"工作": f"我在{v}上班", "住处": f"我住在{v}", "称呼": f"叫我{v}", "恋爱状态": f"我{v}"}[s]
        rows.append(mk([rng.choice(FILL) + base], {s: v}))
    (OUT / "noise.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

    # 3) 多人：混入妈妈/朋友的话，模型不能把别人的事算成她的
    rows = []
    for _ in range(100):
        s = rng.choice(["工作", "住处", "恋爱状态"])
        v = rng.choice(TRAIN_COMPANIES if s == "工作" else TRAIN_CITIES if s == "住处" else TRAIN_REL)
        her = {"工作": f"我在{v}上班", "住处": f"我住在{v}", "恋爱状态": f"我{v}"}[s]
        other = rng.choice([
            "妈妈说：你弟弟考上大学了",
            "朋友说：我换工作了，去腾讯了",
            "妈妈说：你爸搬去南京了",
            "朋友说：我最近在谈恋爱",
        ])
        lines = [other, her] if rng.random() < 0.5 else [her, other]
        rows.append(mk(lines, {s: v}))
    (OUT / "multispeaker.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

    # 4) 新值：训练集没见过的公司/城市/名字/状态
    rows = []
    for _ in range(100):
        s = rng.choice(["工作", "住处", "称呼", "恋爱状态"])
        v = rng.choice(NEW_COMPANIES if s == "工作" else NEW_CITIES if s == "住处"
                       else NEW_NAMES if s == "称呼" else NEW_REL)
        tpl = {"工作": "我在{v}上班", "住处": "我住在{v}", "称呼": "叫我{v}", "恋爱状态": "我{v}"}[s]
        rows.append(mk([tpl.format(v=v)], {s: v}))
    (OUT / "novel.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

    # 5) 长对话 + 干扰：事实埋在 12-15 行闲聊里
    CHAT = ["今天食堂的菜好咸", "地铁上好挤", "外面下雨了", "刚洗完衣服", "手机快没电了",
            "楼下新开了家面馆", "今天风好大", "猫又把杯子碰倒了", "睡到十点才醒", "看完了那本书",
            "昨晚没睡好", "想去剪头发", "路上堵车", "中午吃了拉面", "忘记带伞了"]
    rows = []
    for _ in range(100):
        s = rng.choice(["工作", "住处", "称呼", "恋爱状态"])
        v = rng.choice(TRAIN_COMPANIES if s == "工作" else TRAIN_CITIES if s == "住处"
                       else TRAIN_NAMES if s == "称呼" else TRAIN_REL)
        tpl = {"工作": "我在{v}上班", "住处": "我住在{v}", "称呼": "叫我{v}", "恋爱状态": "我{v}"}[s]
        lines = [rng.choice(CHAT) for _ in range(rng.randint(8, 12))]
        lines.insert(rng.randint(1, len(lines)), tpl.format(v=v))
        rows.append(mk(lines, {s: v}))
    (OUT / "long.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8")

    for f in sorted(OUT.glob("*.jsonl")):
        n = sum(1 for _ in f.read_text(encoding="utf-8").splitlines() if _.strip())
        print(f"{f.name}: {n} 条")


if __name__ == "__main__":
    main()
