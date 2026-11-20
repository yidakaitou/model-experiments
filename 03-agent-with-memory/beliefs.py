# -*- coding: utf-8 -*-
"""信念层：从情景记忆里提炼出的「我信什么」和「我还不知道什么」。

与 Episode 的区别：
- Episode 是「发生了什么」（原话、时间、情绪）
- Belief 是「因此我推断出什么」（结论、置信度、证据链）
- Question 是「我还没搞清的事」

设计要点：
1. 每条结论必须挂 evidence（episode id），可追溯
2. 置信度会随时间轻微衰减，被反复验证会上升
3. 与旧结论矛盾时不覆盖，标记 conflict 等人裁决
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import re

DAY = 86400.0
CHANGELOG_NAME = "变更日志.jsonl"
MAX_DELTA = 0.15      # 单次置信度变动上限（学自画像）
MAX_ITEMS = 3         # 每轮最多改几条
MIN_USER_MSGS = 5     # 质量闸：至少几条"他的话"
MIN_USER_RATIO = 0.15  # 质量闸：他的话至少占比


def bg(s: str) -> set:
    s = re.sub(r"\s+", "", s or "")
    return {s[i:i + 2] for i in range(len(s) - 1)}


def _norm_slot(s: str) -> str:
    return re.sub(r"\s+", "", str(s or "")).lower()[:20]


def sim(a: str, b: str) -> float:
    A, B = bg(a), bg(b)
    return len(A & B) / max(1, len(A | B))


@dataclass
class Node:
    id: str
    kind: str                 # belief | question
    claim: str
    confidence: float = 0.5   # 0-1（question 恒为 0）
    evidence: list = field(default_factory=list)   # [{"ep": "3f2a", "quote": "原话"}]
    created: float = field(default_factory=time.time)
    updated: float = field(default_factory=time.time)
    recalls: int = 0
    status: str = "active"    # active | conflict | superseded | retired | open | answered
    note: str = ""
    slot: str = ""            # 值型键：工作 / 住处 / 称呼 / 关系 / ""(状态型)
    value: str = ""           # 该槽当前的值，例如 slot=工作 value=银行
    valid_from: float = 0.0
    valid_to: float = 0.0     # 0 = 仍有效
    supersedes: str = ""      # 被这条取代的旧节点 id


class BeliefStore:
    def __init__(self, path: str):
        self.path = Path(path)
        self.nodes: list[Node] = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    d = json.loads(line)
                    d.setdefault("slot", "")
                    d.setdefault("value", "")
                    d.setdefault("valid_from", d.get("created", time.time()))
                    d.setdefault("valid_to", 0.0)
                    d.setdefault("supersedes", "")
                    self.nodes.append(Node(**d))
                except Exception:
                    continue

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for n in self.nodes:
                f.write(json.dumps(asdict(n), ensure_ascii=False) + "\n")

    def _eff_conf(self, n: Node) -> float:
        age = max(0.0, time.time() - n.updated) / DAY
        return max(0.0, n.confidence * (0.97 ** age))

    def _log_change(self, action: str, node_id: str, claim: str, frm, to, why: str, evidence: list):
        try:
            p = self.path.parent / CHANGELOG_NAME
            with p.open("a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "action": action, "id": node_id, "claim": claim,
                    "from": round(frm, 3) if isinstance(frm, float) else frm,
                    "to": round(to, 3) if isinstance(to, float) else to,
                    "why": why, "evidence": [e.get("quote", "")[:40] for e in (evidence or [])][:3],
                }, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def add_or_merge(self, kind: str, claim: str, evidence: list, confidence: float = 0.6,
                     why: str = "", slot: str = "", value: str = "") -> tuple[str, str]:
        """返回 (action, id)：new / merged / conflict。置信度变动受 MAX_DELTA 限制。"""
        # 值型槽位：同槽的**所有**仍有效旧值全部失效（必须在合并判断之前做）
        if slot and value:
            olds = [n for n in self.nodes
                    if n.kind == kind and n.slot and not n.valid_to
                    and _norm_slot(n.slot) == _norm_slot(slot)
                    and n.value and _norm_slot(n.value) != _norm_slot(value)]
            if olds:
                now = time.time()
                for o in olds:
                    o.valid_to = now
                    o.status = "superseded"
                    o.updated = now
                    self._log_change("superseded", o.id, o.claim, o.confidence, None,
                                     f"被新说法取代：{claim[:20]}", evidence)
                new = Node(id=uuid.uuid4().hex[:8], kind=kind, claim=claim,
                           confidence=confidence, evidence=list(evidence), status="active",
                           slot=slot, value=value, valid_from=now,
                           supersedes=",".join(o.id for o in olds))
                self.nodes.append(new)
                self._log_change("new", new.id, claim, None, confidence,
                                 f"取代 {len(olds)} 条旧值：{why or '事实变了'}", evidence)
                return "superseded", new.id

        for n in self.nodes:
            if n.kind != kind or n.status == "retired":
                continue
            s = sim(claim, n.claim)
            if s > 0.62:                      # 同一件事 → 合并证据、抬高置信
                for e in evidence:
                    if e not in n.evidence:
                        n.evidence.append(e)
                _old = n.confidence
                _target = min(0.98, 0.7 * n.confidence + 0.3 * confidence + 0.05)
                _new = max(_old - MAX_DELTA, min(_old + MAX_DELTA, _target))   # 单次不跳变
                n.confidence = _new
                n.updated = time.time()
                self._log_change("merged", n.id, n.claim, _old, _new, why or "重复出现", evidence)
                return "merged", n.id
            neg_a = bool(re.search(r"(不|没|别|从不|讨厌)", claim))
            neg_b = bool(re.search(r"(不|没|别|从不|讨厌)", n.claim))
            if 0.3 < s <= 0.62 and neg_a != neg_b:   # 话题相近但说法相反 → 冲突
                n.status = "conflict"
                new = Node(id=uuid.uuid4().hex[:8], kind=kind, claim=claim,
                           confidence=confidence, evidence=evidence, status="conflict",
                           note=f"与 {n.id} 冲突：{n.claim}")
                self.nodes.append(new)
                return "conflict", new.id
        new = Node(id=uuid.uuid4().hex[:8], kind=kind, claim=claim,
                   confidence=confidence, evidence=evidence,
                   status="open" if kind == "question" else "active",
                   slot=slot, value=value, valid_from=time.time())
        self.nodes.append(new)
        self._log_change("new", new.id, claim, None, confidence, why or "首次提炼", evidence)
        return "new", new.id

    def active(self, kind: str = "") -> list[Node]:
        return [n for n in self.nodes
                if n.status in ("active", "open") and not n.valid_to
                and (not kind or n.kind == kind)]

    def superseded(self) -> list[Node]:
        """已失效但还留着的（可以回答"以前是什么"）。"""
        return [n for n in self.nodes if n.valid_to]

    def conflicts(self) -> list[Node]:
        return [n for n in self.nodes if n.status == "conflict"]

    def prompt_block(self, k: int = 5) -> str:
        """给被测模型注入的文本块：结论 + 置信度，而不是原话。"""
        bl = sorted([n for n in self.active("belief")], key=lambda n: -self._eff_conf(n))[:k]
        qs = [n for n in self.active("question")][:3]
        if not bl and not qs:
            return ""
        lines = []
        if bl:
            lines.append("你对她的判断（按可信度）：")
            for n in bl:
                lines.append(f"- {n.claim}（{self._eff_conf(n):.2f}）")
        if qs:
            lines.append("你还没弄清的事：")
            for n in qs:
                lines.append(f"- {n.claim}")
        return "\n".join(lines)

    def stats(self) -> dict:
        from collections import Counter
        return dict(total=len(self.nodes),
                    by_kind=dict(Counter(n.kind for n in self.nodes)),
                    by_status=dict(Counter(n.status for n in self.nodes)))


def quality_gate(texts, min_user: int = MIN_USER_MSGS, min_ratio: float = MIN_USER_RATIO):
    """巩固前检查：这批数据里"他的原话"够不够。

    返回 (ok, reason)。
    - 条数太少 → 不学
    - 她自己/转述占比过高 → 不学（学自画像那条：拿自己的输出当依据会形成回音壁）
    """
    texts = [str(x).strip() for x in texts if str(x).strip()]
    n = len(texts)
    if n < min_user:
        return False, f"样本太少（{n} < {min_user}）"
    self_like = sum(1 for x in texts if re.match(r"^(\[语义\]|她说[:：]|我说[:：]|我[:：])", x))
    ratio = self_like / n
    if ratio > 1 - min_ratio:
        return False, f"她自己/转述占比 {ratio:.0%}，他的话不足 {min_ratio:.0%} → 不据此学习"
    uniq = len({x[:12] for x in texts})
    if uniq < max(3, n // 4):
        return False, f"内容重复度高（{uniq}/{n} 不同开头）"
    return True, f"通过（{n} 条，自述占比 {ratio:.0%}）"
