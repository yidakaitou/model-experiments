# -*- coding: utf-8 -*-
"""记忆 v3：在 v2 之上加 Belief / Question / Conjecture，并做好奇心调度。

认知循环：
  经历 → Episode
  巩固 → 发现矛盾/低置信/高惊讶 → Question
  调度 → 按 好奇心×(1-擅长) 排序
  猜想 → Conjecture（待验证）
  验证 → 更新 Belief / 关闭 Question
"""
from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

from memory_engine_v2 import HumanMemory, Episode, DAY


@dataclass
class Belief:
    id: str
    topic: str
    statement: str
    confidence: float = 0.5
    evidence: list[str] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)
    source: str = "belief"


@dataclass
class Question:
    id: str
    topic: str
    text: str
    curiosity: float = 0.5     # 好奇度：越大越想搞懂
    competence: float = 0.5    # 擅长度：该主题历史正确率
    value: float = 0.5         # 价值：和当前目标的相关性
    status: str = "open"       # open / resolved / dropped
    evidence: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    resolved_at: float = 0.0

    def priority(self) -> float:
        """学习区优先级：好奇高、擅长低的地方最高。"""
        return self.curiosity * (1.0 - self.competence) * self.value


@dataclass
class Conjecture:
    id: str
    question_id: str
    statement: str
    confidence: float = 0.4
    evidence: list[str] = field(default_factory=list)
    status: str = "pending"    # pending / confirmed / refuted
    created_at: float = field(default_factory=time.time)


class CognitiveMemory:
    def __init__(self, path: str):
        self.path = Path(path)
        self.mem = HumanMemory(str(self.path.with_suffix(".episodes.jsonl")))
        self.beliefs: list[Belief] = []
        self.questions: list[Question] = []
        self.conjectures: list[Conjecture] = []
        self._load()

    # ---------- 持久化 ----------
    def _load(self):
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            d = json.loads(line)
            kind = d.get("kind")
            d.pop("kind", None)
            if kind == "belief":
                self.beliefs.append(Belief(**d))
            elif kind == "question":
                self.questions.append(Question(**d))
            elif kind == "conjecture":
                self.conjectures.append(Conjecture(**d))

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for b in self.beliefs:
                f.write(json.dumps({"kind": "belief", **asdict(b)}, ensure_ascii=False) + "\n")
            for q in self.questions:
                f.write(json.dumps({"kind": "question", **asdict(q)}, ensure_ascii=False) + "\n")
            for c in self.conjectures:
                f.write(json.dumps({"kind": "conjecture", **asdict(c)}, ensure_ascii=False) + "\n")

    # ---------- 写入 ----------
    def add_belief(self, topic: str, statement: str, confidence: float = 0.5,
                   evidence: Iterable[str] = ()) -> Belief:
        b = Belief(id=uuid.uuid4().hex[:8], topic=topic, statement=statement,
                   confidence=confidence, evidence=list(evidence))
        self.beliefs.append(b)
        self._save()
        return b

    def add_question(self, topic: str, text: str, curiosity: float = 0.5,
                     competence: float = 0.5, value: float = 0.5,
                     evidence: Iterable[str] = ()) -> Question:
        q = Question(id=uuid.uuid4().hex[:8], topic=topic, text=text,
                     curiosity=curiosity, competence=competence, value=value,
                     evidence=list(evidence))
        self.questions.append(q)
        self._save()
        return q

    def add_conjecture(self, question_id: str, statement: str,
                       confidence: float = 0.4, evidence: Iterable[str] = ()) -> Conjecture:
        c = Conjecture(id=uuid.uuid4().hex[:8], question_id=question_id,
                       statement=statement, confidence=confidence, evidence=list(evidence))
        self.conjectures.append(c)
        self._save()
        return c

    # ---------- 从记忆里“发现问题” ----------
    def mine_questions(self, limit: int = 5) -> list[Question]:
        """从情景里挖问题：矛盾、高唤醒异常、低置信主题。"""
        made = []
        # 1) 同 tag 情绪反复切换 → 矛盾
        by_tag: dict[str, list[Episode]] = {}
        for e in self.mem.episodes:
            for t in e.tags or []:
                by_tag.setdefault(t, []).append(e)
        for tag, eps in by_tag.items():
            emos = {e.emotion for e in eps}
            if len(emos) >= 2 and len(eps) >= 2:
                if any(x.topic == tag and x.status == "open" for x in self.questions):
                    continue
                q = self.add_question(tag, f"为什么「{tag}」里的情绪会反复变化？",
                                      curiosity=0.8, competence=0.4, value=0.7,
                                      evidence=[e.id for e in eps[:3]])
                made.append(q)
        # 2) 高唤醒但只出现一次 → 异常
        from collections import Counter
        tag_count = Counter(t for e in self.mem.episodes for t in e.tags or [])
        for e in self.mem.episodes:
            if e.arousal >= 0.8 and tag_count.get(e.tags[0] if e.tags else "", 0) <= 1:
                q = self.add_question(e.tags[0] if e.tags else "untagged",
                                      f"「{e.text[:30]}」为什么让我这么激动？",
                                      curiosity=0.9, competence=0.3, value=0.6,
                                      evidence=[e.id])
                made.append(q)
        return made[:limit]

    # ---------- 调度 ----------
    def schedule(self, k: int = 3) -> list[Question]:
        open_q = [q for q in self.questions if q.status == "open"]
        open_q.sort(key=lambda q: -q.priority())
        return open_q[:k]

    # ---------- 解决 ----------
    def resolve(self, question_id: str, belief_statement: str | None = None,
                competence_delta: float = 0.15) -> dict:
        q = next((x for x in self.questions if x.id == question_id), None)
        if not q:
            return {"ok": False, "reason": "not found"}
        q.status = "resolved"
        q.resolved_at = time.time()
        q.competence = min(1.0, q.competence + competence_delta)
        for other in self.questions:
            if other.topic == q.topic:
                other.competence = min(1.0, other.competence + competence_delta * 0.5)
        if belief_statement:
            self.add_belief(q.topic, belief_statement, confidence=0.7,
                            evidence=q.evidence)
        self._save()
        return {"ok": True, "question": q.text, "competence": q.competence}

    # ---------- 统计 ----------
    def stats(self) -> dict:
        return {
            "episodes": len(self.mem.episodes),
            "beliefs": len(self.beliefs),
            "questions_open": sum(1 for q in self.questions if q.status == "open"),
            "questions_resolved": sum(1 for q in self.questions if q.status == "resolved"),
            "conjectures": len(self.conjectures),
            "top_priority": [(q.text[:30], round(q.priority(), 2)) for q in self.schedule(3)],
        }
