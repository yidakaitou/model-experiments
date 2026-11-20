# -*- coding: utf-8 -*-
"""类人记忆引擎 v2：情景记忆 + 遗忘曲线 + 情绪显著性 + 检索 + 巩固。

与 v1（topic 合并平铺列表）的区别：
- 每条记忆有强度，随时间衰减（Ebbinghaus），被回忆起会加强
- 情绪显著性高的记忆衰减更慢（闪光灯记忆）
- 检索按 近因/相关度/情绪匹配/显著性 加权
- 相似情景可巩固成语义记忆（主题摘要）
"""
from __future__ import annotations

import json
import math
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Iterable

DAY = 86400.0


@dataclass
class Episode:
    id: str
    at: float                 # 发生时间戳
    text: str
    emotion: str = "neutral"  # 主导情绪
    arousal: float = 0.3      # 唤醒度 0-1（越高越难忘）
    valence: float = 0.0      # 效价 -1..1
    participants: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    salience: float = 0.3     # 显著性 0-1
    strength: float = 1.0     # 记忆强度 0-1（会衰减）
    last_recall: float = 0.0
    recalls: int = 0
    source: str = "episodic"

    def tau(self) -> float:
        """时间常数（秒）：越重要越慢忘，高唤醒再加成。"""
        base = 7 * DAY
        t = base * (1.0 + 2.5 * self.salience)
        if self.arousal >= 0.8:
            t *= 3.0  # 闪光灯记忆
        return t


class HumanMemory:
    def __init__(self, path: str):
        self.path = Path(path)
        self.episodes: list[Episode] = []
        self._load()

    # ---------- 持久化 ----------
    def _load(self):
        if not self.path.exists():
            return
        for line in self.path.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
                self.episodes.append(Episode(**d))
            except Exception:
                continue

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            for e in self.episodes:
                f.write(json.dumps(asdict(e), ensure_ascii=False) + "\n")

    # ---------- 写入 ----------
    def add(self, text: str, emotion: str = "neutral", arousal: float = 0.3,
            valence: float = 0.0, participants: Iterable[str] = (), tags: Iterable[str] = (),
            at: float | None = None, salience: float | None = None) -> Episode:
        at = at if at is not None else time.time()
        sal = salience if salience is not None else min(1.0, 0.5 * arousal + 0.3 * abs(valence) + 0.2)
        ep = Episode(id=uuid.uuid4().hex[:8], at=at, text=text, emotion=emotion,
                     arousal=arousal, valence=valence, participants=list(participants),
                     tags=list(tags), salience=sal)
        self.episodes.append(ep)
        self._save()
        return ep

    # ---------- 衰减 ----------
    def decay(self, now: float | None = None) -> int:
        """按经过时间衰减所有记忆强度，返回仍保留的条数。"""
        now = now or time.time()
        alive = []
        for e in self.episodes:
            dt = max(0.0, now - max(e.at, e.last_recall))
            e.strength *= math.exp(-dt / e.tau())
            if e.strength >= 0.02:
                alive.append(e)
        removed = len(self.episodes) - len(alive)
        self.episodes = alive
        self._save()
        return removed

    # ---------- 检索 ----------
    def recall(self, query: str, k: int = 5, now: float | None = None,
               emotion: str | None = None) -> list[tuple[Episode, float]]:
        now = now or time.time()
        q = set(query.lower().split())
        scored = []
        for e in self.episodes:
            rel = self._relevance(q, e.text)
            rec = math.exp(-(now - e.at) / (14 * DAY))
            emo = 1.0 if (emotion and e.emotion == emotion) else 0.0
            score = 0.35 * rec + 0.35 * rel + 0.20 * emo + 0.10 * e.salience
            scored.append((e, score))
        scored.sort(key=lambda x: -x[1])
        top = scored[:k]
        for e, _ in top:
            e.last_recall = now
            e.recalls += 1
            e.strength = min(1.0, e.strength + 0.2)
        if top:
            self._save()
        return top

    @staticmethod
    def _relevance(q: set[str], text: str) -> float:
        if not q:
            return 0.0
        t = set(text.lower().split())
        inter = len(q & t)
        return inter / (len(q) + 1e-6) if inter else 0.0

    # ---------- 巩固 ----------
    def consolidate(self, min_group: int = 3) -> list[dict]:
        """把同 tag 的弱记忆合并成一条语义记忆（摘要），保留最强的作代表。"""
        groups: dict[str, list[Episode]] = {}
        for e in self.episodes:
            for t in e.tags or ["untagged"]:
                groups.setdefault(t, []).append(e)
        out = []
        for tag, eps in groups.items():
            if len(eps) < min_group:
                continue
            if all(e.source == "semantic" for e in eps):
                continue
            best = max(eps, key=lambda e: e.strength * e.salience)
            summary = " / ".join(e.text[:40] for e in sorted(eps, key=lambda x: -x.strength)[:3])
            sem = Episode(id=uuid.uuid4().hex[:8], at=best.at,
                          text=f"[语义] {tag}: {summary}",
                          emotion=best.emotion, arousal=best.arousal,
                          valence=best.valence, tags=[tag],
                          salience=best.salience, strength=best.strength,
                          source="semantic")
            out.append({"tag": tag, "count": len(eps), "summary": sem.text})
            self.episodes.append(sem)
            keep = set(id(e) for e in eps)
            self.episodes = [e for e in self.episodes if id(e) not in keep or e is sem]
        self._save()
        return out

    def stats(self) -> dict:
        return {
            "total": len(self.episodes),
            "semantic": sum(1 for e in self.episodes if e.source == "semantic"),
            "avg_strength": round(sum(e.strength for e in self.episodes) / max(1, len(self.episodes)), 3),
            "strongest": sorted(self.episodes, key=lambda e: -e.strength)[:3] and
                         [(e.text[:30], round(e.strength, 2)) for e in sorted(self.episodes, key=lambda e: -e.strength)[:3]],
        }
