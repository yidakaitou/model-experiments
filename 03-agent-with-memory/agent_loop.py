# -*- coding: utf-8 -*-
"""认知闭环 v1（规则版）：感知 → 情绪 → 状态 → 记忆 → 问题 → 猜想 → 验证 → 信念 → 状态反馈。

目标不是效果，是把逻辑跑通：每一环的输出是下一环的输入，形成可循环的回路。
后续可逐环替换为学习组件（情绪模型、策略头、LLM 猜想）。
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from cognitive_memory import CognitiveMemory

DAY = 86400.0

TOPIC_RULES = {
    "关系": ["miss", "love", "together", "leave", "disappear", "ignore", "quiet", "busy"],
    "冲突": ["angry", "mad", "yell", "stupid", "hate", "fight", "whatever"],
    "道歉": ["sorry", "apolog", "forgive", "mean to"],
    "离别": ["goodbye", "sleep", "go", "bye", "leave"],
    "自我": ["i feel", "i think", "i am", "myself", "scared", "afraid"],
    "日常": ["eat", "weather", "work", "tired", "food"],
}

def topic_of(text: str) -> str:
    low = text.lower()
    best, hit = "对话", 0
    for topic, words in TOPIC_RULES.items():
        n = sum(1 for w in words if w in low)
        if n > hit:
            best, hit = topic, n
    return best


# ---------------- 情绪后端 ----------------
class EmotionBackend:
    """优先用 109M 脉冲模型，失败退回关键词规则。"""

    LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

    def __init__(self, model_dir=r"D:\dsh-lab\emotion-dl", use_model=True):
        self.model = None
        if use_model:
            try:
                import sys
                import torch
                from transformers import AutoTokenizer
                from peft import LoraConfig, get_peft_model
                sys.path.insert(0, r"D:\dsh-lab\spiking-emotion\model")
                from spiking_bert import BertForMaskedLM
                from spikingjelly.activation_based import neuron, functional

                self.torch = torch
                self.functional = functional
                self.neuron = neuron
                self.tok = AutoTokenizer.from_pretrained(r"D:\dsh-lab\spiking-emotion\model")
                raw = BertForMaskedLM.from_pretrained(
                    r"D:\dsh-lab\spiking-emotion\model", dtype=torch.float32,
                    attn_implementation="eager")
                lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
                                  target_modules=["query", "key", "value", "dense"],
                                  task_type="FEATURE_EXTRACTION")
                self.model = get_peft_model(raw, lora)
                self.base = self.model.base_model.model
                import torch.nn as nn
                head = nn.Sequential(
                    nn.Linear(768, 512), neuron.LIFNode(tau=2.0, v_threshold=1.0, detach_reset=True, step_mode="s"),
                    nn.Dropout(0.2), nn.Linear(512, 256),
                    neuron.LIFNode(tau=2.0, v_threshold=1.0, detach_reset=True, step_mode="s"),
                    nn.Dropout(0.2), nn.Linear(256, 6))
                ck = torch.load(os.path.join(model_dir, "memory_stage2.pt"), map_location="cuda")
                self.model.load_state_dict(ck["model"], strict=False)
                head.load_state_dict(ck["head"])
                self.model = self.model.cuda().eval()
                self.head = head.cuda().eval()
            except Exception as exc:
                print(f"[emotion] 模型加载失败，退回规则：{type(exc).__name__}")
                self.model = None

    def __call__(self, text: str) -> dict:
        if self.model is not None:
            t = self.tok(text, truncation=True, max_length=48, return_tensors="pt").to("cuda")
            self.functional.reset_net(self.model)
            self.functional.reset_net(self.head)
            with self.torch.no_grad():
                h = self.base.bert(**t).last_hidden_state.mean(1)
                p = self.torch.softmax(self.head(h), -1)[0].cpu().tolist()
            i = max(range(6), key=lambda k: p[k])
            return {"label": self.LABELS[i], "p": p[i]}

        low = text.lower()
        rules = {
            "anger": ["angry", "mad", "hate", "yell", "stupid", "idiot"],
            "sadness": ["sad", "cry", "lonely", "miss", "hurt", "ignored"],
            "joy": ["happy", "great", "glad", "haha", "thanks", "love you"],
            "fear": ["afraid", "scared", "worried", "nervous"],
            "surprise": ["wow", "really?", "unexpected", "suddenly"],
            "love": ["love", "care", "together", "hug"],
        }
        for lab, words in rules.items():
            if any(w in low for w in words):
                return {"label": lab, "p": 0.6}
        return {"label": "sadness", "p": 0.3}


# ---------------- 情绪状态 ----------------
@dataclass
class AffectState:
    valence: float = 0.0      # -1..1
    arousal: float = 0.2      # 0..1
    dominance: float = 0.0    # -1..1
    label: str = "neutral"

    def decay(self, factor: float = 0.85):
        self.valence *= factor
        self.arousal = 0.2 + (self.arousal - 0.2) * factor
        self.dominance *= factor

    def update(self, event_valence: float, event_arousal: float,
               memory_valence: float, memory_arousal: float):
        # 加权平均而非累加，避免唤醒度饱和
        target_v = 0.6 * event_valence + 0.4 * memory_valence
        target_a = 0.6 * event_arousal + 0.4 * memory_arousal
        self.valence = max(-1, min(1, 0.65 * self.valence + 0.35 * target_v))
        self.arousal = max(0.05, min(1, 0.65 * self.arousal + 0.35 * target_a))
        self.dominance = max(-1, min(1, 0.8 * self.dominance + 0.2 * event_valence))
        self.label = self._classify()

    def _classify(self) -> str:
        v, a = self.valence, self.arousal
        if a > 0.35 and v < -0.12:
            return "anger"
        if a > 0.35 and v < 0.1 and abs(v) < 0.15:
            return "fear"
        if a > 0.45 and v > 0.15:
            return "excitement"
        if v < -0.18:
            return "sadness"
        if v > 0.18:
            return "joy"
        return "neutral"

    def report(self) -> str:
        return f"我现在有点{self.label}（效价{self.valence:+.2f}，唤醒{self.arousal:.2f}）"


# ---------------- 闭环 ----------------
class CognitiveLoop:
    def __init__(self, mem_path=r"D:\dsh-lab\human-memory\loop_memory.jsonl"):
        self.mem = CognitiveMemory(mem_path)
        self.state = AffectState()
        self.emotion = EmotionBackend()
        self.turn = 0

    def process(self, text: str, speaker: str = "user") -> dict:
        self.turn += 1
        t0 = time.time()

        # 1) 感知：情绪识别
        emo = self.emotion(text)
        label, p = emo["label"], emo["p"]

        # 2) 检索相关记忆（带情绪）
        hits = self.mem.mem.recall(text, k=3, emotion=label)
        mem_val = 0.0
        mem_aro = 0.0
        if hits:
            wsum = sum(max(1e-6, s) for _, s in hits)
            mem_val = sum(e.valence * s for e, s in hits) / wsum
            mem_aro = sum(e.arousal * s for e, s in hits) / wsum

        # 3) 评价：事件对自身的影响
        sign = {"anger": -0.7, "sadness": -0.5, "fear": -0.4,
                "joy": 0.6, "love": 0.7, "surprise": 0.2}[label]
        event_val = sign * (0.5 + 0.5 * p)
        event_aro = 0.65 if label in ("anger", "fear", "surprise") else 0.35

        # 4) 状态更新
        before = (self.state.valence, self.state.arousal)
        self.state.update(event_val, event_aro, mem_val, mem_aro)

        # 5) 写记忆（带情绪与状态）
        ep = self.mem.mem.add(text, emotion=label, arousal=self.state.arousal,
                              valence=self.state.valence, participants=[speaker],
                              tags=[topic_of(text)])

        # 6) 挖问题（每 2 轮一次，避免刷屏）
        new_q = []
        if self.turn % 2 == 0:
            new_q = self.mem.mine_questions(limit=2)

        # 7) 调度
        q = self.mem.schedule(1)
        scheduled = q[0] if q else None

        # 8) 猜想 + 验证（规则版）
        conj = None
        verified = None
        if scheduled:
            conj = self.mem.add_conjecture(
                scheduled.id,
                f"猜测：{scheduled.text} 可能与最近 {len(hits)} 条相关记忆有关",
                confidence=0.4,
                evidence=[e.id for e, _ in hits])
            support = sum(1 for e, s in hits if e.emotion == label and s > 0.3)
            if support >= 2:
                verified = self.mem.resolve(scheduled.id,
                                            belief_statement=f"关于「{scheduled.text}」的猜测被支持")
                conj.status = "confirmed"

        # 9) 状态反馈：解决问题带来正向效价
        if verified:
            self.state.valence = min(1.0, self.state.valence + 0.1)

        return {
            "turn": self.turn,
            "text": text,
            "emotion": label,
            "confidence": round(p, 3),
            "state_before": before,
            "state_after": (round(self.state.valence, 3), round(self.state.arousal, 3)),
            "state_label": self.state.label,
            "report": self.state.report(),
            "memory": ep.id,
            "recalled": [(e.text[:30], round(s, 2)) for e, s in hits],
            "new_questions": [q.text for q in new_q],
            "scheduled": scheduled.text if scheduled else None,
            "conjecture": conj.statement if conj else None,
            "verified": verified,
            "ms": round((time.time() - t0) * 1000),
        }


if __name__ == "__main__":
    import json
    loop = CognitiveLoop()
    convo = [
        "hey, i missed you today",
        "you always say that but then you disappear",
        "i was busy, don't be like that",
        "whatever, i'm going to sleep",
        "sorry, i didn't mean to ignore you",
        "it's fine... i just get scared when you go quiet",
    ]
    for line in convo:
        out = loop.process(line)
        print(json.dumps(out, ensure_ascii=False, indent=1))
