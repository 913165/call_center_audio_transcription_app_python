from __future__ import annotations

from typing import Any, Dict, List

TOPICS = ["Billing", "Tech Support", "Refund", "Sales", "Escalation", "Complaint"]
OUTCOMES = ["resolved", "callback", "escalated", "unresolved"]


def estimate_duration_seconds(file_size_bytes: int, assumed_kbps: int = 96) -> int:
    bytes_per_second = max(1, assumed_kbps * 1000 // 8)
    return max(1, int(file_size_bytes / bytes_per_second))


def format_duration(seconds: int) -> str:
    minutes, sec = divmod(max(0, int(seconds)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}:{minutes:02d}:{sec:02d}"
    return f"{minutes}:{sec:02d}"


def format_size_mb(file_size_bytes: int) -> str:
    return f"{file_size_bytes / (1024 * 1024):.1f} MB"


def safe_score(value: Any, default: int = 0) -> int:
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        parsed = default
    return max(0, min(100, parsed))


def normalize_analysis(data: Dict[str, Any]) -> Dict[str, Any]:
    topic = str(data.get("topic", "Complaint")).strip()
    if topic not in TOPICS:
        topic = "Complaint"

    sentiment = str(data.get("sentiment", "mixed")).lower().strip()
    if sentiment not in {"positive", "negative", "mixed"}:
        sentiment = "mixed"

    outcome = str(data.get("outcome", "unresolved")).lower().strip()
    if outcome not in OUTCOMES:
        outcome = "unresolved"

    key_phrases = data.get("key_phrases", [])
    if not isinstance(key_phrases, list):
        key_phrases = []
    key_phrases = [str(item).strip() for item in key_phrases if str(item).strip()][:3]

    return {
        "topic": topic,
        "topic_confidence": float(data.get("topic_confidence", 0.75)),
        "sentiment": sentiment,
        "professionalism_score": safe_score(data.get("professionalism_score", 0)),
        "empathy_score": safe_score(data.get("empathy_score", 0)),
        "resolution_score": safe_score(data.get("resolution_score", 0)),
        "outcome": outcome,
        "key_phrases": key_phrases,
        "agent_assessment": str(data.get("agent_assessment", "No assessment provided.")).strip(),
    }


def average_score(analysis: Dict[str, Any]) -> float:
    scores = [
        safe_score(analysis.get("professionalism_score", 0)),
        safe_score(analysis.get("empathy_score", 0)),
        safe_score(analysis.get("resolution_score", 0)),
    ]
    return round(sum(scores) / 3.0, 1)


def _seg_get(seg: Any, key: str, default: Any) -> Any:
    if isinstance(seg, dict):
        return seg.get(key, default)
    return getattr(seg, key, default)


def build_speaker_turns(segments: List[Any], pause_threshold_seconds: float = 1.0) -> List[Dict[str, Any]]:
    ordered = sorted(
        segments,
        key=lambda s: float(_seg_get(s, "start", 0.0) or 0.0),
    )
    turns: List[Dict[str, Any]] = []
    current_speaker = "Agent"
    prev_end = None

    for seg in ordered:
        start = float(_seg_get(seg, "start", 0.0) or 0.0)
        end = float(_seg_get(seg, "end", start) or start)
        text = str(_seg_get(seg, "text", "")).strip()
        if not text:
            continue

        if prev_end is not None and (start - prev_end) > pause_threshold_seconds:
            current_speaker = "Customer" if current_speaker == "Agent" else "Agent"

        if turns and turns[-1]["speaker"] == current_speaker:
            turns[-1]["text"] = (turns[-1]["text"] + " " + text).strip()
            turns[-1]["end"] = end
        else:
            turns.append(
                {
                    "speaker": current_speaker,
                    "start": start,
                    "end": end,
                    "text": text,
                }
            )

        prev_end = end

    return turns

