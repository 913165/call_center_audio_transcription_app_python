from __future__ import annotations

import html
import io
import json
import os
from collections import Counter
from datetime import date
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from openai import OpenAI
from wordcloud import WordCloud

from call_center_logic import (
    TOPICS,
    average_score,
    build_speaker_turns,
    estimate_duration_seconds,
    format_duration,
    format_size_mb,
    normalize_analysis,
)

SUPPORTED_TYPES = ["mp3", "wav", "mp4", "ogg", "flac", "webm", "m4a"]

st.set_page_config(page_title="OpenAI Audio Analysis", page_icon="🎧", layout="wide")

st.markdown(
    """
<style>
  .stApp { background: linear-gradient(180deg, #04070c 0%, #050912 100%); color: #e8f7f3; }
  .block-container { padding-top: 1.2rem; }
  .hero { border: 1px solid #0a8f84; border-radius: 14px; padding: 1rem 1.2rem; background: rgba(6, 20, 24, 0.8); }
  .badge { display:inline-block; padding:2px 10px; border:1px solid #0d8578; border-radius:999px; color:#40e0c7; font-size:0.78rem; }
  .kpi-row { border: 1px solid #123942; border-radius: 12px; padding: 0.5rem 0.8rem; background: rgba(4, 17, 22, 0.8); }
  .turn-agent { color: #45f4d2; }
  .turn-customer { color: #f6f18f; }
  .sentiment-card {
    background: rgba(14, 18, 24, 0.9);
    border: 1px solid #2a313d;
    border-top-width: 3px;
    border-radius: 12px;
    padding: 0.8rem 0.9rem 0.9rem 0.9rem;
    margin-bottom: 0.8rem;
  }
  .sentiment-file { color: #7f8da0; font-size: 0.72rem; margin-bottom: 0.25rem; }
  .sentiment-label { font-size: 1.1rem; font-weight: 700; margin-bottom: 0.45rem; }
  .score-row {
    display: grid;
    grid-template-columns: 95px 1fr 28px;
    align-items: center;
    gap: 8px;
    margin: 0.16rem 0;
    font-size: 0.74rem;
    color: #c8d2df;
  }
  .score-track { height: 5px; border-radius: 999px; background: #2d3442; overflow: hidden; }
  .score-fill { height: 100%; border-radius: 999px; background: #26d9a6; }
  .score-value { color: #8b97a8; font-size: 0.72rem; text-align: right; }
  .topic-row {
    border: 1px solid #1b3047;
    border-radius: 11px;
    background: rgba(8, 14, 25, 0.9);
    padding: 0.62rem 0.82rem;
    margin-bottom: 0.42rem;
  }
  .topic-row-main {
    display: grid;
    grid-template-columns: minmax(180px, 1.4fr) auto minmax(260px, 3fr) auto;
    align-items: center;
    gap: 10px;
  }
  .topic-file { color: #8da3bd; font-size: 0.85rem; font-family: monospace; }
  .topic-pill {
    padding: 2px 10px;
    border-radius: 8px;
    font-size: 0.78rem;
    border: 1px solid;
    font-weight: 600;
    white-space: nowrap;
  }
  .topic-snippet {
    color: #9eb2ca;
    font-size: 0.82rem;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }
  .topic-confidence { color: #6f89a8; font-size: 0.86rem; font-weight: 600; }
  .breakdown-panel {
    border: 1px solid #1b3047;
    border-radius: 11px;
    background: rgba(8, 14, 25, 0.9);
    padding: 0.85rem 0.85rem;
  }
  .breakdown-title {
    color: #6f89a8;
    letter-spacing: 1px;
    font-size: 0.78rem;
    font-weight: 700;
    margin-bottom: 0.7rem;
    text-transform: uppercase;
  }
  .break-row { margin-bottom: 0.45rem; }
  .break-label { color: #9eb2ca; font-size: 0.9rem; margin-bottom: 0.18rem; }
  .break-track { height: 3px; border-radius: 999px; background: #1f2d43; overflow: hidden; }
  .break-fill { height: 100%; border-radius: 999px; }
</style>
""",
    unsafe_allow_html=True,
)

if "results" not in st.session_state:
    st.session_state["results"] = []


# ---------- helpers ----------
def _timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def fig_to_bytes(fig: Any) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#050912")
    buf.seek(0)
    return buf.getvalue()


def _to_segments_list(verbose_resp: Any) -> List[Dict[str, Any]]:
    segments = getattr(verbose_resp, "segments", None)
    if segments is None and isinstance(verbose_resp, dict):
        segments = verbose_resp.get("segments", [])
    if segments is None:
        return []

    cleaned: List[Dict[str, Any]] = []
    for seg in segments:
        if isinstance(seg, dict):
            cleaned.append(seg)
        else:
            cleaned.append(
                {
                    "start": getattr(seg, "start", 0.0),
                    "end": getattr(seg, "end", 0.0),
                    "text": getattr(seg, "text", ""),
                }
            )
    return cleaned


def _to_text(verbose_resp: Any) -> str:
    text = getattr(verbose_resp, "text", None)
    if text is None and isinstance(verbose_resp, dict):
        text = verbose_resp.get("text", "")
    return str(text or "").strip()


def _analyze_transcript(client: OpenAI, transcript_text: str) -> Dict[str, Any]:
    system_prompt = (
        "You are a QA auditor for call center conversations. Return only valid JSON with keys: "
        "topic, topic_confidence, sentiment, professionalism_score, empathy_score, "
        "resolution_score, outcome, key_phrases, agent_assessment.\n"
        "topic must be one of: Billing, Tech Support, Refund, Sales, Escalation, Complaint.\n"
        "sentiment must be one of: positive, negative, mixed.\n"
        "outcome must be one of: resolved, callback, escalated, unresolved.\n"
        "professionalism_score, empathy_score, resolution_score are integers 0-100.\n"
        "topic_confidence is a float between 0 and 1.\n"
        "key_phrases is a list of exactly 3 short strings.\n"
        "agent_assessment must be one sentence, max 15 words."
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Analyze this call transcript:\n\n{transcript_text[:16000]}",
            },
        ],
    )

    raw = response.choices[0].message.content or "{}"
    return normalize_analysis(json.loads(raw))


def _process_file(client: OpenAI, uploaded_file: Any) -> Dict[str, Any]:
    data = uploaded_file.getvalue()
    file_tuple = (uploaded_file.name, io.BytesIO(data), uploaded_file.type or "audio/mpeg")

    verbose = client.audio.transcriptions.create(
        model="whisper-1",
        file=file_tuple,
        response_format="verbose_json",
        timestamp_granularities=["word", "segment"],
    )

    transcript_text = _to_text(verbose)
    segments = _to_segments_list(verbose)
    turns = build_speaker_turns(segments, pause_threshold_seconds=1.0)
    analysis = _analyze_transcript(client, transcript_text)

    return {
        "filename": uploaded_file.name,
        "size_bytes": len(data),
        "size_label": format_size_mb(len(data)),
        "estimated_duration": format_duration(estimate_duration_seconds(len(data))),
        "status": "DONE",
        "date": str(date.today()),
        "transcript": transcript_text,
        "segments": segments,
        "turns": turns,
        "analysis": analysis,
        "avg_score": average_score(analysis),
    }


def _badge_color(outcome: str) -> str:
    mapping = {
        "resolved": "#00d2a0",
        "callback": "#ffb020",
        "escalated": "#b780ff",
        "unresolved": "#ff4d5e",
    }
    return mapping.get(outcome, "#6c7685")


def _topic_color(topic: str) -> str:
    topic_colors = {
        "Tech Support": "#21c7ff",
        "Billing": "#ff5f77",
        "Sales": "#1dde99",
        "Complaint": "#ff7d93",
        "Refund": "#f0cb4a",
        "Escalation": "#9c74ff",
    }
    return topic_colors.get(topic, "#6f89a8")


# ---------- sidebar ----------
with st.sidebar:
    st.header("Settings")

    configured_api_key = str(
        st.secrets.get("OPENAI_API_KEY", "") or os.environ.get("OPENAI_API_KEY", "")
    ).strip()
    api_key = st.text_input(
        "OpenAI API Key (optional)",
        type="password",
        value="",
        placeholder="sk-...",
        help="Leave empty to use a configured key from st.secrets or OPENAI_API_KEY.",
    ).strip()

    if api_key:
        st.caption("Using API key entered in this session.")
    elif configured_api_key:
        st.caption("Using API key from configured secrets/environment.")

    all_topics = ["All"] + TOPICS
    selected_topic = st.selectbox("Filter by department/topic", all_topics)

    results_all = st.session_state["results"]
    avg_overall = round(sum(r.get("avg_score", 0) for r in results_all) / max(len(results_all), 1), 1)
    resolved_count = sum(1 for r in results_all if r["analysis"].get("outcome") == "resolved")
    escalated_count = sum(1 for r in results_all if r["analysis"].get("outcome") == "escalated")

    st.markdown("---")
    st.metric("Total files processed", len(results_all))
    st.metric("Resolved", resolved_count)
    st.metric("Escalated", escalated_count)
    st.metric("Average overall score", f"{avg_overall}")


# ---------- header ----------
st.markdown("## OpenAI Audio Analysis")
st.caption("Call Center Batch Pipeline - Whisper + GPT-4o")

st.markdown('<div class="hero">', unsafe_allow_html=True)
uploaded_files = st.file_uploader(
    "Browse call recordings",
    type=SUPPORTED_TYPES,
    accept_multiple_files=True,
)

if uploaded_files:
    st.markdown(f"<span class='badge'>{len(uploaded_files)} files ready</span>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ---------- upload table ----------
if uploaded_files:
    table_rows = []
    total_size = 0
    for f in uploaded_files:
        fsize = len(f.getvalue())
        total_size += fsize
        table_rows.append(
            {
                "Recording file": f.name,
                "Estimated duration": format_duration(estimate_duration_seconds(fsize)),
                "Size": format_size_mb(fsize),
                "Date": str(date.today()),
                "Status": "Ready",
            }
        )

    c1, c2, c3 = st.columns([1, 1, 2])
    c1.metric("Recordings", len(uploaded_files))
    c2.metric("Size", format_size_mb(total_size))
    c3.caption("Whisper transcription + GPT-4o classification and scoring")
    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

# ---------- batch run ----------
run_batch = st.button("Run Whisper Batch", type="primary", use_container_width=True)
if run_batch:
    effective_api_key = api_key or configured_api_key
    if not effective_api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
    elif not uploaded_files:
        st.error("Please upload at least one audio file.")
    else:
        client = OpenAI(api_key=effective_api_key)
        progress = st.progress(0)
        status = st.empty()
        new_results = []

        for idx, uploaded_file in enumerate(uploaded_files, start=1):
            status.info(f"Processing {uploaded_file.name} ({idx}/{len(uploaded_files)})")
            with st.spinner(f"Analyzing {uploaded_file.name}..."):
                try:
                    result = _process_file(client, uploaded_file)
                    new_results.append(result)
                except Exception as exc:
                    st.error(f"{uploaded_file.name}: {exc}")
            progress.progress(idx / len(uploaded_files))

        merged = {r["filename"]: r for r in st.session_state["results"]}
        for item in new_results:
            merged[item["filename"]] = item
        st.session_state["results"] = list(merged.values())
        status.success("Batch completed.")


# ---------- results ----------
results = st.session_state["results"]
if selected_topic != "All":
    results = [r for r in results if r["analysis"].get("topic") == selected_topic]

if results:
    st.markdown("---")
    st.subheader("Analysis Results")

    outcome_counts = Counter(r["analysis"].get("outcome", "unresolved") for r in results)
    oc1, oc2, oc3, oc4 = st.columns(4)
    oc1.metric("Resolved", outcome_counts.get("resolved", 0))
    oc2.metric("Callback", outcome_counts.get("callback", 0))
    oc3.metric("Escalated", outcome_counts.get("escalated", 0))
    oc4.metric("Unresolved", outcome_counts.get("unresolved", 0))

    t1, t2, t3, t4, t5, t6 = st.tabs(
        [
            "😊 Sentiment",
            "🏷️ Topic Classification",
            "🎭 Speaker Diarization",
            "🧑‍💼 Agent Performance",
            "💬 Key Phrases",
            "📄 Transcripts",
        ]
    )

    with t1:
        sentiment_ui = {
            "positive": {"emoji": "😊", "label": "Positive", "color": "#1dd7a2", "border": "#1dd7a2"},
            "negative": {"emoji": "😟", "label": "Negative", "color": "#ff5b63", "border": "#ff5b63"},
            "mixed": {"emoji": "😐", "label": "Mixed", "color": "#f3c343", "border": "#f3c343"},
        }

        for start in range(0, len(results), 2):
            row_items = results[start : start + 2]
            columns = st.columns(2)
            for idx, item in enumerate(row_items):
                a = item["analysis"]
                sentiment = a["sentiment"]
                ui = sentiment_ui.get(sentiment, sentiment_ui["mixed"])

                pro = int(a["professionalism_score"])
                emp = int(a["empathy_score"])
                res = int(a["resolution_score"])

                with columns[idx]:
                    st.markdown(
                        f"""
<div class="sentiment-card" style="border-top-color:{ui['border']};">
  <div class="sentiment-file">{item['filename']}</div>
  <div class="sentiment-label" style="color:{ui['color']};">{ui['emoji']} {ui['label']}</div>

  <div class="score-row">
    <span>Professionalism</span>
    <div class="score-track"><div class="score-fill" style="width:{pro}%;"></div></div>
    <span class="score-value">{pro}</span>
  </div>
  <div class="score-row">
    <span>Empathy</span>
    <div class="score-track"><div class="score-fill" style="width:{emp}%;"></div></div>
    <span class="score-value">{emp}</span>
  </div>
  <div class="score-row">
    <span>Resolution</span>
    <div class="score-track"><div class="score-fill" style="width:{res}%;"></div></div>
    <span class="score-value">{res}</span>
  </div>
</div>
""",
                        unsafe_allow_html=True,
                    )
            st.markdown("---")

    with t2:
        left_col, right_col = st.columns([5, 1])

        with left_col:
            for item in results:
                topic = str(item["analysis"].get("topic", "Complaint"))
                confidence = float(item["analysis"].get("topic_confidence", 0.75))
                confidence_pct = max(0, min(100, int(round(confidence * 100))))
                snippet_raw = item.get("transcript", "")
                snippet = snippet_raw[:95] + ("..." if len(snippet_raw) > 95 else "")
                topic_color = _topic_color(topic)

                st.markdown(
                    f"""
<div class="topic-row">
  <div class="topic-row-main">
    <div class="topic-file">{html.escape(item['filename'])}</div>
    <span class="topic-pill" style="color:{topic_color}; border-color:{topic_color}55; background:{topic_color}1A;">{html.escape(topic)}</span>
    <div class="topic-snippet">{html.escape(snippet) if snippet else 'No transcript snippet available.'}</div>
    <div class="topic-confidence">{confidence_pct}%</div>
  </div>
</div>
""",
                    unsafe_allow_html=True,
                )

        with right_col:
            topic_counts = Counter(str(item["analysis"].get("topic", "Complaint")) for item in results)
            total = max(1, sum(topic_counts.values()))

            st.markdown('<div class="breakdown-panel">', unsafe_allow_html=True)
            st.markdown('<div class="breakdown-title">Topic Breakdown</div>', unsafe_allow_html=True)

            for topic in TOPICS:
                count = topic_counts.get(topic, 0)
                pct = int(round((count / total) * 100))
                color = _topic_color(topic)
                st.markdown(
                    f"""
<div class="break-row">
  <div class="break-label">{html.escape(topic)}</div>
  <div class="break-track"><div class="break-fill" style="width:{pct}%; background:{color};"></div></div>
</div>
""",
                    unsafe_allow_html=True,
                )

            st.markdown('</div>', unsafe_allow_html=True)

    with t3:
        for item in results:
            st.markdown(f"**{item['filename']}**")
            if not item["turns"]:
                st.info("No segments available.")
                continue
            for turn in item["turns"]:
                speaker = turn["speaker"]
                css_class = "turn-agent" if speaker == "Agent" else "turn-customer"
                st.markdown(
                    f"<span class='{css_class}'><strong>{speaker}</strong> [{_timestamp(turn['start'])}]</span>: {turn['text']}",
                    unsafe_allow_html=True,
                )
            st.markdown("---")

    with t4:
        perf_rows = []
        for item in results:
            a = item["analysis"]
            perf_rows.append(
                {
                    "File": item["filename"],
                    "Agent": item["filename"],
                    "Professionalism": a["professionalism_score"],
                    "Empathy": a["empathy_score"],
                    "Resolution": a["resolution_score"],
                    "Average Score": item["avg_score"],
                    "Outcome": a["outcome"],
                    "GPT-4o Assessment": a["agent_assessment"],
                }
            )

        df = pd.DataFrame(perf_rows)

        def avg_style(v: float) -> str:
            if v >= 85:
                return "background-color: #143a2f; color: #9ef3cf;"
            if v >= 70:
                return "background-color: #3d3214; color: #ffe291;"
            return "background-color: #421d22; color: #ffadad;"

        st.dataframe(df.style.map(avg_style, subset=["Average Score"]), use_container_width=True, hide_index=True)

    with t5:
        all_phrases: List[str] = []
        for item in results:
            all_phrases.extend(item["analysis"].get("key_phrases", []))

        if all_phrases:
            wc = WordCloud(width=1200, height=400, background_color="#050912", colormap="viridis").generate(
                " ".join(all_phrases)
            )
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.image(fig_to_bytes(fig), use_container_width=True)
            plt.close(fig)
        else:
            st.info("No key phrases yet.")

        st.markdown("#### Per-file key phrases")
        for item in results:
            st.write(f"**{item['filename']}**: {', '.join(item['analysis'].get('key_phrases', []))}")

    with t6:
        picked = st.selectbox("Select a file", [r["filename"] for r in results])
        selected = next((r for r in results if r["filename"] == picked), None)
        if selected:
            for turn in selected["turns"]:
                color = "#43f0c8" if turn["speaker"] == "Agent" else "#f8e86f"
                st.markdown(
                    f"<p><span style='color:{color};font-weight:700'>{turn['speaker']} [{_timestamp(turn['start'])}]</span><br>{turn['text']}</p>",
                    unsafe_allow_html=True,
                )
            if not selected["turns"]:
                st.write(selected["transcript"])
else:
    st.info("No processed results yet. Upload files and click 'Run Whisper Batch'.")
