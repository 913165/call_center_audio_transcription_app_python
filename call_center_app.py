from __future__ import annotations

import html
import io
import json
import os
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import date
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from openai import OpenAI
from wordcloud import WordCloud

from app_theme import (
    audio_element_id,
    chart_background,
    render_audio_play_component,
    render_queue_table_component,
    render_themed_table,
    score_cell_style,
    theme_css,
    wordcloud_colormap,
)
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
OPENAI_TIMEOUT_SEC = 180.0
FILE_PROCESS_TIMEOUT_SEC = 300.0

st.set_page_config(
    page_title="Call Center Audio Intelligence",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "results" not in st.session_state:
    st.session_state["results"] = []
if "appearance" not in st.session_state:
    st.session_state["appearance"] = "Dark"
if "queued_files" not in st.session_state:
    st.session_state["queued_files"] = []
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0
if "batch_run_active" not in st.session_state:
    st.session_state["batch_run_active"] = False
if "batch_file_index" not in st.session_state:
    st.session_state["batch_file_index"] = 0
if "batch_errors" not in st.session_state:
    st.session_state["batch_errors"] = []


class _StoredUpload:
    """In-memory upload kept in session state so the file_uploader widget can reset."""

    def __init__(self, name: str, data: bytes, mime: str) -> None:
        self.name = name
        self._data = data
        self.type = mime or "audio/mpeg"

    def getvalue(self) -> bytes:
        return self._data


def _queue_uploads(files: List[Any]) -> None:
    by_name = {item["name"]: item for item in st.session_state["queued_files"]}
    for f in files:
        by_name[f.name] = {
            "name": f.name,
            "data": f.getvalue(),
            "type": f.type or "audio/mpeg",
        }
    st.session_state["queued_files"] = list(by_name.values())


def _queued_upload_objects() -> List[_StoredUpload]:
    return [
        _StoredUpload(item["name"], item["data"], item["type"])
        for item in st.session_state["queued_files"]
    ]


def _audio_for_filename(filename: str) -> tuple[bytes, str] | None:
    for item in st.session_state.get("queued_files", []):
        if item["name"] == filename:
            return item["data"], item.get("type") or "audio/mpeg"
    for result in st.session_state.get("results", []):
        if result["filename"] == filename:
            data = result.get("audio_data")
            if data is not None:
                return data, result.get("audio_type") or "audio/mpeg"
    return None


def _sync_file_status(files: List[_StoredUpload]) -> None:
    names = [f.name for f in files]
    current = st.session_state.get("file_status", {})
    st.session_state["file_status"] = {name: current.get(name, "ready") for name in names}


def _render_queue_table(files: List[_StoredUpload], theme: str, *, max_height_px: int = 280) -> None:
    statuses = st.session_state.get("file_status", {})
    rows: List[Dict[str, Any]] = []
    for sr_no, f in enumerate(files, start=1):
        fsize = len(f.getvalue())
        rows.append(
            {
                "sr": sr_no,
                "name": f.name,
                "data": f.getvalue(),
                "mime": f.type or "audio/mpeg",
                "duration": format_duration(estimate_duration_seconds(fsize)),
                "size": format_size_mb(fsize),
                "date": str(date.today()),
                "status": statuses.get(f.name, "ready"),
            }
        )
    render_queue_table_component(rows, theme, max_height_px=max_height_px)


# ---------- helpers ----------
def _timestamp(seconds: float) -> str:
    total = max(0, int(seconds))
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{sec:02d}"
    return f"{minutes:02d}:{sec:02d}"


def fig_to_bytes(fig: Any, theme: str) -> bytes:
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="png",
        bbox_inches="tight",
        facecolor=chart_background(theme),
    )
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


def _create_openai_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key, timeout=OPENAI_TIMEOUT_SEC, max_retries=2)


def _process_file_with_timeout(client: OpenAI, uploaded_file: Any) -> Dict[str, Any]:
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_process_file, client, uploaded_file)
        try:
            return future.result(timeout=FILE_PROCESS_TIMEOUT_SEC)
        except FuturesTimeoutError as exc:
            raise TimeoutError(
                f"Timed out after {int(FILE_PROCESS_TIMEOUT_SEC)}s — skipping to next file."
            ) from exc


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
        "audio_data": data,
        "audio_type": uploaded_file.type or "audio/mpeg",
        "transcript": transcript_text,
        "segments": segments,
        "turns": turns,
        "analysis": analysis,
        "avg_score": average_score(analysis),
    }


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


def _render_sentiment_grid(results: List[Dict[str, Any]]) -> None:
    sentiment_ui = {
        "positive": {"emoji": "😊", "label": "Positive", "color": "#1dd7a2", "border": "#1dd7a2"},
        "negative": {"emoji": "😟", "label": "Negative", "color": "#ff5b63", "border": "#ff5b63"},
        "mixed": {"emoji": "😐", "label": "Mixed", "color": "#f3c343", "border": "#f3c343"},
    }
    cards: List[str] = []
    for item in results:
        a = item["analysis"]
        sentiment = a["sentiment"]
        ui = sentiment_ui.get(sentiment, sentiment_ui["mixed"])
        pro = int(a["professionalism_score"])
        emp = int(a["empathy_score"])
        res = int(a["resolution_score"])
        cards.append(
            f"""
<div class="sentiment-card" style="border-top-color:{ui['border']};">
  <div class="sentiment-file">{html.escape(item['filename'])}</div>
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
"""
        )
    st.markdown(f'<div class="sentiment-grid">{"".join(cards)}</div>', unsafe_allow_html=True)


def _render_kpi_strip(items: List[tuple[str, str]]) -> None:
    cards = "".join(
        f'<div class="kpi-card"><div class="kpi-value">{value}</div>'
        f'<div class="kpi-label">{label}</div></div>'
        for label, value in items
    )
    st.markdown(f'<div class="kpi-strip">{cards}</div>', unsafe_allow_html=True)


def _render_turn(turn: Dict[str, Any], _theme: str) -> None:
    speaker = turn["speaker"]
    role = "agent" if speaker == "Agent" else "customer"
    css_class = "turn-agent" if speaker == "Agent" else "turn-customer"
    st.markdown(
        f"""
<div class="turn-block {role}">
  <span class="{css_class}"><strong>{html.escape(speaker)}</strong> [{_timestamp(turn['start'])}]</span><br>
  <span style="color: inherit;">{html.escape(turn['text'])}</span>
</div>
""",
        unsafe_allow_html=True,
    )


# ---------- sidebar ----------
with st.sidebar:
    st.markdown(
        """
<div class="sidebar-brand">
  <p class="sidebar-brand-title">🎧 Call Intelligence</p>
  <p class="sidebar-brand-sub">Whisper + GPT-4o pipeline</p>
</div>
""",
        unsafe_allow_html=True,
    )

    st.radio(
        "Appearance",
        options=["Dark", "Light"],
        horizontal=True,
        help="Switch between dark and light interface themes.",
        key="appearance",
    )

    st.markdown('<p class="section-label">API & filters</p>', unsafe_allow_html=True)

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
    selected_topic = st.selectbox("Filter by topic", all_topics)

    results_all = st.session_state["results"]
    avg_overall = round(sum(r.get("avg_score", 0) for r in results_all) / max(len(results_all), 1), 1)
    resolved_count = sum(1 for r in results_all if r["analysis"].get("outcome") == "resolved")
    escalated_count = sum(1 for r in results_all if r["analysis"].get("outcome") == "escalated")

    st.markdown('<p class="section-label">Session summary</p>', unsafe_allow_html=True)
    st.metric("Files processed", len(results_all))
    st.metric("Resolved", resolved_count)
    st.metric("Escalated", escalated_count)
    st.metric("Avg. score", f"{avg_overall}")

current_theme = st.session_state["appearance"].lower()
st.markdown(theme_css(current_theme), unsafe_allow_html=True)

# ---------- header ----------
st.markdown(
    """
<div class="app-header">
  <h1 class="app-title">Call Center <span>Audio Intelligence</span></h1>
  <p class="app-subtitle">Upload recordings, transcribe with Whisper, and score conversations with GPT-4o.</p>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="hero"><div class="hero-label">Upload recordings</div>', unsafe_allow_html=True)

new_uploads = st.file_uploader(
    "Drag and drop call recordings here",
    type=SUPPORTED_TYPES,
    accept_multiple_files=True,
    label_visibility="collapsed",
    key=f"file_uploader_{st.session_state['uploader_key']}",
)

if new_uploads:
    _queue_uploads(new_uploads)
    st.session_state["uploader_key"] += 1
    st.rerun()

queued_files = _queued_upload_objects()

if queued_files:
    badge_col, clear_col = st.columns([4, 1])
    with badge_col:
        st.markdown(
            f"<span class='badge'>{len(queued_files)} file{'s' if len(queued_files) != 1 else ''} queued</span>",
            unsafe_allow_html=True,
        )
    with clear_col:
        if st.button("Clear all", use_container_width=True):
            st.session_state["queued_files"] = []
            st.session_state["file_status"] = {}
            st.session_state["uploader_key"] += 1
            st.rerun()

st.markdown("</div>", unsafe_allow_html=True)

effective_api_key = api_key or configured_api_key
batch_active = bool(st.session_state.get("batch_run_active"))
run_batch = st.button(
    "Run analysis batch",
    type="primary",
    use_container_width=True,
    disabled=batch_active,
)

# ---------- upload table with per-file status dots ----------
if queued_files:
    _sync_file_status(queued_files)
    queue_view_slot = st.empty()

    def _draw_queue_view(*, processing: bool = False) -> None:
        total_size = sum(len(f.getvalue()) for f in queued_files)
        done_count = sum(1 for s in st.session_state["file_status"].values() if s == "done")
        with queue_view_slot.container():
            _render_kpi_strip(
                [
                    ("Recordings", str(len(queued_files))),
                    ("Total size", format_size_mb(total_size)),
                    ("Done", str(done_count)),
                    ("Pipeline status", "Processing…" if processing else "Ready to run"),
                ]
            )
            _render_queue_table(queued_files, current_theme)

    _draw_queue_view(processing=batch_active)

# ---------- batch run (one file per rerun — avoids Streamlit disconnect on long batches) ----------
if run_batch and not batch_active:
    if not effective_api_key:
        st.error("Please enter your OpenAI API key in the sidebar.")
    elif not queued_files:
        st.error("Please upload at least one audio file.")
    else:
        st.session_state["batch_run_active"] = True
        st.session_state["batch_file_index"] = 0
        st.session_state["batch_errors"] = []
        st.session_state["file_status"] = {f.name: "ready" for f in queued_files}
        st.rerun()

if batch_active and queued_files and effective_api_key:
    total_files = len(queued_files)
    idx = int(st.session_state["batch_file_index"])

    if idx < total_files:
        uploaded_file = queued_files[idx]
        label = f"{uploaded_file.name} ({idx + 1}/{total_files})"
        st.session_state["file_status"][uploaded_file.name] = "processing"
        st.progress(max(0.0, idx / total_files), text=label)
        _draw_queue_view(processing=True)

        try:
            client = _create_openai_client(effective_api_key)
            result = _process_file_with_timeout(client, uploaded_file)
            merged = {r["filename"]: r for r in st.session_state["results"]}
            merged[result["filename"]] = result
            st.session_state["results"] = list(merged.values())
            st.session_state["file_status"][uploaded_file.name] = "done"
        except Exception as exc:
            st.session_state["batch_errors"].append((uploaded_file.name, str(exc)))
            st.session_state["file_status"][uploaded_file.name] = "error"

        st.session_state["batch_file_index"] = idx + 1
        st.rerun()

    st.session_state["batch_run_active"] = False
    done_count = sum(1 for s in st.session_state["file_status"].values() if s == "done")
    st.progress(1.0, text=f"Completed {done_count}/{total_files} files")
    _draw_queue_view(processing=False)

    batch_errors = list(st.session_state.get("batch_errors", []))
    if batch_errors:
        for filename, message in batch_errors:
            st.error(f"{filename}: {message}")
        st.warning(f"Batch finished with {len(batch_errors)} error(s). {done_count} file(s) succeeded.")
    else:
        st.success(f"Batch completed — {done_count} file(s) analyzed.")

# ---------- results ----------
results = st.session_state["results"]
if selected_topic != "All":
    results = [r for r in results if r["analysis"].get("topic") == selected_topic]

if results:
    st.markdown("---")
    st.subheader("Analysis results")

    outcome_counts = Counter(r["analysis"].get("outcome", "unresolved") for r in results)
    _render_kpi_strip(
        [
            ("Resolved", str(outcome_counts.get("resolved", 0))),
            ("Callback", str(outcome_counts.get("callback", 0))),
            ("Escalated", str(outcome_counts.get("escalated", 0))),
            ("Unresolved", str(outcome_counts.get("unresolved", 0))),
        ]
    )

    t1, t2, t3, t4, t5, t6 = st.tabs(
        [
            "Sentiment",
            "Topics",
            "Speakers",
            "Agent performance",
            "Key phrases",
            "Transcripts",
        ]
    )

    with t1:
        _render_sentiment_grid(results)

    with t2:
        left_col, right_col = st.columns([6, 1])

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
            st.markdown('<div class="breakdown-title">Topic breakdown</div>', unsafe_allow_html=True)

            for topic in TOPICS:
                count = topic_counts.get(topic, 0)
                pct = int(round((count / total) * 100))
                color = _topic_color(topic)
                st.markdown(
                    f"""
<div class="break-row">
  <div class="break-label"><span>{html.escape(topic)}</span><span>{count}</span></div>
  <div class="break-track"><div class="break-fill" style="width:{pct}%; background:{color};"></div></div>
</div>
""",
                    unsafe_allow_html=True,
                )

            st.markdown("</div>", unsafe_allow_html=True)

    with t3:
        for item in results:
            st.markdown(f"**{item['filename']}**")
            if not item["turns"]:
                st.info("No segments available.")
                continue
            for turn in item["turns"]:
                _render_turn(turn, current_theme)
            st.markdown("---")

    with t4:
        perf_rows = []
        for item in results:
            a = item["analysis"]
            perf_rows.append(
                {
                    "File": item["filename"],
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
            return score_cell_style(current_theme, float(v))

        render_themed_table(
            df,
            current_theme,
            cell_style={"Average Score": avg_style},
            max_height_px=480,
        )

    with t5:
        all_phrases: List[str] = []
        for item in results:
            all_phrases.extend(item["analysis"].get("key_phrases", []))

        if all_phrases:
            wc = WordCloud(
                width=1200,
                height=400,
                background_color=chart_background(current_theme),
                colormap=wordcloud_colormap(current_theme),
            ).generate(" ".join(all_phrases))
            fig, ax = plt.subplots(figsize=(12, 4))
            fig.patch.set_facecolor(chart_background(current_theme))
            ax.set_facecolor(chart_background(current_theme))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.image(fig_to_bytes(fig, current_theme), use_container_width=True)
            plt.close(fig)
        else:
            st.info("No key phrases yet.")

        st.markdown("#### Per-file key phrases")
        for item in results:
            phrases = item["analysis"].get("key_phrases", [])
            st.write(f"**{item['filename']}**: {', '.join(phrases) if phrases else '—'}")

    with t6:
        picked = st.selectbox("Select a file", [r["filename"] for r in results])
        selected = next((r for r in results if r["filename"] == picked), None)
        if selected:
            audio_src = _audio_for_filename(selected["filename"])
            if audio_src:
                data, mime = audio_src
                aid = audio_element_id("transcript", selected["filename"])
                st.caption(f"Recording: {selected['filename']}")
                render_audio_play_component(data, mime, aid, current_theme)
            else:
                st.caption("Audio not available for this file (re-upload to enable playback).")
            if selected["turns"]:
                for turn in selected["turns"]:
                    _render_turn(turn, current_theme)
            else:
                st.write(selected["transcript"])
else:
    st.markdown(
        """
<div class="empty-hint">
  <strong>No results yet</strong>
  Upload call recordings above, then click <em>Run analysis batch</em> to transcribe and score them.
</div>
""",
        unsafe_allow_html=True,
    )
