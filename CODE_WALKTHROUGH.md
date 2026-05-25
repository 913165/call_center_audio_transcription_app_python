# Call Center Audio Transcription App - Code Walkthrough

This document explains the full logic flow across:
- `call_center_app.py` (Streamlit UI + orchestration)
- `call_center_logic.py` (pure helper utilities and normalization)

## 1) High-Level Architecture

The app processes uploaded call recordings in a batch:
1. User provides an OpenAI API key (manual input or configured secret/environment).
2. User uploads one or more audio files.
3. On `Run Whisper Batch`, each file is:
   - transcribed with Whisper (`audio.transcriptions.create`),
   - converted into segments/text,
   - transformed into pseudo speaker turns,
   - analyzed with GPT (`chat.completions.create`) for QA metrics.
4. Results are stored in `st.session_state["results"]` and rendered in multiple tabs.

`call_center_app.py` handles UI, API calls, and display.
`call_center_logic.py` keeps reusable business logic clean and testable.

## 2) `call_center_logic.py` Walkthrough

### Constants

- `TOPICS`: allowed department labels.
- `OUTCOMES`: allowed call outcomes.

These are the canonical allow-lists used to sanitize model output.

### `estimate_duration_seconds(file_size_bytes, assumed_kbps=96) -> int`

Purpose:
- Estimates audio duration using file size and assumed bitrate.

How:
- Converts kbps to bytes/sec.
- Returns `file_size_bytes / bytes_per_second` with minimum of 1 sec.

Used by:
- Upload preview table and result metadata.

### `format_duration(seconds) -> str`

Purpose:
- Formats duration into `m:ss` or `h:mm:ss`.

Used by:
- Upload table and result records.

### `format_size_mb(file_size_bytes) -> str`

Purpose:
- Converts bytes to megabytes with one decimal place.

### `safe_score(value, default=0) -> int`

Purpose:
- Normalizes score-like values to integer range 0-100.

How:
- Attempts numeric conversion, rounds, falls back to default on errors.
- Clamps value to `[0, 100]`.

Used by:
- `normalize_analysis`
- `average_score`

### `normalize_analysis(data) -> Dict[str, Any]`

Purpose:
- Converts raw model JSON into predictable, validated shape.

Normalization rules:
- `topic` must be in `TOPICS`, else `Complaint`.
- `sentiment` must be one of `positive|negative|mixed`, else `mixed`.
- `outcome` must be in `OUTCOMES`, else `unresolved`.
- `key_phrases` forced to list of up to 3 non-empty strings.
- Scores sanitized by `safe_score`.
- Provides default `agent_assessment` if missing.

This function is critical for robustness because LLM output can drift.

### `average_score(analysis) -> float`

Purpose:
- Computes average of `professionalism`, `empathy`, `resolution` scores.

Returns:
- One decimal float.

### `_seg_get(seg, key, default)`

Purpose:
- Uniform accessor for both dict-like and object-like segment items.

Used by:
- `build_speaker_turns`

### `build_speaker_turns(segments, pause_threshold_seconds=1.0) -> List[Dict[str, Any]]`

Purpose:
- Builds pseudo speaker diarization from Whisper segments.

How:
1. Sorts segments by start time.
2. Starts speaker as `Agent`.
3. If pause gap between segments exceeds threshold, toggles speaker.
4. Merges consecutive segments for same speaker.
5. Returns turn objects with `speaker`, `start`, `end`, `text`.

Note:
- This is heuristic diarization (not true speaker ID model output).

## 3) `call_center_app.py` Walkthrough

### Imports and setup

- Imports Streamlit, plotting, OpenAI client, and helper functions from `call_center_logic.py`.
- Defines `SUPPORTED_TYPES` for uploader file filtering.
- Calls `st.set_page_config(...)`.
- Injects custom CSS for dark-theme UI components.
- Initializes `st.session_state["results"]` if absent.

### Helper functions in app file

#### `_timestamp(seconds) -> str`
- Converts seconds into `mm:ss` or `hh:mm:ss` for transcript timeline display.

#### `fig_to_bytes(fig) -> bytes`
- Serializes Matplotlib figure to PNG bytes so Streamlit can render with `st.image`.

#### `_to_segments_list(verbose_resp) -> List[Dict[str, Any]]`
- Extracts `segments` from Whisper response whether response is object-style or dict-style.
- Produces clean list with `start`, `end`, `text` fields.

#### `_to_text(verbose_resp) -> str`
- Extracts transcript text from Whisper verbose response.

#### `_analyze_transcript(client, transcript_text) -> Dict[str, Any]`

Purpose:
- Sends transcript to GPT model for QA classification/scoring.

Key behavior:
- Uses system prompt with strict schema instructions.
- Uses `response_format={"type": "json_object"}`.
- Truncates transcript to first 16000 chars to control token usage.
- Parses returned JSON text.
- Passes parsed object through `normalize_analysis` for safety.

#### `_process_file(client, uploaded_file) -> Dict[str, Any]`

Purpose:
- Full per-file pipeline function.

Flow:
1. Read uploaded binary data.
2. Transcribe with `client.audio.transcriptions.create(...)` using `whisper-1`.
3. Extract transcript and segments.
4. Build speaker turns (`build_speaker_turns`).
5. Analyze transcript with GPT (`_analyze_transcript`).
6. Return normalized result record with metadata and average score.

#### `_badge_color(outcome)` and `_topic_color(topic)`
- Return color strings for UI accents.
- `_topic_color` is used by topic tab visuals.

### Sidebar logic

The sidebar handles:
- API key resolution inputs.
- Topic filter selection.
- KPI metrics from existing session results.

API key behavior:
- `configured_api_key` tries `st.secrets["OPENAI_API_KEY"]` first, then `os.environ`.
- `api_key` from `st.text_input(..., type="password")` is optional and not prefilled.
- Effective key is chosen later as `api_key or configured_api_key`.

### Upload area and pre-run table

- `st.file_uploader(..., accept_multiple_files=True)` allows multiple recordings.
- When files exist, app computes:
  - estimated duration (size heuristic),
  - file size label,
  - date/status.
- Displays summary metrics + table before processing.

### Batch execution (`Run Whisper Batch`)

Trigger:
- `run_batch = st.button("Run Whisper Batch", ...)`

Validation order:
1. Missing API key -> error.
2. Missing uploads -> error.
3. Else process all files.

Processing behavior:
- Creates `OpenAI(api_key=effective_api_key)` client.
- Shows progress bar and per-file status/spinner.
- Wraps each file in `try/except` so one failure does not stop batch.
- Collects successful results in `new_results`.

Merge behavior:
- Existing results are keyed by filename.
- New results overwrite same filename entries.
- Final list written back to `st.session_state["results"]`.

### Results filtering

- Reads `results = st.session_state["results"]`.
- Applies topic filter if selected topic is not `All`.

### Results tabs

When results exist, app shows six tabs:

1. `Sentiment`
- Card per file with sentiment label and three score bars.

2. `Topic Classification`
- Per-file topic row, confidence percent, transcript snippet.
- Side panel with topic distribution bars.

3. `Speaker Diarization`
- Prints speaker turns with timestamps.
- Falls back to info message when no turns exist.

4. `Agent Performance`
- DataFrame with key metrics and GPT assessment.
- Conditional color styling on average score.

5. `Key Phrases`
- Builds aggregate word cloud from `key_phrases`.
- Also lists per-file phrases.

6. `Transcripts`
- File picker.
- Shows turn-by-turn transcript with speaker color coding.
- Falls back to plain transcript text if no turns.

If there are no processed results, the app shows a guidance info message.

## 4) Data Contract for a Result Item

Each processed file produces a dict shaped like:

```python
{
  "filename": str,
  "size_bytes": int,
  "size_label": str,
  "estimated_duration": str,
  "status": "DONE",
  "date": "YYYY-MM-DD",
  "transcript": str,
  "segments": List[Dict[str, Any]],
  "turns": List[Dict[str, Any]],
  "analysis": {
    "topic": str,
    "topic_confidence": float,
    "sentiment": str,
    "professionalism_score": int,
    "empathy_score": int,
    "resolution_score": int,
    "outcome": str,
    "key_phrases": List[str],
    "agent_assessment": str,
  },
  "avg_score": float,
}
```

This shape powers all dashboard tabs and metrics.

## 5) Error Handling and Resilience

- Per-file `try/except` in batch loop prevents full-batch failure.
- `normalize_analysis` guards against malformed/missing model fields.
- Helper defaults avoid crashes on absent segment/text values.
- UI shows user-friendly `st.error` or `st.info` states.

## 6) Security Notes

- API key input is masked (`type="password"`).
- The field is intentionally not prefilled with secret values.
- Secrets can be sourced from `st.secrets` or `OPENAI_API_KEY` env var.
- Sensitive files should be ignored in `.gitignore` (for example `.streamlit/secrets.toml`, `.env*`).

## 7) End-to-End Runtime Trace (Short)

1. App loads and initializes session state.
2. User sets key context and uploads files.
3. User clicks run.
4. Each file is transcribed, analyzed, normalized, and stored.
5. Results are merged into session cache.
6. Dashboard tabs render from the same normalized result structure.

## 8) Coverage Checklist

- [x] All functions in `call_center_logic.py` documented.
- [x] All helper functions in `call_center_app.py` documented.
- [x] Sidebar, upload table, batch flow, and state merge covered.
- [x] Topic filtering and all six result tabs covered.
- [x] Result data shape and error/security behavior covered.

