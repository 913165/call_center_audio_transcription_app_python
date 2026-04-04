Here's a ready-to-paste prompt for GitHub Copilot:

---

**Build a Streamlit call center audio analysis app using OpenAI APIs.**

The app should have the following structure and behavior:

**Upload Section**
Create a file uploader at the top that accepts multiple audio files (mp3, wav, mp4, ogg, flac, webm, m4a). Once files are uploaded, display them in a styled `st.dataframe` or table showing filename, estimated duration (derived from file size), file size, and a "Ready" status badge.

**Batch Processing**
Add a "Run Whisper Batch" button. When clicked, iterate through each uploaded file and do the following for each one using the OpenAI Python SDK:

1. Call `client.audio.transcriptions.create()` with `model="whisper-1"`, `response_format="verbose_json"`, and `timestamp_granularities=["word", "segment"]` to get the full transcript with timestamps.
2. Pass the transcript text to `client.chat.completions.create()` with `model="gpt-4o"` and `response_format={"type": "json_object"}`. The system prompt should instruct the model to return a JSON object containing: `topic` (one of: Billing, Tech Support, Refund, Sales, Escalation, Complaint), `sentiment` (positive, negative, or mixed), `professionalism_score` (0–100), `empathy_score` (0–100), `resolution_score` (0–100), `outcome` (resolved, callback, escalated, or unresolved), `key_phrases` (list of 3 strings), and `agent_assessment` (one sentence, max 15 words).
3. Show a `st.progress` bar updating as each file completes.
4. Store all results in `st.session_state`.

**Results Section (Tabbed)**
After processing, show results using `st.tabs` with six tabs:

- **😊 Sentiment** — For each file show a card with the sentiment label, a colored indicator (green=positive, red=negative, yellow=mixed), and three horizontal `st.progress` bars for professionalism, empathy, and resolution scores.
- **🏷️ Topic Classification** — A table with filename, topic tag, snippet from transcript, and confidence. Also show a `st.bar_chart` of topic counts on the side.
- **🎭 Speaker Diarization** — For each file, display the word-level segments from the Whisper response grouped by inferred speaker turns (alternate speakers when there is a pause over 1 second between segments). Show speaker label, timestamp, and text for each turn.
- **🧑‍💼 Agent Performance** — A `st.dataframe` table with columns: File, Agent (use filename), Professionalism, Empathy, Resolution, Average Score, Outcome, GPT-4o Assessment. Color-code the average score column (green ≥ 85, yellow ≥ 70, red below 70).
- **💬 Key Phrases** — Display all key phrases from all files as a word cloud using the `wordcloud` library rendered via `st.image`. Also show a per-file breakdown of its three key phrases.
- **📄 Transcripts** — A `st.selectbox` to pick a file, then display the full transcript with speaker turns color-coded using `st.markdown` with inline HTML.

**Sidebar**
Include a sidebar with an `st.text_input` for the OpenAI API key (type=password), a `st.selectbox` to filter results by department/topic, and summary KPI metrics using `st.metric` for total files processed, resolved count, escalated count, and average overall score.

**Requirements**
Use `streamlit`, `openai`, `pandas`, `wordcloud`, and `matplotlib`. Store all processed results in `st.session_state["results"]` so they persist across reruns. Handle API errors gracefully with `st.error`. Show `st.spinner` during each file's processing. The API key should never be hardcoded — read it from the sidebar input or from `os.environ.get("OPENAI_API_KEY")` as fallback.

---

Paste this directly into Copilot Chat and it will scaffold the full `app.py`. You can follow up with *"add a requirements.txt"* and *"add a .env.example file"* to get the full project setup.