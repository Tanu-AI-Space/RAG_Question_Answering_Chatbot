"""
Streamlit UI for the Research Paper RAG pipeline.
Lets users upload PDFs, builds/loads the FAISS index, and provides
a chat-style interface for citation-aware Q&A over the documents.

Design: "Research Terminal" — a dark, glassy command-deck aesthetic.
Gradient glow accents, a connected step-timeline in the sidebar, and
each Q&A turn rendered as a log entry with syntax-style Q/A markers.

Run with:
    export MISTRAL_API_KEY="your-key"
    streamlit run app.py
"""

import os
import sys
import time
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from ingest import ingest_directory                              # Step 1
from embed_index import build_and_save_index, load_index, get_embedder, EMBED_MODEL_NAME, INDEX_DIR  # Step 2
from query import query_rag                                       # Step 3
from mistralai import Mistral
from dotenv import load_dotenv

load_dotenv()
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
api_key = os.getenv("MISTRAL_API_KEY")

# ----------------------------- Page config -----------------------------

st.set_page_config(
    page_title="Research Terminal · Mistral RAG",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- Design system (CSS) -----------------------------

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    :root {
        --bg: #08090D;
        --bg-2: #0D0F16;
        --surface: rgba(255,255,255,0.045);
        --surface-hi: rgba(255,255,255,0.075);
        --border: rgba(255,255,255,0.09);
        --border-hi: rgba(255,255,255,0.16);
        --text: #ECECF4;
        --text-soft: #8D8AA0;
        --text-faint: #5C5A6E;
        --violet: #8B5CF6;
        --cyan: #22D3EE;
        --amber: #FBBF24;
        --good: #34D399;
        --grad: linear-gradient(90deg, var(--violet), var(--cyan));
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: var(--text); }

    .stApp {
        background:
            radial-gradient(circle at 12% -5%, rgba(139,92,246,0.16), transparent 42%),
            radial-gradient(circle at 100% 10%, rgba(34,211,238,0.10), transparent 38%),
            radial-gradient(circle at 50% 100%, rgba(139,92,246,0.06), transparent 45%),
            var(--bg);
    }

    [data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 2rem; }

    h1, h2, h3, .app-title, .section-label { font-family: 'Space Grotesk', sans-serif; }

    ::-webkit-scrollbar { width: 10px; height: 10px; }
    ::-webkit-scrollbar-track { background: var(--bg-2); }
    ::-webkit-scrollbar-thumb { background: var(--border-hi); border-radius: 6px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--violet); }

    /* ---------- Header ---------- */
    .app-header { padding: 0.2rem 0 1.4rem 0; margin-bottom: 1.6rem; border-bottom: 1px solid var(--border); }
    .eyebrow {
        font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; letter-spacing: 0.18em;
        text-transform: uppercase; color: var(--cyan); font-weight: 600;
        display: inline-flex; align-items: center; gap: 0.5rem;
    }
    .eyebrow .dot {
        width: 7px; height: 7px; border-radius: 50%; background: var(--good); display: inline-block;
        box-shadow: 0 0 0 0 rgba(52,211,153,0.6); animation: pulse-dot 1.8s infinite;
    }
    @keyframes pulse-dot {
        0% { box-shadow: 0 0 0 0 rgba(52,211,153,0.55); }
        70% { box-shadow: 0 0 0 7px rgba(52,211,153,0); }
        100% { box-shadow: 0 0 0 0 rgba(52,211,153,0); }
    }
    .app-title {
        font-size: 2.5rem; font-weight: 700; margin: 0.35rem 0 0.4rem 0;
        letter-spacing: -0.01em; line-height: 1.1;
        background: var(--grad); -webkit-background-clip: text; background-clip: text; color: transparent;
        background-size: 200% auto; animation: shine 6s ease-in-out infinite;
    }
    @keyframes shine {
        0%, 100% { background-position: 0% center; }
        50% { background-position: 100% center; }
    }
    .app-subtitle { font-size: 0.98rem; color: var(--text-soft); max-width: 50ch; }

    /* ---------- Stat / status cards ---------- */
    .catalog-card {
        background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
        padding: 1.05rem 1.2rem; backdrop-filter: blur(10px);
        transition: transform 0.18s ease, border-color 0.18s ease, background 0.18s ease;
        position: relative; overflow: hidden;
    }
    .catalog-card::before {
        content: ""; position: absolute; inset: 0; border-radius: 12px; padding: 1px;
        background: var(--grad); opacity: 0; transition: opacity 0.2s ease;
        -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
        -webkit-mask-composite: xor; mask-composite: exclude; pointer-events: none;
    }
    .catalog-card:hover { transform: translateY(-3px); background: var(--surface-hi); border-color: transparent; }
    .catalog-card:hover::before { opacity: 1; }
    .catalog-label {
        font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; letter-spacing: 0.1em;
        text-transform: uppercase; color: var(--text-soft); margin-bottom: 0.35rem;
    }
    .catalog-value { font-family: 'Space Grotesk', sans-serif; font-size: 1.9rem; font-weight: 700; color: var(--text); line-height: 1; }
    .catalog-value.ready { color: var(--good); }
    .catalog-value.pending { color: var(--amber); }
    .status-line { display: inline-flex; align-items: center; gap: 0.45rem; }
    .status-line .led { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
    .led.ready { background: var(--good); box-shadow: 0 0 8px var(--good); animation: pulse-dot 1.8s infinite; }
    .led.pending { background: var(--amber); box-shadow: 0 0 8px var(--amber); }

    /* ---------- Sidebar: glass command deck ---------- */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0B0C13 0%, #101223 100%);
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] * { color: var(--text) !important; }
    section[data-testid="stSidebar"] .sidebar-kicker {
        font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; letter-spacing: 0.16em;
        text-transform: uppercase; color: var(--cyan) !important;
    }
    section[data-testid="stSidebar"] .sidebar-title {
        font-family: 'Space Grotesk', sans-serif; font-size: 1.4rem; font-weight: 700; margin: 0.15rem 0 0.5rem 0;
    }
    section[data-testid="stSidebar"] .sidebar-desc { font-size: 0.85rem; color: var(--text-soft) !important; line-height: 1.55; }

    .timeline-step { position: relative; padding-left: 1.6rem; margin: 1.3rem 0 0.6rem 0; }
    .timeline-step::before {
        content: ""; position: absolute; left: 0; top: 3px; width: 10px; height: 10px; border-radius: 50%;
        background: var(--grad); box-shadow: 0 0 10px rgba(139,92,246,0.7);
    }
    .timeline-step::after {
        content: ""; position: absolute; left: 4px; top: 15px; width: 1px; height: 46px;
        background: linear-gradient(180deg, var(--violet), transparent);
    }
    .timeline-step .step-tag {
        font-family: 'JetBrains Mono', monospace; font-size: 0.68rem; letter-spacing: 0.1em;
        color: var(--cyan) !important; text-transform: uppercase; display: block; margin-bottom: 0.1rem;
    }
    .timeline-step .step-title { font-weight: 600; font-size: 0.95rem; }

    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background: var(--surface); border: 1.5px dashed var(--border-hi); border-radius: 10px;
        transition: border-color 0.15s ease;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]:hover { border-color: var(--cyan); }
    section[data-testid="stSidebar"] hr { border-color: var(--border); }

    /* Slider accent */
    [data-testid="stSlider"] [role="slider"] { background: var(--cyan) !important; }
    section[data-testid="stSidebar"] [data-baseweb="slider"] div div div { background: var(--grad) !important; }

    /* ---------- Buttons ---------- */
    .stButton>button {
        border-radius: 9px; font-weight: 600; background: var(--grad); color: #0A0A0F;
        border: none; padding: 0.6rem 1.2rem; transition: all 0.18s ease; letter-spacing: 0.01em;
        box-shadow: 0 0 0 rgba(139,92,246,0);
    }
    .stButton>button:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(139,92,246,0.35); filter: brightness(1.08); }
    .stButton>button:active { transform: translateY(0); }
    section[data-testid="stSidebar"] .stButton>button {
        background: var(--surface); color: var(--text) !important; border: 1px solid var(--border-hi);
        width: 100%;
    }
    section[data-testid="stSidebar"] .stButton>button:hover {
        background: var(--surface-hi); border-color: var(--cyan); box-shadow: 0 0 16px rgba(34,211,238,0.15);
    }

    /* ---------- Log entries (chat) ---------- */
    .field-note {
        background: var(--surface); border: 1px solid var(--border); border-radius: 14px;
        margin-bottom: 1.15rem; overflow: hidden; backdrop-filter: blur(10px);
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
        animation: rise-in 0.35s ease-out;
    }
    .field-note:hover { border-color: var(--border-hi); box-shadow: 0 8px 28px rgba(0,0,0,0.35); }
    @keyframes rise-in { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
    .field-note-head {
        display: flex; align-items: center; justify-content: space-between;
        padding: 0.55rem 1.15rem; background: rgba(255,255,255,0.02); border-bottom: 1px solid var(--border);
        font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; color: var(--text-faint); letter-spacing: 0.05em;
    }
    .field-note-head .entry-no { color: var(--cyan); font-weight: 700; }
    .field-note-head .tag-live {
        color: var(--amber); display: inline-flex; align-items: center; gap: 0.35rem;
    }
    .field-note-head .tag-live::before {
        content: ""; width: 6px; height: 6px; border-radius: 50%; background: var(--amber);
        animation: pulse-dot 1.2s infinite;
    }
    .question-box {
        font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1.28rem; color: var(--text);
        padding: 1.05rem 1.25rem 0.55rem 1.25rem; line-height: 1.45;
    }
    .question-box .q-mark {
        background: var(--grad); -webkit-background-clip: text; background-clip: text; color: transparent;
        font-weight: 700; margin-right: 0.25rem;
    }
    .answer-box {
        padding: 0 1.25rem 1.2rem 1.25rem; line-height: 1.7; color: var(--text-soft); font-size: 0.98rem;
        border-left: 2px solid transparent; margin-left: 1.25rem; padding-left: 0.95rem;
    }
    .sources-box {
        background: rgba(255,255,255,0.03); border-radius: 8px; padding: 0.85rem 1rem;
        font-size: 0.82rem; color: #B9E4EF; border-left: 3px solid var(--cyan);
        font-family: 'JetBrains Mono', monospace; line-height: 1.75;
    }

    /* Expander (sources) styling */
    [data-testid="stExpander"] {
        background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 10px !important;
    }
    .streamlit-expanderHeader, [data-testid="stExpander"] summary {
        font-family: 'JetBrains Mono', monospace !important; font-size: 0.78rem !important;
        letter-spacing: 0.03em; color: var(--cyan) !important;
    }

    /* Empty state */
    .empty-state {
        border: 1.5px dashed var(--border-hi); border-radius: 16px; padding: 3.2rem 2rem; text-align: center;
        background: var(--surface); backdrop-filter: blur(10px); position: relative;
    }
    .empty-state .emoji { font-size: 2.4rem; margin-bottom: 0.7rem; filter: drop-shadow(0 0 12px rgba(139,92,246,0.5)); }
    .empty-state .headline {
        font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.3rem; margin-bottom: 0.4rem; color: var(--text);
    }
    .empty-state .sub { color: var(--text-soft); font-size: 0.94rem; max-width: 42ch; margin: 0 auto; }

    /* chat input */
    [data-testid="stChatInput"] {
        background: var(--surface) !important; border: 1px solid var(--border-hi) !important;
        border-radius: 12px !important; backdrop-filter: blur(10px);
    }
    [data-testid="stChatInput"]:focus-within { border-color: var(--cyan) !important; box-shadow: 0 0 0 3px rgba(34,211,238,0.15); }
    [data-testid="stChatInput"] textarea { font-family: 'Inter', sans-serif !important; color: var(--text) !important; }

    [data-testid="stAlert"] { background: var(--surface) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; }
</style>
""", unsafe_allow_html=True)

# ----------------------------- Session state -----------------------------

if "index" not in st.session_state:
    st.session_state.index = None
if "chunks" not in st.session_state:
    st.session_state.chunks = None
if "embedder" not in st.session_state:
    st.session_state.embedder = None
if "history" not in st.session_state:
    st.session_state.history = []  # list of dicts: {question, answer, sources}

# ----------------------------- Sidebar -----------------------------

with st.sidebar:
    st.markdown('<div class="sidebar-kicker">◈ Command Deck</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">Research Terminal</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-desc">Semantic Q&amp;A over your papers — powered by '
        '<b style="color:#ECECF4">Mistral</b>, <b style="color:#ECECF4">FAISS</b> &amp; '
        '<b style="color:#ECECF4">Sentence Transformers</b>.</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="timeline-step"><span class="step-tag">Step 01</span>'
        '<span class="step-title">Upload PDFs</span></div>',
        unsafe_allow_html=True,
    )
    uploaded_files = st.file_uploader(
        "Drop research papers here", type=["pdf"], accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        os.makedirs(DATA_DIR, exist_ok=True)
        for f in uploaded_files:
            with open(os.path.join(DATA_DIR, f.name), "wb") as out:
                out.write(f.getbuffer())
        st.success(f"✓ {len(uploaded_files)} file(s) saved")

    st.markdown(
        '<div class="timeline-step"><span class="step-tag">Step 02</span>'
        '<span class="step-title">Build the index</span></div>',
        unsafe_allow_html=True,
    )
    k = st.slider("Top-k passages to retrieve", min_value=2, max_value=10, value=4)

    build_clicked = st.button("⚡ Build / Rebuild Index", use_container_width=True)
    load_clicked = st.button("↺ Load Existing Index", use_container_width=True)

    st.divider()
    if st.session_state.chunks:
        st.markdown(f"**Indexed chunks:** {len(st.session_state.chunks)}")
    if st.session_state.history:
        st.markdown(
            '<div class="timeline-step" style="margin-top:1rem;"><span class="step-tag">Archive</span>'
            '<span class="step-title">Conversation log</span></div>',
            unsafe_allow_html=True,
        )
        if st.button("🗑 Clear conversation", use_container_width=True):
            st.session_state.history = []
            st.rerun()

# ----------------------------- Index building -----------------------------

def build_index_with_progress():
    if not os.path.exists(DATA_DIR) or not any(f.endswith(".pdf") for f in os.listdir(DATA_DIR)):
        st.error("No PDFs found. Upload at least one PDF first.")
        return

    progress = st.progress(0, text="Extracting & chunking PDFs...")
    chunks = ingest_directory(DATA_DIR)
    if not chunks:
        st.error("No text could be extracted from the uploaded PDFs.")
        return

    progress.progress(40, text=f"Embedding {len(chunks)} chunks...")
    embedder = get_embedder(EMBED_MODEL_NAME)
    from embed_index import embed_chunks, build_faiss_index, save_index
    embeddings = embed_chunks(chunks, embedder)

    progress.progress(80, text="Building FAISS index...")
    index = build_faiss_index(embeddings)
    save_index(index, chunks)

    progress.progress(100, text="Done!")
    time.sleep(0.4)
    progress.empty()

    st.session_state.index = index
    st.session_state.chunks = chunks
    st.session_state.embedder = embedder
    st.success(f"Indexed {len(chunks)} chunks from your PDFs.")


def load_existing_index():
    index_path = os.path.join(INDEX_DIR, "faiss.index")
    if not os.path.exists(index_path):
        st.error("No saved index found. Build one first.")
        return
    index, chunks = load_index()
    embedder = get_embedder(EMBED_MODEL_NAME)
    st.session_state.index = index
    st.session_state.chunks = chunks
    st.session_state.embedder = embedder
    st.success(f"Loaded index with {len(chunks)} chunks.")


if build_clicked:
    build_index_with_progress()
if load_clicked:
    load_existing_index()

# ----------------------------- Main area -----------------------------

st.markdown(
    """
    <div class="app-header">
        <div class="eyebrow"><span class="dot"></span>System Online · Research Terminal</div>
        <p class="app-title">Research Paper RAG</p>
        <p class="app-subtitle">Ask grounded, citation-aware questions across your uploaded research papers.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

col1, col2, col3 = st.columns(3)
num_pdfs = len([f for f in os.listdir(DATA_DIR) if f.endswith(".pdf")]) if os.path.exists(DATA_DIR) else 0
num_chunks = len(st.session_state.chunks) if st.session_state.chunks else 0
index_ready = st.session_state.index is not None
status_label = "Ready" if index_ready else "Not built"
status_class = "ready" if index_ready else "pending"
led_class = "ready" if index_ready else "pending"

with col1:
    st.markdown(
        f'<div class="catalog-card"><div class="catalog-label">PDFs Uploaded</div>'
        f'<div class="catalog-value">{num_pdfs}</div></div>',
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f'<div class="catalog-card"><div class="catalog-label">Indexed Chunks</div>'
        f'<div class="catalog-value">{num_chunks}</div></div>',
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f'<div class="catalog-card"><div class="catalog-label">Index Status</div>'
        f'<div class="catalog-value {status_class} status-line"><span class="led {led_class}"></span>{status_label}</div></div>',
        unsafe_allow_html=True,
    )

st.write("")

if not st.session_state.index:
    st.markdown(
        """
        <div class="empty-state">
            <div class="emoji">◈</div>
            <div class="headline">Terminal is idle</div>
            <div class="sub">Upload PDFs and click <b>Build / Rebuild Index</b> in the sidebar to power up the system.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    query = st.chat_input("Ask a question about your papers...")

    if query:
        if not api_key:
            st.error("MISTRAL_API_KEY not found.")
        else:
            entry_no = len(st.session_state.history) + 1

            st.markdown(
                f'<div class="field-note">'
                f'<div class="field-note-head"><span class="entry-no">ENTRY {entry_no:03d}</span>'
                f'<span class="tag-live">processing</span></div>'
                f'<div class="question-box"><span class="q-mark">Q.</span>{query}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            answer_placeholder = st.empty()
            with answer_placeholder:
                with st.spinner("Thinking..."):
                    client = Mistral(api_key=api_key)
                    result = query_rag(
                        query,
                        st.session_state.index,
                        st.session_state.chunks,
                        st.session_state.embedder,
                        client,
                        k=k,
                    )
            answer_placeholder.empty()

            st.markdown(f'<div class="answer-box">{result["answer"]}</div>', unsafe_allow_html=True)

            with st.expander("📖 Sources"):
                st.markdown(
                    f'<div class="sources-box">{result["sources"].replace(chr(10), "<br>")}</div>',
                    unsafe_allow_html=True,
                )

            st.session_state.history.append({
                "question": query,
                "answer": result["answer"],
                "sources": result["sources"],
            })
            st.rerun()

    # Render conversation history, most recent first
    for i, turn in enumerate(reversed(st.session_state.history)):
        entry_no = len(st.session_state.history) - i
        question = turn["question"]
        st.markdown(
            f'<div class="field-note">'
            f'<div class="field-note-head"><span class="entry-no">ENTRY {entry_no:03d}</span><span>archived</span></div>'
            f'<div class="question-box"><span class="q-mark">Q.</span>{question}</div>'
            f'<div class="answer-box">{turn["answer"]}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        with st.expander("📖 Sources"):
            st.markdown(
                f'<div class="sources-box">{turn["sources"].replace(chr(10), "<br>")}</div>',
                unsafe_allow_html=True,
            )