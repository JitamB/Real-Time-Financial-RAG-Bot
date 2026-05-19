"""Streamlit dashboard — talks to the Pathway backend over HTTP (RAGClient).

The whole point for the demo: upload or delete a document from the sidebar
and watch the chat answer change within seconds, with no backend restart.
"""

from __future__ import annotations

import os
import sys
import time

import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))
from components import (  # noqa: E402
    DEFAULT_BACKEND,
    add_document,
    ask,
    backend_stats,
    delete_document,
    get_client,
    local_doc_names,
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FinRAG · Real-Time",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject CSS
_css_path = os.path.join(os.path.dirname(__file__), "..", "styles.css")
if os.path.exists(_css_path):
    with open(_css_path) as fh:
        st.markdown(f"<style>{fh.read()}</style>", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "suggested" not in st.session_state:
    st.session_state.suggested = None

client = get_client(os.environ.get("RAG_BACKEND_URL", DEFAULT_BACKEND))

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand header
    st.markdown(
        """
        <div style="padding:0.25rem 0 1.25rem; border-bottom:1px solid rgba(255,255,255,0.07); margin-bottom:1.25rem;">
            <div style="font-size:1.15rem;font-weight:700;color:#F1F5F9;letter-spacing:-0.3px;">
                📈 FinRAG
            </div>
            <div style="font-size:0.72rem;color:#4B5C73;margin-top:3px;">
                Real-Time Incremental RAG · Pathway
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Live backend status (auto-refreshes every 5 s) ────────────────────
    st.markdown('<div class="section-label">Backend Status</div>', unsafe_allow_html=True)

    @st.fragment(run_every=5)
    def _status_panel() -> None:
        s = backend_stats(client)
        if not s:
            st.markdown(
                '<div class="error-banner">⚠ Backend unreachable</div>',
                unsafe_allow_html=True,
            )
            return

        st.markdown(
            f'<span class="live-dot"></span>'
            f'<span style="font-size:0.78rem;color:#94A3B8;">Live · auto-updates every 5 s</span>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        c1.metric("Indexed files", s.get("file_count", "—"))
        last = s.get("last_indexed") or s.get("last_modified") or "—"
        c2.metric("Last updated", last)
        st.markdown(
            f'<div class="backend-url">{DEFAULT_BACKEND}</div>',
            unsafe_allow_html=True,
        )

    _status_panel()
    st.divider()

    # ── Document injection (Channel A) ────────────────────────────────────
    st.markdown('<div class="section-label">Inject Document · Channel A</div>', unsafe_allow_html=True)
    st.caption("Upload to the watched folder — indexed live, no restart needed.")
    up = st.file_uploader(
        "drop_zone",
        type=["txt", "pdf", "md"],
        label_visibility="collapsed",
    )
    if up is not None:
        with st.spinner("Indexing…"):
            ok, msg = add_document(up.name, up.getvalue(), up.type == "application/pdf")
        if ok:
            st.toast(f"Indexed · {msg}", icon="✅")
        else:
            st.toast(f"Failed · {msg}", icon="⚠️")

    # ── Indexed documents list ─────────────────────────────────────────────
    st.markdown('<div class="section-label" style="margin-top:1.1rem;">Indexed Documents</div>', unsafe_allow_html=True)
    names = local_doc_names()
    if names:
        for name in names:
            ext = name.rsplit(".", 1)[-1].lower() if "." in name else "txt"
            icon = {"pdf": "📄", "md": "📝", "txt": "📃"}.get(ext, "📄")
            col_name, col_btn = st.columns([5, 1], vertical_alignment="center")
            col_name.markdown(
                f'<div class="file-item"><span class="file-icon">{icon}</span>{name}</div>',
                unsafe_allow_html=True,
            )
            if col_btn.button("✕", key=f"del_{name}", help=f"Delete {name}"):
                with st.spinner(f"Removing {name}…"):
                    ok, msg = delete_document(name)
                if ok:
                    st.toast(f"Deleted · {msg}", icon="🗑️")
                else:
                    st.toast(f"Error · {msg}", icon="⚠️")
                st.rerun()
    else:
        st.markdown(
            '<div style="font-size:0.78rem;color:#4B5C73;padding:6px 2px;">No documents in data/docs/</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── New chat ───────────────────────────────────────────────────────────
    if st.button("＋  New conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.suggested = None
        st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
# Header
st.markdown(
    """
    <div style="padding:0.5rem 0 1.5rem;border-bottom:1px solid rgba(255,255,255,0.06);margin-bottom:1.5rem;">
        <div style="display:flex;align-items:baseline;gap:10px;">
            <h1 style="font-size:1.65rem;font-weight:700;color:#EEF2F7;margin:0;letter-spacing:-0.5px;">
                Real-Time Financial RAG
            </h1>
            <span style="font-size:0.72rem;font-weight:600;background:rgba(22,199,132,0.12);
                         color:#16C784;border:1px solid rgba(22,199,132,0.3);
                         border-radius:99px;padding:2px 10px;letter-spacing:0.04em;">LIVE</span>
        </div>
        <p style="font-size:0.83rem;color:#4B5C73;margin:6px 0 0;">
            Powered by Pathway differential dataflow · add / edit / delete a document and re-ask — the answer follows in seconds.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Welcome screen ────────────────────────────────────────────────────────────
EXAMPLE_PROMPTS = [
    "What is Acme Robotics' Q1 revenue?",
    "What does 'P/E ratio' mean?",
    "Latest AAPL stock price?",
    "Summarise the uploaded documents.",
    "What is the operating margin for Acme?",
]

if not st.session_state.messages:
    st.markdown(
        """
        <div class="welcome-card">
            <h2>What would you like to know?</h2>
            <p>Ask about uploaded documents, live market data, or financial concepts.<br>
               The index updates in real time as you add or delete files.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    # Render example prompts as clickable pills via columns
    cols = st.columns(len(EXAMPLE_PROMPTS))
    for i, ep in enumerate(EXAMPLE_PROMPTS):
        if cols[i].button(ep, key=f"ep_{i}", use_container_width=True):
            st.session_state.suggested = ep
            st.rerun()

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            _sources = msg["sources"]
            _valid = [d for d in _sources if isinstance(d, dict) and d.get("text", "").strip()]
            if _valid:
                with st.expander(f"📎 {len(_valid)} source{'s' if len(_valid) != 1 else ''} retrieved", expanded=False):
                    for idx, doc in enumerate(_valid, 1):
                        meta = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}
                        path = meta.get("path", "unknown")
                        text = doc.get("text", "")[:350]
                        st.markdown(
                            f"""
                            <div class="source-card">
                                <div class="source-card-path">#{idx} · {os.path.basename(path)}</div>
                                <div class="source-card-text">{text}{"…" if len(doc.get("text","")) > 350 else ""}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

# ── Handle suggested prompt (from welcome pills) ──────────────────────────────
if st.session_state.suggested:
    _prompt = st.session_state.suggested
    st.session_state.suggested = None
    st.session_state.messages.append({"role": "user", "content": _prompt})
    with st.chat_message("user"):
        st.markdown(_prompt)
    with st.chat_message("assistant"):
        with st.spinner("Querying live index…"):
            _t0 = time.monotonic()
            _res = ask(client, _prompt)
            _latency = time.monotonic() - _t0
        _answer = _res.get("response", "(no response)")
        _docs = _res.get("context_docs", []) or []
        st.markdown(_answer)
        _valid_docs = [d for d in _docs if isinstance(d, dict) and d.get("text", "").strip()]
        if _valid_docs:
            with st.expander(f"📎 {len(_valid_docs)} source{'s' if len(_valid_docs) != 1 else ''} retrieved", expanded=False):
                for idx, doc in enumerate(_valid_docs, 1):
                    meta = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}
                    path = meta.get("path", "unknown")
                    text = doc.get("text", "")[:350]
                    st.markdown(
                        f"""
                        <div class="source-card">
                            <div class="source-card-path">#{idx} · {os.path.basename(path)}</div>
                            <div class="source-card-text">{text}{"…" if len(doc.get("text","")) > 350 else ""}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        st.caption(f"⚡ {_latency:.2f} s")
    st.session_state.messages.append(
        {"role": "assistant", "content": _answer, "sources": _docs}
    )

# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about the market, news, or your documents…"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        with st.spinner("Querying live index…"):
            t0 = time.monotonic()
            res = ask(client, prompt)
            latency = time.monotonic() - t0
        answer = res.get("response", "(no response)")
        docs = res.get("context_docs", []) or []
        st.markdown(answer)
        valid_docs = [d for d in docs if isinstance(d, dict) and d.get("text", "").strip()]
        if valid_docs:
            with st.expander(f"📎 {len(valid_docs)} source{'s' if len(valid_docs) != 1 else ''} retrieved", expanded=False):
                for idx, doc in enumerate(valid_docs, 1):
                    meta = doc.get("metadata", {}) if isinstance(doc.get("metadata"), dict) else {}
                    path = meta.get("path", "unknown")
                    text = doc.get("text", "")[:350]
                    st.markdown(
                        f"""
                        <div class="source-card">
                            <div class="source-card-path">#{idx} · {os.path.basename(path)}</div>
                            <div class="source-card-text">{text}{"…" if len(doc.get("text","")) > 350 else ""}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        st.caption(f"⚡ {latency:.2f} s")
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": docs}
    )
