"""
Model: sshleifer/distilbart-cnn-12-6
"""

import streamlit as st
from transformers import pipeline

# ─── Page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Document Summarizer",
    page_icon="📄",
    layout="centered"
)

# ─── Load model (cached — runs once per session) ─────────────────────────────
@st.cache_resource
def load_model():
    """
    @st.cache_resource: Streamlit's way of caching heavy objects like models.
    Without this, the model re-loads on every interaction — takes 30+ seconds.
    With this, it loads once and stays in memory for the session.
    """
    return pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6"
    )

# ─── Header ──────────────────────────────────────────────────────────────────
st.title("📄 Document Summarizer")
st.caption(
    "Paste any long text and get a concise 2–4 sentence summary. "
    "Powered by DistilBART (CNN/DailyMail fine-tuned)."
)

st.divider()

# ─── Input ───────────────────────────────────────────────────────────────────
st.subheader("Input")
text_input = st.text_area(
    label="Paste your document, article, or support ticket below:",
    height=280,
    placeholder=(
        "e.g. A customer support ticket, product documentation, "
        "email thread, or news article..."
    )
)

# Settings expander (advanced users can tweak, beginners ignore it)
with st.expander("⚙️ Settings"):
    max_len = st.slider("Max summary length (tokens)", min_value=50, max_value=200, value=130, step=10)
    min_len = st.slider("Min summary length (tokens)", min_value=20, max_value=80, value=40, step=5)

# ─── Summarize button ────────────────────────────────────────────────────────
summarize_btn = st.button("Summarize →", type="primary", use_container_width=True)

if summarize_btn:
    # Input validation
    word_count = len(text_input.split())

    if not text_input.strip():
        st.warning("Please paste some text above.")

    elif word_count < 50:
        st.warning(
            f"Your text is only {word_count} words. "
            "For a meaningful summary, paste at least 50 words."
        )

    else:
        # Run inference
        with st.spinner("Summarizing..."):
            summarizer = load_model()
            result = summarizer(
                text_input,
                max_length=max_len,
                min_length=min_len,
                do_sample=False,
                truncation=True     # handle inputs that exceed model's max token length (1024)
            )
        summary_text = result[0]["summary_text"]

        # ─── Output ──────────────────────────────────────────────────────────
        st.divider()
        st.subheader("Summary")
        st.success(summary_text)

        # Compression stats
        summary_words = len(summary_text.split())
        reduction = round((1 - summary_words / word_count) * 100)

        col1, col2, col3 = st.columns(3)
        col1.metric("Original", f"{word_count} words")
        col2.metric("Summary", f"{summary_words} words")
        col3.metric("Reduced by", f"{reduction}%")

# ─── Footer ──────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built with 🤗 DistilBART (sshleifer/distilbart-cnn-12-6) + Streamlit · "
    "[GitHub](https://github.com/codewith-krishh)"
)