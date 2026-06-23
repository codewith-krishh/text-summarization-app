import streamlit as st
from transformers import pipeline, AutoTokenizer

st.set_page_config(page_title="NLP Demo", page_icon="🤖", layout="centered")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 NLP Demo App")
    st.markdown("Built with HuggingFace Transformers")
    st.divider()
    st.markdown("**Models in use:**")
    st.markdown(
        "📝 **Summarizer** \n`sshleifer/distilbart-cnn-12-6` \n"
        "DistilBART — encoder-decoder, trained on CNN/DailyMail news articles."
    )
    st.markdown(
        "❓ **QA System** \n`deepset/roberta-base-squad2` \n"
        "RoBERTa — encoder-only, fine-tuned on SQuAD 2.0 for extractive QA."
    )
    st.divider()
    st.caption("Demo by Krish · Phase 1 Portfolio · Week 22")

# ── Page Header ───────────────────────────────────────────────────────────────
st.title("🤖 NLP Demo App")
st.caption("Text Summarizer + Extractive QA — built with HuggingFace Transformers")

# ── Model Loading (cached so they don't reload on every interaction) ──────────
SUMMARIZER_MODEL = "sshleifer/distilbart-cnn-12-6"
QA_MODEL = "deepset/roberta-base-squad2"
SUMMARIZER_MAX_TOKENS = 1024   # BART's hard input limit

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model=SUMMARIZER_MODEL)

@st.cache_resource
def load_summarizer_tokenizer():
    return AutoTokenizer.from_pretrained(SUMMARIZER_MODEL)

@st.cache_resource
def load_qa():
    return pipeline("question-answering", model=QA_MODEL)

summarizer         = load_summarizer()
summarizer_tok     = load_summarizer_tokenizer()
qa_pipeline        = load_qa()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["📝 Summarizer", "❓ Question Answering"])

# ── TAB 1: Summarizer ─────────────────────────────────────────────────────────
with tab1:
    st.subheader("Document Summarizer")
    st.markdown("Paste any long text — get a 2–3 sentence summary.")

    text_input = st.text_area(
        "Your text:",
        height=280,
        placeholder="Paste article, doc, or any long text here...",
    )

    # Word count + token count info — shows before the button is clicked
    if text_input.strip():
        word_count  = len(text_input.split())
        token_ids   = summarizer_tok.encode(text_input, truncation=False)
        token_count = len(token_ids)
        col_a, col_b = st.columns(2)
        col_a.caption(f"Words: {word_count}")
        col_b.caption(
            f"Tokens: {token_count}"
            + (f" ⚠️ Trimmed to {SUMMARIZER_MAX_TOKENS}" if token_count > SUMMARIZER_MAX_TOKENS else "")
        )

    if st.button("Summarize", key="summarize_btn"):
        if not text_input.strip():
            st.warning("Please paste some text first.")
        else:
            # Truncate at the tokenizer level — prevents HuggingFace runtime errors
            # on long inputs. decode() converts the safe token list back to a string
            # that the pipeline can accept normally.
            token_ids = summarizer_tok.encode(text_input, truncation=False)
            safe_ids  = token_ids[:SUMMARIZER_MAX_TOKENS]
            safe_text = summarizer_tok.decode(safe_ids, skip_special_tokens=True)

            with st.spinner("Summarizing..."):
                result = summarizer(safe_text, max_length=130, min_length=30, do_sample=False)
            st.success("Summary")
            st.write(result[0]["summary_text"])

# ── TAB 2: QA System ──────────────────────────────────────────────────────────
with tab2:
    st.subheader("Extractive Question Answering")
    st.markdown(
        "Paste a paragraph as **context**, then ask a question. "
        "The model finds the answer *within your text* — it doesn't generate new words."
    )

    context = st.text_area(
        "Context paragraph:",
        height=220,
        placeholder="Paste any paragraph of text here...",
        key="qa_context",
    )
    question = st.text_input(
        "Your question:",
        placeholder="e.g. Who founded the company? What year did this happen?",
        key="qa_question",
    )

    if st.button("Find Answer", key="qa_btn"):
        if not context.strip() or not question.strip():
            st.warning("Please provide both a context paragraph and a question.")
        else:
            with st.spinner("Scanning context for answer..."):
                result = qa_pipeline(question=question, context=context)

            st.success("Answer found")
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**Answer:** {result['answer']}")
            with col2:
                st.metric("Confidence", f"{result['score']:.1%}")

            with st.expander("Where in the context?"):
                start, end = result["start"], result["end"]
                highlighted = (
                    context[:start]
                    + f"**:blue[{context[start:end]}]**"
                    + context[end:]
                )
                st.markdown(highlighted)