import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="NLP Demo", page_icon="🤖", layout="centered")
st.title("🤖 NLP Demo App")
st.caption("Text Summarizer + Extractive QA — built with HuggingFace Transformers")

# Model Loading (cached so they don't reload on every interaction)
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

@st.cache_resource
def load_qa():
    return pipeline("question-answering", model="deepset/roberta-base-squad2")

summarizer = load_summarizer()
qa_pipeline = load_qa()

# Tabs
tab1, tab2 = st.tabs(["📝 Summarizer", "❓ Question Answering"])

# TAB 1: Summarizer
with tab1:
    st.subheader("Document Summarizer")
    st.markdown("Paste any long text — get a 2–3 sentence summary.")
    
    text_input = st.text_area("Your text:", height=280, placeholder="Paste article, doc, or any long text here...")
    
    if st.button("Summarize", key="summarize_btn"):
        if not text_input.strip():
            st.warning("Please paste some text first.")
        else:
            with st.spinner("Summarizing..."):
                result = summarizer(text_input, max_length=130, min_length=30, do_sample=False)
                st.success("Summary")
                st.write(result[0]["summary_text"])

# TAB 2: QA System
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
        key="qa_context"
    )
    question = st.text_input(
        "Your question:",
        placeholder="e.g. Who founded the company? What year did this happen?",
        key="qa_question"
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
            
            # Highlight where the answer came from
            with st.expander("Where in the context?"):
                start, end = result["start"], result["end"]
                highlighted = (
                    context[:start]
                    + f"**:blue[{context[start:end]}]**"
                    + context[end:]
                )
                st.markdown(highlighted)