# QA + Document Summarizer

Two production-grade NLP pipelines in one Streamlit app: paste any document and get a
3-sentence summary; ask any question over a passage and get the exact answer span extracted
with a confidence score.

**Live demo →** [Streamlit App](YOUR_STREAMLIT_LINK) | [60-sec Loom walkthrough](YOUR_LOOM_LINK)

---

## What this solves

SaaS companies drown in long documents — support tickets, changelogs, internal wikis,
onboarding PDFs. This app demonstrates two NLP workflows that matter for AI tooling:

- **Document summarization** — compress any long text to 3 precise sentences (BART-large-CNN)
- **Extractive QA** — ask a question in plain English, get the exact answer span pulled from
  a passage with a confidence score (RoBERTa-base-SQuAD2)

These are the inference primitives that sit inside a production Employee Handbook Chatbot or
internal knowledge base assistant — before adding a retrieval layer.

---

## Architecture decisions

**Why BART over T5 for summarization?**
`facebook/bart-large-cnn` is fine-tuned directly on CNN/DailyMail articles — it produces
cleaner output on document-style text than `t5-small` out of the box. T5's text-to-text
framing is more flexible across tasks, but BART needs no prompt engineering to generate
coherent summaries.

**Why extractive QA over generative?**
`deepset/roberta-base-squad2` does span extraction — it highlights the exact answer from
the provided context rather than generating text. This means no hallucination: if the answer
isn't in the passage, the model returns an empty string with a low confidence score.
Generative QA models can fabricate plausible-sounding answers. For a client's internal
knowledge base, extractive first is the safer production choice.

**Why not RAG here?**
RAG adds a retrieval layer on top of this QA system — a vector database stores chunked
documents, and retrieval finds the right passage before QA extracts the answer. This app
isolates the QA + summarization inference step: the hardest component to debug in a full
RAG pipeline. The retrieval layer is Phase 2.

---

## Stack

| Component | Model | Notes |
|---|---|---|
| Summarization | `facebook/bart-large-cnn` | `pipeline("summarization")` |
| Extractive QA | `deepset/roberta-base-squad2` | `pipeline("question-answering")` |
| UI | Streamlit (two tabs) | `@st.cache_resource` for model loading |
| Deployment | Streamlit Cloud | — |

---

## Run locally

```bash
git clone https://github.com/codewith-krishh/qa-summarizer-app
cd qa-summarizer-app
pip install -r requirements.txt
streamlit run app.py
```

---


Next: Prompt Engineering (OpenAI + Claude API, chain-of-thought, system prompts)