import streamlit as st
from rag_pipeline.retriever import Retriever
from rag_pipeline.generator import Generator
import pandas as pd
import re
import base64

st.set_page_config(page_title="🧠 GenAI Product Intelligence", layout="wide")
st.title("🧠 GenAI Product Q&A Assistant")

retriever = Retriever()
generator = Generator()

st.markdown("### 💬 Ask a Question")
query = st.text_area("Enter your product-related question:")

if st.button("🔍 Analyze"):
    with st.spinner("🔎 Retrieving relevant documents..."):
        retrieved_chunks = retriever.retrieve(query)

        # ✅ Show all retrieved chunks inside one expander
        st.markdown("### 📚 Retrieved Context")
        with st.expander("📄 Click to expand all retrieved documents", expanded=False):
            for i, chunk in enumerate(retrieved_chunks[:5]):
                st.markdown(f"**Document {i+1}:**")
                st.markdown(chunk['text'])
                st.markdown("---")

    with st.spinner("🤖 Generating answer..."):
        answer = generator.generate(query, retrieved_chunks)

    # ✅ Display answer
    st.markdown("### 💡 Generated Answer")

    def extract_table_data(answer):
        # Heuristically parse answer into key-value table
        rows = []
        for line in answer.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                rows.append((key.strip(), value.strip()))
        df = pd.DataFrame(rows, columns=["Field", "Value"])
        return df if not df.empty else None

    try:
        df = extract_table_data(answer)
        if df is not None and len(df.columns) == 2 and len(df) >= 2:
            with st.expander("🧾 View as Table", expanded=True):
                st.table(df)
        else:
            st.markdown(answer)
    except Exception:
        st.markdown(answer)

    # ✅ Download answer
    b64 = base64.b64encode(answer.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="answer.txt">📥 Download Answer as .txt</a>'
    st.markdown(href, unsafe_allow_html=True)
