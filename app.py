import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


# Load vector store
def load_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return FAISS.load_local(
        "embeddings", embeddings, allow_dangerous_deserialization=True
    )


# Build RAG pipeline
def build_rag():
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatOpenAI(model="gpt-4o-mini")

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful assistant. Answer the user's question using ONLY the context below.
If the answer is not in the context, say you cannot find it.

<context>
{context}
</context>

Question: {question}
"""
    )

    rag_chain = (
        {"context": retriever, "question": "question"}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


rag_chain = build_rag()

# --- Streamlit UI ---

st.set_page_config(page_title="PDF AI Chatbot", page_icon="📄")

st.title("📄 PDF AI Chatbot")
st.write("Ask questions based on your custom PDF-trained model.")

# Initialize session history
if "history" not in st.session_state:
    st.session_state.history = []

question = st.text_input("Enter your question:")

if st.button("Ask") and question.strip():
    answer = rag_chain.invoke({"question": question})

    st.session_state.history.append((question, answer))

# Chat history UI
for q, a in st.session_state.history:
    st.markdown(f"**🧑‍💻 You:** {q}")
    st.markdown(f"**🤖 AI:** {a}")
    st.markdown("---")
