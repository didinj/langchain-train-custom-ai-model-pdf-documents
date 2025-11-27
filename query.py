from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


def load_vectorstore():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(
        "embeddings", embeddings, allow_dangerous_deserialization=True
    )
    return vectorstore


def get_retriever(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})  # return top 4 chunks
    return retriever


def get_llm():
    return ChatOpenAI(model="gpt-4o-mini")


def build_rag(retriever):
    llm = ChatOpenAI(model="gpt-4o-mini")

    prompt = ChatPromptTemplate.from_template(
        """
You are an AI assistant that answers questions based ONLY on the provided context.

<context>
{context}
</context>

Question: {question}
"""
    )

    # LCEL pipeline
    return (
        {"context": retriever, "question": "question"}
        | prompt
        | llm
        | StrOutputParser()
    )


def ask(question: str):
    vs = load_vectorstore()
    retriever = get_retriever(vs)
    rag = build_rag(retriever)
    return rag.invoke({"question": question})


if __name__ == "__main__":
    while True:
        q = input("\nAsk a question (or 'exit'): ")
        if q.lower() == "exit":
            break
        print("\nANSWER:\n", ask(q))
