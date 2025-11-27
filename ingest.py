import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS

DATA_PATH = "data"

load_dotenv()


def clean_text(text: str) -> str:
    lines = text.split("\n")
    cleaned = [line.strip() for line in lines if line.strip()]
    return " ".join(cleaned)


def load_pdfs():
    documents = []
    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(DATA_PATH, file))
            docs = loader.load()
            for d in docs:
                d.page_content = clean_text(d.page_content)
            documents.extend(docs)
    return documents


if __name__ == "__main__":
    docs = load_pdfs()
    print(f"Loaded and cleaned {len(docs)} pages.")


def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150, separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    return chunks


if __name__ == "__main__":
    docs = load_pdfs()
    chunks = chunk_documents(docs)
    print(f"Loaded {len(docs)} pages.")
    print(f"Created {len(chunks)} chunks.")


def create_embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small")


def store_embeddings(chunks, embeddings):
    vectorstore = FAISS.from_documents(chunks, embeddings)
    vectorstore.save_local("embeddings")
    print("Vector store saved to /embeddings directory")


if __name__ == "__main__":
    # 1. Load
    docs = load_pdfs()
    print(f"Loaded {len(docs)} pages.")

    # 2. Chunk
    chunks = chunk_documents(docs)
    print(f"Created {len(chunks)} chunks.")

    # 3. Embeddings
    embeddings = create_embeddings()

    # 4. Store
    store_embeddings(chunks, embeddings)

    print("Ingestion complete.")
