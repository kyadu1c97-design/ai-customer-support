from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import shutil
import time

# Load embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def safe_delete_folder(folder_path):
    """Safely delete folder (handles Windows permission issues)"""
    if os.path.exists(folder_path):
        for i in range(5):  # 🔥 retry 5 times
            try:
                shutil.rmtree(folder_path)
                print(f"🗑️ Deleted old {folder_path}")
                return
            except Exception as e:
                print(f"⚠️ Retry {i+1}: {e}")
                time.sleep(2)

        print(f"❌ Could not delete {folder_path}. Please delete manually.")


def create_vector_store():
    print("📄 Loading PDF...")

    loader = PyPDFLoader("data/sample.pdf")
    documents = loader.load()

    # Add metadata
    for doc in documents:
        doc.metadata["source"] = "pdf"

    print("📄 Loading FAQs...")

    try:
        with open("data/faqs.txt", "r", encoding="utf-8") as f:
            faq_text = f.read()

        documents.append(
            Document(page_content=faq_text, metadata={"source": "faq"})
        )
    except Exception as e:
        print(f"⚠️ Error loading faqs.txt: {e}")

    print("✂️ Splitting text...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    docs = splitter.split_documents(documents)

    print(f"📊 Total chunks created: {len(docs)}")  # 🔥 debug

    print("🧠 Creating vector store...")

    # Ensure all processes closed
    time.sleep(2)

    # Delete old DB
    safe_delete_folder("faiss_index")

    # Create new FAISS index
    db = FAISS.from_documents(docs, embeddings)

    db.save_local("faiss_index")

    print("✅ Vector store saved successfully!")


if __name__ == "__main__":
    create_vector_store()