from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
import os
import shutil
import time

# Load embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


def safe_delete_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path, ignore_errors=True)
        print(f"🗑️ Deleted old {folder_path}")


def create_vector_store():
    documents = []

    # ================= PDF =================
    print("📄 Loading PDF...")

    loader = PyPDFLoader("data/sample.pdf")
    pages = loader.load()

    text = "\n".join([p.page_content for p in pages])
    lines = text.split("\n")

    current_q = None

    print("🔍 Extracting PDF Q&A...")

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # Question
        if line.startswith("Q:"):
            current_q = line

        # Answer
        elif line.startswith("A:") and current_q:
            qa_text = f"{current_q}\n{line}"

            documents.append(
                Document(
                    page_content=qa_text,
                    metadata={"source": "pdf"}
                )
            )

            current_q = None

    print(f"📊 PDF Q&A added: {len(documents)}")

    # ================= FAQ =================
    print("📄 Loading FAQs...")

    try:
        with open("data/faqs.txt", "r", encoding="utf-8") as f:
            faq_text = f.read()

        faq_items = faq_text.split("Q:")

        for item in faq_items:
            if item.strip():
                qa = "Q:" + item.strip()

                documents.append(
                    Document(
                        page_content=qa,
                        metadata={"source": "faq"}
                    )
                )

        print("📊 FAQ Q&A added")

    except Exception as e:
        print(f"⚠️ Error loading faqs.txt: {e}")

    print(f"📊 Total documents: {len(documents)}")

    time.sleep(1)

    # Delete old DB
    safe_delete_folder("faiss_index")

    print("🧠 Creating vector store...")

    db = FAISS.from_documents(documents, embeddings)
    db.save_local("faiss_index")

    print("✅ Vector store saved successfully!")


if __name__ == "__main__":
    create_vector_store()  