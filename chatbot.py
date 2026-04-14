from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests

# =========================
# LOAD DB
# =========================
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

db = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# =========================
# CLEAN INPUT
# =========================
def clean(text):
    return text.lower().replace("?", "").strip()

# =========================
# CLEAN OUTPUT (IMPORTANT)
# =========================
def clean_output(text):
    if not text:
        return text

    text = text.split("\n")[0]

    junk_words = [
        "Sure", "Respuesta", "Pasaje", "Paseo",
        "Rules", "Context", "Instructions"
    ]

    for w in junk_words:
        text = text.replace(w, "")

    return text.strip()

# =========================
# LLM FUNCTION (STRICT MODE)
# =========================
def generate_llm_response(context, question):
    # 🚨 If context is empty → don't use LLM
    if not context or len(context.strip()) < 5:
        return "I don't have this information in the provided documents."

    prompt = f"""
You are a STRICT customer support bot.

RULE:
- Use ONLY the context
- Return ONLY 1 short sentence
- NO explanation
- NO repetition
- NO extra text
- English only

Context:
{context}

Question:
{question}

Answer:
"""

    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model":"phi3:mini",
                "prompt": prompt,
                "stream": False,
                "temperature": 0.0,
                "top_p": 0.01
            },
            timeout=60
        )

        return clean_output(response.json().get("response", ""))

    except:
        return context


# =========================
# MAIN FUNCTION
# =========================
def get_response(query):
    try:
        query_clean = clean(query)

        docs_with_score = db.similarity_search_with_score(query, k=1)

        best_answer = None

        # =========================
        # STRICT MATCH SEARCH
        # =========================
        for doc, score in docs_with_score:

            if score > 2.0:
                continue

            text = doc.page_content

            if "Q:" in text and "A:" in text:
                q_part = text.split("A:")[0].replace("Q:", "").strip()
                a_part = text.split("A:")[1].strip()

                if clean(q_part) == query_clean:
                    best_answer = a_part.split("\n")[0].strip()
                    break

        # =========================
        # NO DATA CASE
        # =========================
        if not best_answer:
            return {
                "answer": "I don't have this information in the provided documents.",
                "sources": []
            }

        # =========================
        # FAST PATH (NO LLM FOR SMALL ANSWERS)
        # =========================
        if len(best_answer.split()) <= 10:
            return {
                "answer": best_answer,
                "sources": ["pdf", "faq"]
            }

        # =========================
        # LLM ONLY FOR POLISHING
        # =========================
        final_answer = generate_llm_response(best_answer, query)

        # safety fallback
        if not final_answer:
            final_answer = best_answer

        return {
            "answer": final_answer,
            "sources": ["pdf", "faq"]
        }

    except Exception as e:
        print("Error:", e)
        return {
            "answer": "Something went wrong.",
            "sources": []
        }