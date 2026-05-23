import os
import glob
import numpy as np
import pandas as pd
import streamlit as st
import faiss
from sentence_transformers import SentenceTransformer

#----

st.set_page_config(
    page_title="DelinquencyAI Assistant",
    page_icon="📊",
    layout="wide"
)

DATA_DIR = "data"
POLICY_DIR = "policies"

PREDICTION_FILE = os.path.join(DATA_DIR, "gold_delinquency_predictions.csv")
ACTIONS_FILE = os.path.join(DATA_DIR, "gold_collection_actions.csv")
MONITORING_FILE = os.path.join(DATA_DIR, "gold_model_monitoring.csv")


# ---

@st.cache_data
def load_fabric_exports():
    predictions = pd.read_csv(PREDICTION_FILE)
    actions = pd.read_csv(ACTIONS_FILE)
    monitoring = pd.read_csv(MONITORING_FILE)

    if "CUSTOMER_ID" in predictions.columns:
        predictions["CUSTOMER_ID"] = pd.to_numeric(
            predictions["CUSTOMER_ID"], errors="coerce"
        ).astype("Int64")

    if "CUSTOMER_ID" in actions.columns:
        actions["CUSTOMER_ID"] = pd.to_numeric(
            actions["CUSTOMER_ID"], errors="coerce"
        ).astype("Int64")

    return predictions, actions, monitoring


# ---

def chunk_text(text, chunk_size=120, overlap=25):
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap

    return chunks


@st.cache_data
def load_policy_documents():
    docs = []

    policy_files = glob.glob(os.path.join(POLICY_DIR, "*.txt"))

    for path in policy_files:
        with open(path, "r", encoding="utf-8") as file:
            text = file.read()

        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            docs.append(
                {
                    "source": os.path.basename(path),
                    "chunk_id": i,
                    "text": chunk,
                }
            )

    return docs


# ---

@st.cache_resource
def build_vector_index():
    docs = load_policy_documents()

    if len(docs) == 0:
        raise ValueError("No policy documents found. Please add .txt files inside the policies folder.")

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    texts = [doc["text"] for doc in docs]
    embeddings = embedder.encode(texts, normalize_embeddings=True)

    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return embedder, index, docs


def retrieve_policy_context(query, top_k=4):
    embedder, index, docs = build_vector_index()

    query_embedding = embedder.encode([query], normalize_embeddings=True)
    query_embedding = np.array(query_embedding).astype("float32")

    scores, ids = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], ids[0]):
        if idx >= 0:
            doc = docs[idx]
            results.append(
                {
                    "source": doc["source"],
                    "chunk_id": doc["chunk_id"],
                    "score": float(score),
                    "text": doc["text"],
                }
            )

    return results


# ---

def get_customer_data(customer_id, predictions, actions):
    pred_match = predictions[predictions["CUSTOMER_ID"] == customer_id]
    action_match = actions[actions["CUSTOMER_ID"] == customer_id]

    if pred_match.empty:
        return None, None

    pred_row = pred_match.iloc[0].to_dict()
    action_row = action_match.iloc[0].to_dict() if not action_match.empty else {}

    return pred_row, action_row


def build_risk_summary(pred_row):
    risk_band = pred_row.get("RISK_BAND", "UNKNOWN")
    probability = pred_row.get("DEFAULT_PROBABILITY", 0)
    driver1 = pred_row.get("TOP_RISK_DRIVER_1", "UNKNOWN")
    driver2 = pred_row.get("TOP_RISK_DRIVER_2", "UNKNOWN")
    driver3 = pred_row.get("TOP_RISK_DRIVER_3", "UNKNOWN")
    action = pred_row.get("RECOMMENDED_ACTION", "REVIEW_REQUIRED")

    return (
        f"This borrower is classified as {risk_band} risk with an estimated "
        f"default probability of {probability:.2%}. The main risk drivers are "
        f"{driver1}, {driver2}, and {driver3}. The recommended action is {action}."
    )


def responsible_ai_checks(pred_row, retrieved_context):
    risk_band = pred_row.get("RISK_BAND", "UNKNOWN")
    probability = pred_row.get("DEFAULT_PROBABILITY", 0)

    policy_context_found = "YES" if len(retrieved_context) > 0 else "NO"
    hallucination_risk = "LOW" if len(retrieved_context) > 0 else "HIGH"
    human_review_required = "YES" if risk_band == "HIGH" else "NO"

    if risk_band == "HIGH":
        governance_note = "High-risk borrowers require human analyst review before final collections action."
    elif probability >= 0.40:
        governance_note = "Medium-risk borrowers should receive monitoring and payment reminder review."
    else:
        governance_note = "Low-risk borrowers can remain in standard monitoring."

    return {
        "policy_context_found": policy_context_found,
        "hallucination_risk": hallucination_risk,
        "human_review_required": human_review_required,
        "decision_support_only": "YES",
        "governance_note": governance_note,
    }


def build_final_answer(question, pred_row, action_row, retrieved_context):
    risk_band = pred_row.get("RISK_BAND", "UNKNOWN")
    probability = pred_row.get("DEFAULT_PROBABILITY", 0)
    action = pred_row.get("RECOMMENDED_ACTION", "REVIEW_REQUIRED")

    driver1 = pred_row.get("TOP_RISK_DRIVER_1", "UNKNOWN")
    driver2 = pred_row.get("TOP_RISK_DRIVER_2", "UNKNOWN")
    driver3 = pred_row.get("TOP_RISK_DRIVER_3", "UNKNOWN")

    sources = sorted(set([item["source"] for item in retrieved_context]))
    source_text = ", ".join(sources) if sources else "No policy source found"

    checks = responsible_ai_checks(pred_row, retrieved_context)

    if risk_band == "HIGH":
        decision = "Escalate to the collections review team for early intervention and human review."
    elif risk_band == "MEDIUM":
        decision = "Send a payment reminder and continue monitoring."
    else:
        decision = "Continue standard portfolio monitoring."

    answer = f"""
### Final AI Risk Answer

**Question:** {question}

**Risk Level:** {risk_band}

**Default Probability:** {probability:.2%}

**Reason:**  
This borrower is flagged as **{risk_band} risk** mainly because of these drivers:

1. `{driver1}`
2. `{driver2}`
3. `{driver3}`

**Recommended Action:**  
{decision}

**Business Action from Model:**  
`{action}`

**Policy Support:**  
The recommendation is supported by retrieved policy context from: **{source_text}**

**Responsible AI Note:**  
This is a decision-support prototype. It should not be used as the sole basis for lending, credit, or collections decisions. High-risk cases require human review.

**Governance Checks:**  
- Policy context found: `{checks["policy_context_found"]}`
- Human review required: `{checks["human_review_required"]}`
- Hallucination risk: `{checks["hallucination_risk"]}`
- Decision-support only: `{checks["decision_support_only"]}`
"""

    return answer, checks


#---

try:
    predictions, actions, monitoring = load_fabric_exports()
except Exception as e:
    st.error(f"Could not load data files. Please check the data folder. Error: {e}")
    st.stop()


# ---

st.sidebar.title("DelinquencyAI")
st.sidebar.caption("Fabric + ML + RAG + Responsible AI")

page = st.sidebar.radio(
    "Select Page",
    [
        "Project Overview",
        "Risk Dashboard",
        "Borrower Lookup",
        "Policy RAG Search",
        "AI Risk Assistant",
        "Responsible AI Evaluation",
    ],
)


# ---

if page == "Project Overview":
    st.title("DelinquencyAI Assistant")
    st.subheader("Financial Risk Prediction + RAG + Responsible AI")

    st.markdown(
        """
        This app is the AI assistant layer of the **DelinquencyAI** project.

        The Microsoft Fabric side created:

        - Bronze, Silver, and Gold Lakehouse tables
        - Delinquency/default prediction model
        - MLflow model metrics
        - Power BI risk dashboard
        - Exported prediction, action, and monitoring files

        This Streamlit app adds:

        - RAG over financial policy documents
        - Hugging Face sentence-transformer embeddings
        - FAISS vector search
        - AI-style risk explanation
        - Responsible AI checks
        - Human review recommendation
        """
    )

    st.info(
        "This Streamlit Cloud version is lightweight and free. In production, the final generation layer can be replaced with Azure OpenAI."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("Prediction Records", f"{len(predictions):,}")
    col2.metric("Action Records", f"{len(actions):,}")
    col3.metric("Monitoring Records", f"{len(monitoring):,}")

    st.subheader("Project Architecture")

    st.code(
        """
Microsoft Fabric Lakehouse
    ↓
Bronze / Silver / Gold Tables
    ↓
ML Model + MLflow
    ↓
Prediction / Action / Monitoring Tables
    ↓
Power BI Dashboard
    ↓
Streamlit AI Assistant
    ↓
RAG over Policy Documents + Responsible AI Checks
        """,
        language="text",
    )


#----

elif page == "Risk Dashboard":
    st.title("Risk Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    total_customers = len(predictions)
    high_risk_count = int((predictions["RISK_BAND"] == "HIGH").sum())
    avg_default_probability = predictions["DEFAULT_PROBABILITY"].mean()

    latest_monitoring = monitoring.sort_values("RUN_DATE").iloc[-1]
    latest_auc = latest_monitoring["AUC_SCORE"]

    col1.metric("Total Customers", f"{total_customers:,}")
    col2.metric("High Risk Customers", f"{high_risk_count:,}")
    col3.metric("Avg Default Probability", f"{avg_default_probability:.2%}")
    col4.metric("Latest AUC", f"{latest_auc:.3f}")

    st.subheader("Risk Band Distribution")
    risk_counts = predictions["RISK_BAND"].value_counts()
    st.bar_chart(risk_counts)

    st.subheader("Recommended Actions")
    action_counts = predictions["RECOMMENDED_ACTION"].value_counts()
    st.bar_chart(action_counts)

    st.subheader("Prediction Records")
    st.dataframe(predictions.head(200), use_container_width=True)


#-----

elif page == "Borrower Lookup":
    st.title("Borrower Lookup")

    customer_ids = predictions["CUSTOMER_ID"].dropna().astype(int).tolist()

    customer_id = st.selectbox(
        "Select CUSTOMER_ID",
        customer_ids,
        index=0,
    )

    pred_row, action_row = get_customer_data(customer_id, predictions, actions)

    if pred_row is None:
        st.error("Customer not found.")
    else:
        col1, col2, col3 = st.columns(3)

        col1.metric("Risk Band", pred_row.get("RISK_BAND", "UNKNOWN"))
        col2.metric("Default Probability", f"{pred_row.get('DEFAULT_PROBABILITY', 0):.2%}")
        col3.metric("Recommended Action", pred_row.get("RECOMMENDED_ACTION", "UNKNOWN"))

        st.subheader("Top Risk Drivers")
        st.write("1.", pred_row.get("TOP_RISK_DRIVER_1", "UNKNOWN"))
        st.write("2.", pred_row.get("TOP_RISK_DRIVER_2", "UNKNOWN"))
        st.write("3.", pred_row.get("TOP_RISK_DRIVER_3", "UNKNOWN"))

        st.subheader("Fabric Explanation")
        st.info(pred_row.get("AI_READY_EXPLANATION", "No explanation available."))

        st.subheader("Action Record")
        st.json(action_row)


#----

elif page == "Policy RAG Search":
    st.title("Policy RAG Search")

    st.markdown(
        """
        Ask a policy question. The app converts your question into an embedding, 
        searches the policy documents using FAISS, and returns the most relevant policy chunks.
        """
    )

    query = st.text_input(
        "Ask a policy question",
        value="What should happen for high risk borrowers?",
    )

    top_k = st.slider("Number of policy chunks to retrieve", 1, 6, 4)

    if st.button("Search Policies"):
        with st.spinner("Building embeddings and searching policies..."):
            results = retrieve_policy_context(query, top_k=top_k)

        st.subheader("Retrieved Policy Evidence")

        for result in results:
            st.markdown(f"### Source: {result['source']}")
            st.caption(f"Similarity score: {result['score']:.3f}")
            st.write(result["text"])


#----

elif page == "AI Risk Assistant":
    st.title("AI Risk Assistant")

    st.markdown(
        """
        This page combines the Fabric ML prediction, RAG policy retrieval, 
        and responsible AI checks to produce a policy-grounded risk explanation.
        """
    )

    customer_ids = predictions["CUSTOMER_ID"].dropna().astype(int).tolist()

    customer_id = st.selectbox(
        "Select CUSTOMER_ID",
        customer_ids,
        index=0,
        key="assistant_customer",
    )

    question = st.text_area(
        "Ask a question",
        value="Why is this borrower risky and what action should we take?",
    )

    if st.button("Run AI Analysis"):
        pred_row, action_row = get_customer_data(customer_id, predictions, actions)

        if pred_row is None:
            st.error("Customer not found.")
        else:
            rag_query = (
                f"{question}. "
                f"Risk band: {pred_row.get('RISK_BAND')}. "
                f"Default probability: {pred_row.get('DEFAULT_PROBABILITY')}. "
                f"Recommended action: {pred_row.get('RECOMMENDED_ACTION')}. "
                f"Risk drivers: {pred_row.get('TOP_RISK_DRIVER_1')}, "
                f"{pred_row.get('TOP_RISK_DRIVER_2')}, "
                f"{pred_row.get('TOP_RISK_DRIVER_3')}."
            )

            with st.spinner("Retrieving policy evidence..."):
                retrieved_context = retrieve_policy_context(rag_query, top_k=4)

            answer, checks = build_final_answer(
                question,
                pred_row,
                action_row,
                retrieved_context,
            )

            st.markdown(answer)

            st.subheader("Responsible AI Checks")
            st.json(checks)

            st.subheader("Retrieved Policy Sources")
            for context in retrieved_context:
                with st.expander(f"{context['source']} | Score: {context['score']:.3f}"):
                    st.write(context["text"])


------

elif page == "Responsible AI Evaluation":
    st.title("Responsible AI Evaluation")

    latest = monitoring.sort_values("RUN_DATE").iloc[-1]

    col1, col2, col3, col4, col5 = st.columns(5)

    col1.metric("AUC", f"{latest['AUC_SCORE']:.3f}")
    col2.metric("Precision", f"{latest['PRECISION_SCORE']:.3f}")
    col3.metric("Recall", f"{latest['RECALL_SCORE']:.3f}")
    col4.metric("F1 Score", f"{latest['F1_SCORE']:.3f}")
    col5.metric("False Negatives", int(latest["FALSE_NEGATIVES"]))

    st.subheader("Governance Statement")

    st.warning(
        "This model is a decision-support prototype. It should not be used as the sole basis for lending, credit, or collections decisions. High-risk cases require human review."
    )

    st.subheader("Model Monitoring Table")
    st.dataframe(monitoring, use_container_width=True)

    st.subheader("High Risk Customers Requiring Human Review")
    high_risk = predictions[predictions["RISK_BAND"] == "HIGH"]
    st.dataframe(high_risk, use_container_width=True)

    st.subheader("Responsible AI Checklist")

    checklist = pd.DataFrame(
        [
            {"Check": "Model output includes probability", "Status": "PASS"},
            {"Check": "Risk decision includes reason drivers", "Status": "PASS"},
            {"Check": "Policy context is retrieved using RAG", "Status": "PASS"},
            {"Check": "High risk cases require human review", "Status": "PASS"},
            {"Check": "System is labeled as decision-support only", "Status": "PASS"},
        ]
    )

    st.dataframe(checklist, use_container_width=True)