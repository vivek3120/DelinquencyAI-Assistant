# DelinquencyAI-Assistant

## Financial Risk Prediction + Microsoft Fabric + Power BI + RAG + Responsible AI

DelinquencyAI Assistant is an end-to-end financial AI project that predicts customer delinquency/default risk and explains the prediction using policy-grounded AI reasoning.

The project combines **Microsoft Fabric**, **machine learning**, **Power BI reporting**, and a **Streamlit AI assistant**. The goal is to show how financial institutions can use predictive analytics and AI to identify high-risk borrowers, explain risk drivers, recommend collection actions, and apply Responsible AI checks.

---

## Live Demo

Streamlit App: ' https://delinquencyai-assistant-ai.streamlit.app/'

Power BI Report: 
<img width="939" height="526" alt="image" src="https://github.com/user-attachments/assets/f1944184-3603-42dc-bbb1-e4eb6cbddea9" />
<img width="939" height="542" alt="image" src="https://github.com/user-attachments/assets/097610cf-3db5-478c-83b4-68ccf8f0b643" />

---

## Project Summary

This project has two main parts:

### 1. Microsoft Fabric ML Pipeline

The Fabric side handles data engineering, model training, scoring, monitoring, and reporting.
It includes:

- Lakehouse-based Bronze, Silver, and Gold tables
- Data cleaning and feature engineering
- Delinquency/default prediction model
- MLflow experiment tracking
- Prediction, action, and model-monitoring tables
- Power BI dashboard for business users

### 2. Streamlit AI Assistant
The Streamlit side adds the AI layer.

It includes:
- RAG over financial policy documents
- Sentence-transformer embeddings
- FAISS vector search
- Customer-level risk explanation
- Policy-supported recommended actions
- Responsible AI checks
- Human-review recommendations

---

## Business Problem

Financial institutions need to identify customers who are likely to become delinquent or default. However, a normal machine learning score is not enough for business users.

Business teams also need to know:

- Which customers are high risk?
- Why are they high risk?
- What action should be taken?
- Which policy supports the recommendation?
- Does the case require human review?
- Is the AI output explainable and responsible?

DelinquencyAI solves this by combining predictive modeling with policy-grounded AI explanations.

---

## Solution Overview
The project predicts customer delinquency risk and converts the model output into an explainable business decision.
