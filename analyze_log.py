import json
import os
from dotenv import load_dotenv

# LangChain imports
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# -----------------------------
# Environment Setup
# -----------------------------

api_token = os.getenv("HUGGINGFACE_API_TOKEN")
if not api_token:
    raise ValueError("HUGGINGFACE_API_TOKEN not found in .env file")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_token

# -----------------------------
# Load MITRE Data and Create Documents
# -----------------------------

print("Loading MITRE ATT&CK data...")
with open("data/mitre_clean_rag_ready.json", "r") as f:
    mitre_data = json.load(f)

# Create LangChain Documents
documents = []
for technique in mitre_data:
    content = f"""Technique ID: {technique['technique_id']}
Name: {technique['name']}
Description: {technique['description']}"""
    
    doc = Document(
        page_content=content,
        metadata={
            "technique_id": technique["technique_id"],
            "name": technique["name"],
            "description": technique["description"],
            "mitigations": technique.get("mitigations", [])
        }
    )
    documents.append(doc)

print(f"✅ Created {len(documents)} documents")

# -----------------------------
# Initialize Embeddings and Vector Store
# -----------------------------

print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
print("✅ Embedding model loaded")

print("Creating FAISS vector store...")
vector_store = FAISS.from_documents(documents, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
print("✅ Vector store ready")

# -----------------------------
# Initialize HuggingFace LLM
# -----------------------------

print("Initializing HuggingFace LLM...")
llm_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    temperature=0.3,
    max_new_tokens=1024,
    huggingfacehub_api_token=api_token
)
llm = ChatHuggingFace(llm=llm_endpoint)
print("✅ HuggingFace LLM initialized")

# -----------------------------
# Take User Input
# -----------------------------

log = input("\nEnter suspicious log: ")

# -----------------------------
# Retrieve Top 3 Similar Techniques
# -----------------------------

print("\nRetrieving relevant techniques...")
retrieved_docs = retriever.invoke(log)

# Extract technique information
top_techniques = []
for doc in retrieved_docs:
    top_techniques.append({
        "id": doc.metadata["technique_id"],
        "name": doc.metadata["name"],
        "description": doc.metadata["description"][:200]
    })

print(f"✅ Retrieved {len(top_techniques)} techniques")

# -----------------------------
# Create Prompt and Invoke LLM
# -----------------------------

prompt_template = PromptTemplate(
    input_variables=["log_text", "technique_1", "technique_2", "technique_3"],
    template="""You are a cybersecurity SOC analyst. Analyze this security log and identify the most relevant MITRE ATT&CK technique.

Security Log:
{log_text}

Top 3 Candidate Techniques:
1. {technique_1}
2. {technique_2}
3. {technique_3}

Respond ONLY in this JSON format:
{{
  "technique_id": "",
  "technique_name": "",
  "reasoning": "",
  "mitigation": "",
  "confidence": "Low/Medium/High"
}}

Do not include anything outside JSON."""
)

formatted_prompt = prompt_template.format(
    log_text=log,
    technique_1=f"{top_techniques[0]['id']} - {top_techniques[0]['name']}",
    technique_2=f"{top_techniques[1]['id']} - {top_techniques[1]['name']}",
    technique_3=f"{top_techniques[2]['id']} - {top_techniques[2]['name']}"
)

# -----------------------------
# Call HuggingFace LLM
# -----------------------------

print("\nAnalyzing with HuggingFace LLM...\n")

response = llm.invoke(formatted_prompt)
output_text = response.content

print("Raw Model Output:\n")
print(output_text)

# -----------------------------
# Parse and Display Clean Output
# -----------------------------

import re

try:
    # Try to find JSON in the response (LLMs sometimes add extra text)
    json_match = re.search(r'\{.*\}', output_text, re.DOTALL)
    
    if json_match:
        result = json.loads(json_match.group())
    else:
        result = json.loads(output_text)

    print("\n===== SOC ANALYSIS =====")
    print(f"Technique: {result.get('technique_id', 'N/A')} - {result.get('technique_name', 'N/A')}")
    print(f"Confidence: {result.get('confidence', 'N/A')}")
    print(f"\nReasoning:\n{result.get('reasoning', 'N/A')}")
    print(f"\nMitigation:\n{result.get('mitigation', 'N/A')}")

except Exception as e:
    print(f"\n⚠ JSON parsing failed: {str(e)}")
    print("Showing raw output above.")
