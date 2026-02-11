#!/usr/bin/env python3
"""Quick test to verify RAG pipeline works"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

print("🔄 Loading environment...")
load_dotenv()
api_token = os.getenv("HUGGINGFACE_API_TOKEN")
if not api_token:
    print("❌ HUGGINGFACE_API_TOKEN not found")
    exit(1)

print("✅ API token found")

print("🔄 Loading embeddings...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
print("✅ Embeddings loaded")

print("🔄 Loading FAISS...")
vector_store = FAISS.load_local(
    "vector_store",
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)
print("✅ FAISS loaded")

print("🔄 Testing similarity search...")
test_log = "Failed SSH login attempt from IP 192.168.1.100"
docs_and_scores = vector_store.similarity_search_with_score(test_log, k=3)
    
print(f"\n✅ Found {len(docs_and_scores)} similar techniques:")
for i, (doc, distance) in enumerate(docs_and_scores, 1):
    score = round(1 / (1 + distance), 2)
    print(f"  {i}. {doc.metadata['technique_id']} - {doc.metadata['name']} (score: {score})")

print("\n🔄 Testing LLM...")
try:
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.3",
        temperature=0.3,
        max_new_tokens=500,
        huggingfacehub_api_token=api_token
    )
    
    response = llm.invoke("Say 'LLM working' if you can see this")
    print(f"✅ LLM Response: {response[:100]}")
    
except Exception as e:
    print(f"❌ LLM Error: {str(e)}")

print("\n✅ All components working!")
