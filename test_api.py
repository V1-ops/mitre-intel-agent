# ============================================
# STABILITY CONFIGURATION
# ============================================
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================
# IMPORTS
# ============================================
from fastapi import FastAPI
from pydantic import BaseModel
import json
import re
from dotenv import load_dotenv

# LangChain imports
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

# ============================================
# ENVIRONMENT SETUP
# ============================================
# Load environment variables from .env file
load_dotenv()

# Get HuggingFace API token from environment
api_token = os.getenv("HUGGINGFACE_API_TOKEN")
if not api_token:
    raise ValueError("HUGGINGFACE_API_TOKEN not found in environment variables")

os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_token

# ============================================
# LOAD DATA AND CREATE RAG PIPELINE AT STARTUP
# ============================================
# IMPORTANT: All heavy objects (embeddings, vector store, LLM) are initialized
# at module level and loaded ONCE when the server starts. They are NOT created
# inside the /analyze endpoint. This ensures efficient memory usage and fast 
# response times for production deployment.
print("🔄 Loading AI models and data...")

# STEP 1: Load MITRE ATT&CK data from JSON source
with open("data/mitre_clean_rag_ready.json", "r") as f:
    mitre_data = json.load(f)
print(f"✅ Loaded {len(mitre_data)} MITRE techniques")

# STEP 2: Create LangChain Documents
documents = []
for technique in mitre_data:
    # Combine technique info into content
    content = f"""Technique ID: {technique['technique_id']}
Name: {technique['name']}
Description: {technique['description']}"""
    
    # Create document with metadata
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

# STEP 3: Initialize embeddings model
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
print("✅ Embedding model loaded")

# STEP 4: Create FAISS vector store from documents at startup
# NOTE: Vector store is rebuilt from source documents on every startup.
# This ensures cloud deployment compatibility (no need for persistent vector_store folder)
vector_store = FAISS.from_documents(documents, embeddings)
print("✅ FAISS vector store created")

# STEP 5: Create retriever (retrieves top k=3 similar documents)
retriever = vector_store.as_retriever(search_kwargs={"k": 3})
print("✅ Retriever configured")

# STEP 6: Initialize LLM with ChatHuggingFace wrapper
llm_endpoint = HuggingFaceEndpoint(
    repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
    temperature=0.3,
    max_new_tokens=500,
    huggingfacehub_api_token=api_token
)
llm = ChatHuggingFace(llm=llm_endpoint)
print("✅ LLM initialized")

# STEP 7: Create prompt template
prompt_template = PromptTemplate(
    input_variables=["log_text", "technique_1", "technique_2", "technique_3"],
    template="""You are a cybersecurity analyst. Analyze this security log and identify the most relevant MITRE ATT&CK technique.

Security Log:
{log_text}

Top 3 Candidate Techniques:
1. {technique_1}
2. {technique_2}
3. {technique_3}

You MUST respond with ONLY valid JSON in this exact format (no additional text):
{{
  "technique_id": "the ID of the most relevant technique",
  "technique_name": "the name of the technique",
  "reasoning": "brief explanation of why this technique matches",
  "confidence": "high, medium, or low"
}}"""
)
print("✅ Prompt template created")

print("🚀 All systems ready!\n")

# ============================================
# INITIALIZE FASTAPI APP
# ============================================
app = FastAPI(title="MITRE SOC Copilot", version="1.0")

# ============================================
# REQUEST/RESPONSE MODELS
# ============================================
# Define what data the API expects to receive
class LogRequest(BaseModel):
    log: str  # The security log to analyze

# ============================================
# API ENDPOINTS
# ============================================

@app.get("/")
def root():
    """
    Root endpoint - Returns service information and status
    """
    return {
        "service_name": "MITRE SOC Copilot",
        "version": "1.0",
        "status": "operational",
        "architecture": "RAG (Retrieval-Augmented Generation)",
        "description": "AI-powered security operations assistant that analyzes security logs and maps them to MITRE ATT&CK techniques using vector similarity search and language model inference."
    }


@app.get("/health")
def health():
    """
    Health check endpoint - Returns service health status
    """
    return {"status": "ok"}


@app.post("/analyze")
def analyze(request: LogRequest):
    """
    Main endpoint - Analyzes a security log and returns MITRE ATT&CK technique
    
    Steps:
    1. Validate input is security-related
    2. Retrieve top 3 similar techniques using LangChain retriever
    3. Create prompt and invoke LLM
    4. Extract and validate JSON response
    5. Add mitigation information
    6. Return final result
    """
    
    try:
        # ============================================
        # STEP 1: INPUT VALIDATION (RULE-BASED)
        # ============================================
        log_text = request.log

        # Basic sanity checks to filter out obvious junk while allowing
        # diverse but valid security logs (DNS, network, process, registry, etc.).

        # 1) Minimum length check (at least 20 characters)
        if len(log_text.strip()) < 20:
            return {
                "error": "Input does not appear to be a valid or complete security log entry."
            }

        # 2) Minimum word count (at least 3 words)
        words = log_text.split()
        if len(words) < 3:
            return {
                "error": "Input does not appear to be a valid or complete security log entry."
            }

        # 3) Require at least one alphabetic character
        # This filters out inputs that are only numbers/symbols (e.g., "12345 ::: !!!").
        if not any(ch.isalpha() for ch in log_text):
            return {
                "error": "Input does not appear to be a valid or complete security log entry."
            }
        
        # ============================================
        # STEP 2: RETRIEVE TOP 3 SIMILAR TECHNIQUES
        # ============================================
        # Use vector store directly to get similarity scores (distances)
        # FAISS returns L2 distance by default
        docs_and_scores = vector_store.similarity_search_with_score(log_text, k=3)
        
        # Extract documents and transform scores
        # Convert L2 distance to similarity score between 0 and 1
        # Formula: similarity = 1 / (1 + distance)
        retrieved_docs = []
        top_techniques = []
        similarity_score = 0.0
        
        for i, (doc, distance) in enumerate(docs_and_scores):
            # Calculate similarity
            score = round(1 / (1 + distance), 2)
            
            # Store the top score for final response
            if i == 0:
                similarity_score = score
                
            retrieved_docs.append(doc)
            top_techniques.append({
                "id": doc.metadata["technique_id"],
                "name": doc.metadata["name"]
            })
        
        # ============================================
        # STEP 3: CREATE PROMPT AND INVOKE LLM (SAFE)
        # ============================================
        try:
            # Format the prompt with retrieved techniques
            formatted_prompt = prompt_template.format(
                log_text=log_text,
                technique_1=f"{top_techniques[0]['id']} - {top_techniques[0]['name']}",
                technique_2=f"{top_techniques[1]['id']} - {top_techniques[1]['name']}",
                technique_3=f"{top_techniques[2]['id']} - {top_techniques[2]['name']}"
            )
            
            # Invoke LLM
            response = llm.invoke(formatted_prompt)
            llm_output = response.content
            
        except Exception:
            return {
                "error": "Analysis failed",
                "message": "Unable to process request"
            }
        
        # ============================================
        # STEP 4: EXTRACT JSON FROM RESPONSE (SAFE)
        # ============================================
        try:
            # Try to find JSON in the response (LLMs sometimes add extra text)
            json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
            
            if not json_match:
                return {
                    "error": "Analysis failed",
                    "message": "Unable to process request"
                }
            
            # Parse the JSON
            result = json.loads(json_match.group())
            
            # Validate required fields
            required_fields = ["technique_id", "technique_name", "reasoning", "confidence"]
            for field in required_fields:
                if field not in result:
                    return {
                        "error": "Analysis failed",
                        "message": "Unable to process request"
                    }
            
        except (json.JSONDecodeError, Exception):
            return {
                "error": "Analysis failed",
                "message": "Unable to process request"
            }
        
        # ============================================
        # STEP 5: FETCH MITIGATION FROM RETRIEVED DOCS
        # ============================================
        # Find mitigation from the retrieved documents' metadata
        mitigation = "No mitigation available"
        
        for doc in retrieved_docs:
            if doc.metadata["technique_id"] == result["technique_id"]:
                mitigations_list = doc.metadata.get("mitigations", [])
                if mitigations_list:
                    mitigation = mitigations_list[0][:500] + "..."
                break
        
        # ============================================
        # STEP 6: SCORING ENHANCEMENTS (SEVERITY & CALIBRATION)
        # ============================================
        
        # 1. SEVERITY SCORING (Rule-based)
        technique_name = result["technique_name"].lower()
        technique_id = result["technique_id"].upper()
        severity = "Medium"  # Default (covers Persistence, Discovery, etc.)
        
        # Define high-risk patterns
        high_risk_keywords = [
            "escalation", "credential", "lateral", "command and control", 
            "impact", "encryption", "ransomware", "brute force", 
            "password guessing", "password spraying"
        ]
        
        # Define low-risk patterns
        low_risk_keywords = ["reconnaissance"]
        
        # Check for HIGH severity (Keywords OR specific IDs)
        if any(keyword in technique_name for keyword in high_risk_keywords) or technique_id.startswith("T1110"):
            severity = "High"
        # Check for LOW severity
        elif any(keyword in technique_name for keyword in low_risk_keywords):
            severity = "Low"
            
        # 2. CONFIDENCE CALIBRATION
        # Combine FAISS similarity and LLM confidence
        llm_conf = result["confidence"].capitalize()
        final_confidence = "Low"
        
        if similarity_score > 0.85 and llm_conf == "High":
            final_confidence = "High"
        elif similarity_score >= 0.70:
            final_confidence = "Medium"
        else:
            final_confidence = "Low"

        # ============================================
        # STEP 7: BUILD FINAL RESPONSE
        # ============================================
        final_response = {
            "technique_id": result["technique_id"],
            "technique_name": result["technique_name"],
            "similarity_score": similarity_score,
            "severity": severity,
            "confidence": final_confidence,
            "reasoning": result["reasoning"],
            "mitigation": mitigation,
            "log_analyzed": log_text
        }
        
        return final_response
        
    except Exception:
        return {
            "error": "Analysis failed",
            "message": "Unable to process request"
        }


# ============================================
# RUN THE APP
# ============================================
# To run this app, use: uvicorn test_api:app --reload
# Then visit: http://localhost:8000/docs
