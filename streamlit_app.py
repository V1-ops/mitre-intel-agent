# ============================================
# STABILITY CONFIGURATION
# ============================================
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ============================================
# IMPORTS
# ============================================
import streamlit as st
import json
import re
from pathlib import Path
from dotenv import load_dotenv

# LangChain imports
from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings, ChatHuggingFace
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="MITRE SOC Copilot",
    page_icon="🛡️",
    layout="centered"
)

# ============================================
# LOAD ENVIRONMENT & API TOKEN
# ============================================
load_dotenv()
api_token = os.getenv("HUGGINGFACE_API_TOKEN")
if not api_token:
    st.error("❌ HUGGINGFACE_API_TOKEN not found. Please set it in .env file.")
    st.stop()

# ============================================
# LOAD PREBUILT FAISS VECTOR STORE (CACHED)
# ============================================
@st.cache_resource
def load_rag_pipeline():
    """
    Loads the prebuilt FAISS vector store and initializes LLM.
    Vector store is loaded from disk (no embeddings needed).
    This is cached so it only runs once per session.
    Returns: (vector_store, llm, prompt_template)
    """
    # Load prebuilt FAISS vector store
    vector_store_path = Path("vector_store")
    
    if not vector_store_path.exists():
        st.error(f"❌ Vector store not found at {vector_store_path}")
        st.error("Please run: python build_index.py")
        st.stop()
    
    # Initialize embeddings model (needed for query-time embedding)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    # Load prebuilt FAISS index (no rebuild needed, just load from disk)
    vector_store = FAISS.load_local(
        "vector_store",
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    
    # Initialize LLM via HuggingFace API with chat wrapper
    llm_endpoint = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-72B-Instruct",
        temperature=0.0,
        max_new_tokens=500,
        huggingfacehub_api_token=api_token
    )
    llm = ChatHuggingFace(llm=llm_endpoint)
    
    # Create prompt template
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
    
    return vector_store, llm, prompt_template

# ============================================
# HELPER FUNCTIONS
# ============================================
def validate_input(log_text):
    """Simple input validation"""
    if len(log_text.strip()) < 20:
        return False, "Input must be at least 20 characters."
    
    words = log_text.split()
    if len(words) < 3:
        return False, "Input must contain at least 3 words."
    
    if not any(ch.isalpha() for ch in log_text):
        return False, "Input must contain alphabetic characters."
    
    return True, ""

def classify_severity(technique_name, technique_id):
    """Rule-based severity classification"""
    technique_name_lower = technique_name.lower()
    technique_id_upper = technique_id.upper()
    
    high_risk_keywords = [
        "escalation", "credential", "lateral", "command and control",
        "impact", "encryption", "ransomware", "brute force",
        "password guessing", "password spraying"
    ]
    low_risk_keywords = ["reconnaissance"]
    
    if any(kw in technique_name_lower for kw in high_risk_keywords) or technique_id_upper.startswith("T1110"):
        return "High"
    elif any(kw in technique_name_lower for kw in low_risk_keywords):
        return "Low"
    else:
        return "Medium"

def calibrate_confidence(similarity_score, llm_confidence):
    """Calibrate confidence using similarity score and LLM output"""
    llm_conf_cap = llm_confidence.capitalize()
    
    if similarity_score > 0.85 and llm_conf_cap == "High":
        return "High"
    elif similarity_score >= 0.70:
        return "Medium"
    else:
        return "Low"

def analyze_log(log_text, vector_store, llm, prompt_template):
    """Main analysis function"""
    try:
        # --------------------------------------------
        # 1️⃣ Retrieve Top 3 Similar Techniques
        # --------------------------------------------
        docs_and_scores = vector_store.similarity_search_with_score(log_text, k=3)

        if len(docs_and_scores) < 3:
            return {"error": "Analysis failed", "message": "Not enough MITRE techniques retrieved"}

        retrieved_docs = []
        top_techniques = []
        similarity_score = 0.0

        for i, (doc, distance) in enumerate(docs_and_scores):
            score = round(1 / (1 + distance), 2)

            if i == 0:
                similarity_score = score

            retrieved_docs.append(doc)
            top_techniques.append({
                "id": doc.metadata.get("technique_id", "Unknown"),
                "name": doc.metadata.get("name", "Unknown")
            })

        # --------------------------------------------
        # 2️⃣ Format Prompt
        # --------------------------------------------
        formatted_prompt = prompt_template.format(
            log_text=log_text,
            technique_1=f"{top_techniques[0]['id']} - {top_techniques[0]['name']}",
            technique_2=f"{top_techniques[1]['id']} - {top_techniques[1]['name']}",
            technique_3=f"{top_techniques[2]['id']} - {top_techniques[2]['name']}"
        )

        # --------------------------------------------
        # 3️⃣ Invoke LLM (Fix for AIMessage issue)
        # --------------------------------------------
        llm_response = llm.invoke(formatted_prompt)

        # Extract text safely from AIMessage
        if hasattr(llm_response, "content"):
            llm_output = llm_response.content
        else:
            llm_output = str(llm_response)

        # --------------------------------------------
        # 4️⃣ Safe JSON Extraction
        # --------------------------------------------
        try:
            json_start = llm_output.find("{")
            json_end = llm_output.rfind("}") + 1

            if json_start == -1 or json_end == -1:
                return {"error": "Analysis failed", "message": "No JSON found in LLM response"}

            json_str = llm_output[json_start:json_end]
            result = json.loads(json_str)

        except Exception:
            return {"error": "Analysis failed", "message": "Invalid JSON returned by LLM"}

        # --------------------------------------------
        # 5️⃣ Validate Required Fields
        # --------------------------------------------
        required_fields = ["technique_id", "technique_name", "reasoning", "confidence"]

        for field in required_fields:
            if field not in result:
                return {"error": "Analysis failed", "message": f"Missing field: {field}"}

        # --------------------------------------------
        # 6️⃣ Fetch Mitigation
        # --------------------------------------------
        mitigation = "No mitigation available"

        for doc in retrieved_docs:
            if doc.metadata.get("technique_id") == result["technique_id"]:
                mitigations_list = doc.metadata.get("mitigations", [])
                if mitigations_list:
                    mitigation = mitigations_list[0][:500] + "..."
                break

        # --------------------------------------------
        # 7️⃣ Scoring Enhancements
        # --------------------------------------------
        severity = classify_severity(
            result["technique_name"],
            result["technique_id"]
        )

        final_confidence = calibrate_confidence(
            similarity_score,
            result["confidence"]
        )

        # --------------------------------------------
        # 8️⃣ Final Response
        # --------------------------------------------
        return {
            "technique_id": result["technique_id"],
            "technique_name": result["technique_name"],
            "similarity_score": similarity_score,
            "severity": severity,
            "confidence": final_confidence,
            "reasoning": result["reasoning"],
            "mitigation": mitigation,
            "log_analyzed": log_text
        }

    except Exception as e:
        return {"error": "Analysis failed", "message": str(e)}

# ============================================
# MAIN UI
# ============================================
st.title("🛡️ MITRE SOC Copilot")
st.markdown(
    """
    **Production-Grade RAG System** using the MITRE ATT&CK framework  
    Analyzes security logs and maps them to attack techniques with AI-powered intelligence.
    """
)

st.divider()

# ============================================
# LOAD RAG PIPELINE
# ============================================
with st.spinner("🔄 Loading prebuilt vector store and LLM..."):
    try:
        vector_store, llm, prompt_template = load_rag_pipeline()
        st.success("✅ System ready!")
    except Exception as e:
        st.error(f"❌ Failed to initialize system: {str(e)}")
        st.stop()

st.divider()

# ============================================
# INPUT SECTION
# ============================================
st.subheader("Enter Security Log")

log_input = st.text_area(
    label="Security Log",
    placeholder="Example: Failed login attempt from IP 192.168.1.100 user admin at 2024-01-15 10:30:45",
    height=150,
    label_visibility="collapsed"
)

analyze_button = st.button("🔍 Analyze Log", use_container_width=True)

# ============================================
# ANALYSIS SECTION
# ============================================
if analyze_button:
    if not log_input.strip():
        st.warning("⚠️ Please enter a security log to analyze.")
    else:
        # Validate input
        is_valid, error_msg = validate_input(log_input)
        if not is_valid:
            st.error(f"❌ {error_msg}")
        else:
            # Analyze log
            with st.spinner("🔄 Analyzing log with AI..."):
                result = analyze_log(log_input, vector_store, llm, prompt_template)
                
                if result is None or "error" in result:
                    error_msg = result.get('message', 'Unable to process request') if result else 'Unable to process request'
                    st.error(f"❌ Analysis failed: {error_msg}")
                else:
                    # ============================================
                    # DISPLAY RESULTS
                    # ============================================
                    st.success("✅ Analysis Complete")
                    
                    # Technique ID and Name
                    st.subheader(f"{result['technique_id']}: {result['technique_name']}")
                    
                    # Metrics for Severity, Confidence, and Similarity
                    m_col1, m_col2, m_col3 = st.columns(3)
                    
                    m_col1.metric("Severity", result['severity'])
                    m_col2.metric("Confidence", result['confidence'])
                    m_col3.metric("Similarity Score", f"{result['similarity_score']:.2f}")
                    
                    st.divider()
                    
                    # Reasoning
                    st.markdown("**Reasoning:**")
                    st.info(result['reasoning'])
                    
                    # Recommended Mitigation
                    st.markdown("**Recommended Mitigation:**")
                    if result['mitigation'] != "No mitigation available":
                        st.success(result['mitigation'])
                    else:
                        st.warning("No mitigation information available")
                    
                    # Expandable section for raw JSON
                    with st.expander("📄 View Raw JSON Response"):
                        st.json(result)

# ============================================
# FOOTER SECTION
# ============================================
st.divider()
st.caption("Powered by MITRE ATT&CK Framework • Prebuilt FAISS Index • HuggingFace API")
