# 🛡️ MITRE SOC Copilot

A **production-grade AI-powered RAG (Retrieval-Augmented Generation) system** that analyzes security logs and intelligently maps them to **MITRE ATT&CK techniques** with detailed explanations, severity ratings, and mitigation strategies.

This system combines modern AI/ML techniques with cybersecurity threat intelligence to assist Security Operations Center (SOC) analysts in rapid threat classification and response.

---

🚀 **Live App:**  
https://mitre-intel-agent-c5adaknlmmu9xdck7hdg73.streamlit.app

---

## 🎯 Features

- **🤖 AI-Powered Analysis**: Uses large language models (LLMs) for intelligent threat classification
- **🔍 Semantic Search**: FAISS vector database for fast similarity-based retrieval
- **📊 Multi-Dimensional Scoring**: 
  - Semantic similarity scores from vector search
  - Rule-based severity classification (High/Medium/Low)
  - Calibrated confidence levels
- **🎨 Interactive UI**: Streamlit-based web interface with real-time analysis
- **☁️ Cloud-Native**: Optimized for Streamlit Cloud deployment (<512MB memory)
- **🔒 Secure**: Environment-based API key management, input validation
- **⚡ Fast**: Pre-built FAISS index for instant retrieval
- **📚 MITRE ATT&CK Framework**: Comprehensive coverage of 200+ attack techniques

---

## 🏗️ System Architecture

### High-Level Flow

```
┌─────────────────┐
│   User Input    │  Security Log Text
│  (Streamlit UI) │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│              Text Embedding Generation                   │
│  (sentence-transformers/all-MiniLM-L6-v2 via HF API)    │
│              384-dimensional vector                      │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│           FAISS Vector Similarity Search                 │
│  Retrieves Top 3 Most Similar MITRE Techniques          │
│  Returns: Documents + Similarity Scores                  │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│              Prompt Construction                         │
│  Log + Top 3 Techniques → Structured Prompt             │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│         Large Language Model (LLM) Analysis              │
│     (Qwen/Qwen2.5-72B-Instruct via HF API)              │
│  Returns: Structured JSON with analysis                 │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│          Post-Processing & Enrichment                    │
│  • Severity Classification (Rule-based)                 │
│  • Confidence Calibration                                │
│  • Mitigation Strategy Retrieval                         │
└────────┬────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────┐
│  Results Display│  Technique ID, Reasoning, Mitigations
│  (Streamlit UI) │
└─────────────────┘
```

---

## 📚 Core Concepts Explained

### 1️⃣ **RAG (Retrieval-Augmented Generation)**

**What is RAG?**
RAG is a hybrid AI architecture that combines:
- **Retrieval**: Finding relevant information from a knowledge base
- **Generation**: Using an LLM to generate contextual responses

**Why RAG?**
- Pure LLMs can hallucinate or provide outdated information
- RAG grounds LLM responses in factual, up-to-date data
- More cost-effective than fine-tuning large models
- Allows for domain-specific knowledge injection

**In This Project:**
```
User Query → Retrieve MITRE techniques → LLM analyzes with context → Accurate classification
```

---

### 2️⃣ **Vector Embeddings**

**What are Embeddings?**
Embeddings are numerical representations of text in high-dimensional space where semantically similar texts are positioned closer together.

**Example:**
```
"Failed login attempt" → [0.23, -0.45, 0.67, ..., 0.12]  (384 dimensions)
"Brute force attack"  → [0.21, -0.43, 0.69, ..., 0.15]  (similar vector = close in meaning)
```

**Model Used:** `sentence-transformers/all-MiniLM-L6-v2`
- **Type**: Transformer-based encoder
- **Output**: 384-dimensional dense vectors
- **Trained on**: 1B+ sentence pairs for semantic similarity
- **Advantages**: Fast, lightweight, multilingual support

**Why This Model?**
- Excellent balance between speed and accuracy
- Small footprint (~80MB)
- Optimized for sentence-level semantic search

---

### 3️⃣ **FAISS (Facebook AI Similarity Search)**

**What is FAISS?**
FAISS is a library for efficient similarity search in high-dimensional vector spaces.

**How It Works:**
1. **Indexing**: Pre-compute embeddings for all MITRE techniques
2. **Storage**: Store vectors in optimized data structures (indexes)
3. **Search**: Find k-nearest neighbors using cosine similarity or L2 distance

**Search Algorithm:**
```python
# Simplified concept
query_vector = embed("Failed SSH login")
distances, indices = index.search(query_vector, k=3)
# Returns: Top 3 most similar technique vectors
```

**Distance Metric:**
This project uses **L2 distance**, converted to similarity score:
```
similarity_score = 1 / (1 + L2_distance)
```

**Advantages:**
- **Speed**: Searches millions of vectors in milliseconds
- **Scalability**: Handles billions of vectors with GPU support
- **Memory Efficient**: Compressed indexes reduce memory usage

---

### 4️⃣ **MITRE ATT&CK Framework**

**What is MITRE ATT&CK?**
A globally accessible knowledge base of adversary tactics and techniques based on real-world observations.

**Structure:**
```
Tactic (Why?) → Technique (What?) → Sub-Technique (How?)
Example:
Credential Access → Brute Force (T1110) → Password Guessing (T1110.001)
```

**In This Project:**
- **Knowledge Base**: 200+ techniques with descriptions, mitigations, platforms
- **Vector Store**: Each technique embedded for semantic search
- **Metadata**: Technique ID, name, description, mitigations, platforms

**Example Technique:**
```json
{
  "technique_id": "T1110.001",
  "name": "Brute Force: Password Guessing",
  "description": "Adversaries may use brute force techniques...",
  "mitigations": ["Account lockout policies", "MFA", "..."]
}
```

---

### 5️⃣ **LangChain Framework**

**What is LangChain?**
A framework for developing applications powered by language models.

**Components Used:**

1. **HuggingFaceEmbeddings**: Wrapper for embedding models
   ```python
   embeddings = HuggingFaceEmbeddings(model_name="...")
   ```

2. **FAISS VectorStore**: Vector database integration
   ```python
   vector_store = FAISS.from_documents(documents, embeddings)
   ```

3. **ChatHuggingFace**: LLM chat interface
   ```python
   llm = ChatHuggingFace(llm=endpoint)
   ```

4. **PromptTemplate**: Structured prompt creation
   ```python
   template = PromptTemplate(
       input_variables=["log_text", "technique_1", ...],
       template="You are a cybersecurity analyst..."
   )
   ```

**Benefits:**
- Standardized interfaces for different AI services
- Easy switching between models/providers
- Built-in memory and chain management

---

### 6️⃣ **Large Language Model (LLM)**

**Model Used:** `Qwen/Qwen2.5-72B-Instruct`

**Specifications:**
- **Parameters**: 72 billion
- **Type**: Instruction-tuned transformer
- **Context Window**: 32K tokens
- **Training**: Multilingual, instruction-following, reasoning

**Why This Model?**
- Strong reasoning capabilities for threat analysis
- Follows structured output format (JSON)
- Low temperature (0.0) for deterministic responses
- Balanced between accuracy and API cost

**Prompt Engineering:**
```python
"""You are a cybersecurity analyst. Analyze this security log...

Security Log: {log_text}

Top 3 Candidate Techniques:
1. {technique_1}
2. {technique_2}
3. {technique_3}

Respond with ONLY valid JSON...
"""
```

**Output Structure:**
```json
{
  "technique_id": "T1110.001",
  "technique_name": "Brute Force: Password Guessing",
  "reasoning": "Multiple failed logins indicate...",
  "confidence": "high"
}
```

---

### 7️⃣ **Severity Classification**

**Rule-Based System:**

```python
def classify_severity(technique_name, technique_id):
    if any(["escalation", "credential", "lateral", ...]):
        return "High"
    elif any(["reconnaissance"]):
        return "Low"
    else:
        return "Medium"
```

**Severity Levels:**
- **High**: Privilege escalation, credential theft, lateral movement, C2, impact
- **Medium**: General execution, persistence, defense evasion
- **Low**: Reconnaissance, discovery

**Special Cases:**
- T1110.* (Brute Force) → Always High
- Based on keywords in technique name

---

### 8️⃣ **Confidence Calibration**

**Hybrid Approach:**
Combines vector similarity with LLM confidence:

```python
def calibrate_confidence(similarity_score, llm_confidence):
    if similarity_score > 0.85 and llm_confidence == "High":
        return "High"
    elif similarity_score >= 0.70:
        return "Medium"
    else:
        return "Low"
```

**Rationale:**
- High similarity + High LLM confidence = Very reliable
- Moderate similarity = Temper confidence
- Low similarity = Low confidence (potential edge case)

---

### 9️⃣ **Streamlit Framework**

**What is Streamlit?**
A Python framework for building interactive web apps with minimal code.

**Key Features Used:**

1. **Caching** (`@st.cache_resource`):
   ```python
   @st.cache_resource
   def load_rag_pipeline():
       # Loaded once, reused across sessions
   ```

2. **UI Components**:
   - `st.text_area`: Log input
   - `st.button`: Trigger analysis
   - `st.metric`: Display scores
   - `st.spinner`: Loading indicators
   - `st.expander`: Collapsible sections

3. **Session Management**: Maintains state across interactions

**Why Streamlit?**
- Rapid prototyping (100+ lines → full app)
- No HTML/CSS/JavaScript required
- Built-in deployment to Streamlit Cloud
- Perfect for ML/AI demos

---

### 🔟 **Input Validation & Security**

**Validation Rules:**
```python
def validate_input(log_text):
    if len(log_text) < 20:
        return False, "Input too short"
    if len(log_text.split()) < 3:
        return False, "Need at least 3 words"
    if not any(ch.isalpha() for ch in log_text):
        return False, "Need alphabetic characters"
    return True, ""
```

**Security Measures:**
- Environment variables for API keys (`.env`)
- Input sanitization before LLM processing
- Safe JSON parsing with error handling
- No credential exposure in logs
- `.gitignore` configured for sensitive files

---

## 🚀 Quick Start

### Prerequisites
- **Python**: 3.12+ (or 3.8+)
- **HuggingFace Account**: For API token
- **Git**: For cloning repository

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd mitre-intel-agent
```

### 2. Set Up Python Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate
```

### 3. Install Dependencies

**For Development:**
```bash
pip install -r requirements-dev.txt
```

**For Production (Streamlit Cloud):**
```bash
pip install -r requirements.txt
```

**Dependency Overview:**
- `streamlit`: Web UI framework
- `langchain`, `langchain-community`, `langchain-huggingface`: RAG framework
- `faiss-cpu`: Vector similarity search
- `sentence-transformers`: Embedding models
- `python-dotenv`: Environment variable management
- `huggingface-hub`: HuggingFace API client

### 4. Set Up Environment

Create a `.env` file in the project root:

```bash
HUGGINGFACE_API_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"
```

**Get Your Free Token:**
1. Sign up at [HuggingFace](https://huggingface.co/join)
2. Go to [Settings > Access Tokens](https://huggingface.co/settings/tokens)
3. Create a new token (read access is sufficient)
4. Copy and paste into `.env`

### 5. Build Vector Store (One-Time Setup)

```bash
python build_index.py
```

**What This Does:**
- Loads MITRE ATT&CK data from `data/mitre_clean_rag_ready.json`
- Generates embeddings for 200+ techniques
- Builds FAISS index
- Saves to `vector_store/` directory

**Expected Output:**
```
============================================================
MITRE ATT&CK Vector Store Builder
============================================================

[1/4] Loading MITRE ATT&CK data...
✅ Loaded 200+ MITRE techniques

[2/4] Creating LangChain documents...
✅ Created 200+ documents

[3/4] Initializing embedding model...
✅ Embedding model loaded

[4/4] Building FAISS vector store...
✅ Vector store saved to: vector_store/

Generated files:
  - vector_store/index.faiss (X KB)
  - vector_store/index.pkl (Y KB)
```

**Note:** This step takes 2-5 minutes on first run. The vector store is reusable and doesn't need rebuilding unless the MITRE data changes.

### 6. Run Application

```bash
streamlit run streamlit_app.py
```

The app will automatically open at `http://localhost:8501`

**Troubleshooting:**
- **Port already in use**: `streamlit run streamlit_app.py --server.port 8502`
- **API errors**: Verify your HuggingFace token is correct
- **Vector store error**: Ensure `build_index.py` completed successfully

---

## 📦 Project Structure

```
mitre-intel-agent/
│
├── 📄 streamlit_app.py              # Main application (UI + RAG logic)
│   ├── Configuration (API keys, env setup)
│   ├── RAG pipeline loading (cached)
│   ├── Helper functions
│   │   ├── validate_input()
│   │   ├── classify_severity()
│   │   └── calibrate_confidence()
│   ├── analyze_log() (core analysis)
│   └── UI components (Streamlit)
│
├── 📄 build_index.py                # Vector store builder (run once)
│   ├── Load MITRE data
│   ├── Create LangChain documents
│   ├── Initialize embeddings
│   └── Build & save FAISS index
│
├── 📄 requirements.txt              # Production dependencies
├── 📄 requirements-dev.txt          # Development dependencies (includes torch)
├── 📄 pyproject.toml                # Project metadata
├── 📄 .env                          # Environment variables (API token)
├── 📄 .gitignore                    # Git ignore rules
├── 📄 test_quick.py                 # Quick testing script
│
├── 📁 data/
│   └── mitre_clean_rag_ready.json   # MITRE ATT&CK knowledge base (200+ techniques)
│
├── 📁 vector_store/                 # Generated by build_index.py
│   ├── index.faiss                  # FAISS vector index
│   ├── index.pkl                    # Document metadata
│   └── mitre.index                  # Additional index files
│
└── 📁 __pycache__/                  # Python cache (auto-generated)
```

### Key Files Explained

#### `streamlit_app.py` (Main Application)
**Lines 1-40**: Configuration and imports
- Sets `TOKENIZERS_PARALLELISM=false` to avoid warnings
- Imports LangChain, Streamlit, FAISS components

**Lines 42-105**: `load_rag_pipeline()` - Cached pipeline loader
- Loads pre-built FAISS vector store from disk
- Initializes embedding model (for query-time encoding)
- Configures LLM (Qwen 2.5 72B)
- Creates prompt template

**Lines 107-150**: Helper functions
- `validate_input()`: Basic input sanitization
- `classify_severity()`: Rule-based severity scoring
- `calibrate_confidence()`: Hybrid confidence calculation

**Lines 152-280**: `analyze_log()` - Core analysis function
1. Vector search (top 3 similar techniques)
2. Prompt formatting
3. LLM invocation
4. JSON extraction and validation
5. Mitigation retrieval
6. Scoring enhancements

**Lines 282-364**: Streamlit UI
- Page configuration
- Input section (text area)
- Analysis trigger (button)
- Results display (metrics, reasoning, mitigations)

#### `build_index.py` (Vector Store Builder)
**Purpose**: One-time script to create FAISS index

**Process:**
1. Load MITRE data (JSON)
2. Create LangChain `Document` objects
3. Initialize embedding model (local)
4. Generate embeddings for all techniques
5. Build FAISS index
6. Save to `vector_store/` directory

**When to Run:**
- Initial setup
- After updating MITRE data
- If vector store is corrupted

#### `data/mitre_clean_rag_ready.json`
**Format:**
```json
[
  {
    "technique_id": "T1110.001",
    "name": "Brute Force: Password Guessing",
    "description": "Adversaries may use brute force...",
    "platforms": ["Windows", "Linux", "macOS"],
    "mitigations": ["Account lockout policies..."]
  },
  ...
]
```

**Source**: Pre-processed MITRE ATT&CK data
**Count**: 200+ techniques across all tactics

---

## � Usage Guide

### Basic Workflow

1. **Launch the Application**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Enter a Security Log**
   - Paste a security event, alert, or log message
   - Minimum 20 characters, 3 words
   - Should contain actionable security information

3. **Click "🔍 Analyze Log"**
   - System performs vector search
   - LLM analyzes top candidates
   - Results displayed in ~2-5 seconds

4. **Review Results**
   - **Technique ID & Name**: Matched MITRE technique
   - **Severity**: High/Medium/Low classification
   - **Confidence**: High/Medium/Low (calibrated)
   - **Similarity Score**: Vector similarity (0-1)
   - **Reasoning**: LLM explanation
   - **Mitigation**: Recommended countermeasures

### Example Inputs & Expected Outputs

#### Example 1: Brute Force Attack
**Input:**
```
Multiple failed SSH login attempts detected from 203.0.113.47 
for user 'root' within 60 seconds. Authentication logs show 
15 failed attempts using different passwords.
```

**Expected Output:**
```json
{
  "technique_id": "T1110.001",
  "technique_name": "Brute Force: Password Guessing",
  "similarity_score": 0.91,
  "severity": "High",
  "confidence": "High",
  "reasoning": "Multiple failed login attempts with different passwords indicate a password guessing attack targeting the root account.",
  "mitigation": "Implement account lockout policies after failed attempts. Use multi-factor authentication. Monitor and alert on repeated failed logins..."
}
```

#### Example 2: Scheduled Task Persistence
**Input:**
```
Windows Event 4698: A scheduled task was created. 
Task Name: \Microsoft\Windows\UpdateOrchestrator\Backup
Creator: SYSTEM
Action: C:\Windows\Temp\update.exe
```

**Expected Output:**
```json
{
  "technique_id": "T1053.005",
  "technique_name": "Scheduled Task",
  "similarity_score": 0.87,
  "severity": "Medium",
  "confidence": "High",
  "reasoning": "Creation of a scheduled task pointing to an executable in a temporary directory is consistent with persistence techniques.",
  "mitigation": "Audit scheduled tasks regularly. Restrict task creation privileges. Monitor for suspicious task names and paths..."
}
```

#### Example 3: Process Injection
**Input:**
```
Security alert: Process C:\Windows\System32\svchost.exe 
attempted to inject code into explorer.exe using 
WriteProcessMemory and CreateRemoteThread APIs
```

**Expected Output:**
```json
{
  "technique_id": "T1055",
  "technique_name": "Process Injection",
  "similarity_score": 0.93,
  "severity": "High",
  "confidence": "High",
  "reasoning": "Use of WriteProcessMemory and CreateRemoteThread APIs is a classic indicator of process injection techniques.",
  "mitigation": "Enable Endpoint Detection and Response (EDR) solutions. Monitor API call patterns. Implement behavior-based detection..."
}
```

### Input Best Practices

✅ **Good Inputs:**
- Specific event details (IPs, usernames, ports)
- Contextual information (timestamps, frequency)
- Technical indicators (API calls, registry keys, file paths)
- Clear action descriptions

❌ **Poor Inputs:**
- Too generic ("suspicious activity detected")
- Too short ("failed login")
- Non-security related ("server is slow")
- No context or details

---

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
# Required
HUGGINGFACE_API_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"

# Optional (for advanced users)
# TOKENIZERS_PARALLELISM="false"  # Already set in code
```

### Model Configuration

To change models, edit `streamlit_app.py`:

**Embedding Model** (Line ~65):
```python
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",  # Change here
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
```

**Alternative Embedding Models:**
- `sentence-transformers/all-mpnet-base-v2` (Higher quality, slower)
- `sentence-transformers/paraphrase-MiniLM-L3-v2` (Faster, smaller)
- `BAAI/bge-small-en-v1.5` (Optimized for retrieval)

**LLM Model** (Line ~77):
```python
llm_endpoint = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-72B-Instruct",  # Change here
    temperature=0.0,  # 0.0 = deterministic, 1.0 = creative
    max_new_tokens=500,  # Max response length
    huggingfacehub_api_token=api_token
)
```

**Alternative LLM Models:**
- `mistralai/Mistral-7B-Instruct-v0.3` (Lighter, faster)
- `meta-llama/Llama-3.1-8B-Instruct` (Good balance)
- `google/gemma-2-9b-it` (Google's instruction-tuned model)

**Note:** Some models require additional permissions or paid API access.

### Advanced Configuration

#### Adjust Retrieval Count
Change the number of retrieved techniques (Line ~165):
```python
docs_and_scores = vector_store.similarity_search_with_score(log_text, k=3)  # Change k
```

#### Modify Similarity Scoring
Adjust the similarity calculation (Line ~173):
```python
score = round(1 / (1 + distance), 2)  # Change formula
```

#### Customize Severity Rules
Edit `classify_severity()` function (Line ~120):
```python
high_risk_keywords = [
    "escalation", "credential", "lateral", "command and control",
    "your_custom_keyword"  # Add your own
]
```

---

## 🌐 Deployment

### Deploy to Streamlit Cloud (Free Tier)

Streamlit Cloud offers free hosting for public applications with:
- 1 GB RAM
- 1 CPU core
- Unlimited apps (public repositories)

#### Prerequisites
- GitHub account
- HuggingFace API token
- Git repository with your code

#### Deployment Steps

**1. Prepare Repository**

Ensure these files are committed:
```bash
git add .
git commit -m "Prepare for Streamlit Cloud deployment"
git push origin main
```

**Required files:**
- `streamlit_app.py`
- `requirements.txt`
- `data/mitre_clean_rag_ready.json`
- `vector_store/` (entire directory)

**2. Deploy on Streamlit Cloud**

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click "New app"
3. Connect your GitHub repository
4. Configure deployment:
   - **Main file path**: `streamlit_app.py`
   - **Python version**: 3.12 (or 3.8+)
   - **Branch**: `main`

5. Add secrets (click "Advanced settings" → "Secrets"):
   ```toml
   HUGGINGFACE_API_TOKEN = "hf_xxxxxxxxxxxxxxxxxxxx"
   ```

6. Click "Deploy"

**3. Monitor Deployment**

- Initial deployment: 5-10 minutes
- Watch logs for errors
- App will auto-update on new commits

**Common Deployment Issues:**

| Issue | Solution |
|-------|----------|
| Memory limit exceeded | Ensure `vector_store/` is pre-built and committed |
| Module not found | Check `requirements.txt` includes all dependencies |
| API token error | Verify secret is named `HUGGINGFACE_API_TOKEN` |
| FAISS error | Ensure `faiss-cpu` in `requirements.txt` (not `faiss-gpu`) |

---

### Deploy to Other Platforms

#### Hugging Face Spaces

1. Create a new Space: [huggingface.co/new-space](https://huggingface.co/new-space)
2. Select **Streamlit** as SDK
3. Upload files via Git LFS (for large `vector_store/` files)
4. Add `HUGGINGFACE_API_TOKEN` in Space settings → Repository secrets

#### Docker Deployment

Create a `Dockerfile`:
```dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run application
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:
```bash
docker build -t mitre-agent .
docker run -p 8501:8501 --env-file .env mitre-agent
```

#### AWS/GCP/Azure

Use container services:
- **AWS**: Elastic Container Service (ECS) or App Runner
- **GCP**: Cloud Run
- **Azure**: Container Instances or App Service

---

## 💻 Development

### Running Tests

```bash
# Quick test script
python test_quick.py
```

### Adding New MITRE Techniques

1. Update `data/mitre_clean_rag_ready.json`:
   ```json
   {
     "technique_id": "T1234.567",
     "name": "New Technique",
     "description": "...",
     "platforms": ["Windows"],
     "mitigations": ["..."]
   }
   ```

2. Rebuild vector store:
   ```bash
   python build_index.py
   ```

3. Restart application

### Customizing Severity Rules

Edit `classify_severity()` in [streamlit_app.py](streamlit_app.py#L120):

```python
def classify_severity(technique_name, technique_id):
    technique_name_lower = technique_name.lower()
    
    # Add your custom rules
    if "ransomware" in technique_name_lower:
        return "Critical"  # New severity level
    
    # Existing rules...
```

### Debugging

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

View Streamlit debug info:
```bash
streamlit run streamlit_app.py --logger.level=debug
```

---

## 📊 Performance & Optimization

### Memory Usage

**Components:**
- FAISS Index: ~5-10 MB
- Embedding Model (API): 0 MB (remote)
- LLM (API): 0 MB (remote)
- Streamlit App: ~50-100 MB

**Total**: ~100-150 MB (well under 512MB Streamlit Cloud limit)

### Speed Benchmarks

**Average Analysis Time:**
- Vector search: 10-50ms
- Embedding generation (API): 200-500ms
- LLM inference (API): 1-3 seconds
- **Total**: ~2-5 seconds per query

**Optimization Tips:**
1. **Caching**: Use `@st.cache_resource` for pipeline loading
2. **Batch Queries**: Process multiple logs in parallel (future enhancement)
3. **Local Embeddings**: Use local model for faster query encoding (trades memory for speed)
4. **Index Optimization**: Use FAISS IVF indexes for >10K documents

### Scaling Considerations

**Current Capacity:**
- 200+ MITRE techniques
- Single-query processing
- Suitable for: SOC analysts, demos, prototypes

**To Scale Up:**
- Use PostgreSQL + pgvector for distributed storage
- Implement batch processing for log ingestion
- Add API layer (FastAPI) for multi-user access
- Use Redis for caching frequent queries
- Deploy LLM on dedicated infrastructure (AWS Bedrock, Azure OpenAI)

---

## 🔒 Security & Privacy

### Data Protection

✅ **Implemented:**
- Environment-based API key management
- No hardcoded credentials
- Input sanitization and validation
- Error messages don't expose internal details
- `.gitignore` prevents committing sensitive files

⚠️ **Considerations:**
- User inputs are sent to HuggingFace API (read their privacy policy)
- Logs are not stored persistently (ephemeral)
- No user authentication (suitable for internal use only)

### Adding Authentication

For production deployment with sensitive data, add authentication:

**Option 1: Streamlit Cloud (Built-in)**
```python
import streamlit_authenticator as stauth

authenticator = stauth.Authenticate(
    credentials, cookie_name, key, cookie_expiry_days
)
name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    # Show app
else:
    st.error("Unauthorized")
```

**Option 2: Nginx Reverse Proxy (Basic Auth)**
```nginx
location / {
    auth_basic "Restricted";
    auth_basic_user_file /etc/nginx/.htpasswd;
    proxy_pass http://localhost:8501;
}
```

### Compliance

**For Regulated Environments:**
- Use on-premises LLM (e.g., LLaMA via Ollama)
- Replace HuggingFace API with Azure OpenAI (SOC 2, ISO 27001 certified)
- Implement audit logging (log all queries and results)
- Add data retention policies

---

## 🧠 Technical Deep Dive

### RAG Pipeline Architecture

```python
# Simplified pipeline flow
def rag_pipeline(user_query):
    # Step 1: Embed query
    query_vector = embedding_model.encode(user_query)
    
    # Step 2: Retrieve similar documents
    docs = vector_store.similarity_search(query_vector, k=3)
    
    # Step 3: Construct prompt with context
    context = "\n".join([doc.content for doc in docs])
    prompt = f"Context: {context}\n\nQuery: {user_query}\n\nAnswer:"
    
    # Step 4: Generate response
    response = llm.generate(prompt)
    
    return response
```

### Vector Similarity Mathematics

**Cosine Similarity** (conceptual, FAISS uses L2):
```
similarity(A, B) = (A · B) / (||A|| × ||B||)

Where:
- A · B = dot product of vectors
- ||A|| = magnitude of vector A
- Result ranges from -1 (opposite) to 1 (identical)
```

**L2 Distance** (used in this project):
```
L2_distance = sqrt(sum((A_i - B_i)^2))

Converted to similarity:
similarity = 1 / (1 + L2_distance)
```

**Why L2 instead of Cosine?**
- FAISS default for most index types
- Faster computation
- Similar results for normalized vectors
- Better for hierarchical indexes

### Prompt Engineering Breakdown

**Our Prompt Structure:**
```
System Role → Task Description → Context → Output Format
```

**Key Techniques:**
1. **Role Assignment**: "You are a cybersecurity analyst"
2. **Structured Input**: Numbered list of candidates
3. **Output Constraints**: "ONLY valid JSON"
4. **Few-Shot Learning**: Implicit (through training data)
5. **Temperature Control**: 0.0 for deterministic output

**Why This Works:**
- Clear role → model adopts expert persona
- Structured format → easier parsing
- Explicit constraints → reduces hallucination
- Low temperature → consistent results

### FAISS Index Types

**Current**: Flat (Brute-Force)
- **Pros**: Exact search, simple, fast for <100K vectors
- **Cons**: Linear time complexity O(n)

**Alternatives for Scaling:**

1. **IndexIVFFlat** (Inverted File Index)
   ```python
   quantizer = faiss.IndexFlatL2(d)
   index = faiss.IndexIVFFlat(quantizer, d, nlist)
   # nlist = number of clusters
   ```
   - **Speed**: ~10x faster for >10K vectors
   - **Accuracy**: ~95-99% recall

2. **IndexHNSW** (Hierarchical Navigable Small World)
   ```python
   index = faiss.IndexHNSWFlat(d, M)
   # M = number of connections per layer
   ```
   - **Speed**: ~100x faster for >1M vectors
   - **Accuracy**: >99% recall
   - **Memory**: Higher overhead

3. **Product Quantization** (for massive scale)
   ```python
   index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
   ```
   - **Compression**: 10-100x smaller indexes
   - **Trade-off**: Slight accuracy loss

### Embedding Model Comparison

| Model | Dimensions | SBERT Score | Speed | Size |
|-------|-----------|-------------|-------|------|
| **all-MiniLM-L6-v2** ⭐ | 384 | 68.06 | Fast | 80MB |
| all-mpnet-base-v2 | 768 | 69.57 | Medium | 420MB |
| bge-small-en-v1.5 | 384 | 62.17 | Fast | 133MB |
| bge-large-en-v1.5 | 1024 | 67.34 | Slow | 1.34GB |

**SBERT Score**: Semantic Textual Similarity Benchmark

**Our Choice**: all-MiniLM-L6-v2 (best speed/accuracy trade-off)

### LLM Model Comparison

| Model | Parameters | Context | Speed | Quality |
|-------|-----------|---------|-------|---------|
| Mistral-7B | 7B | 32K | Fast | Good |
| **Qwen2.5-72B** ⭐ | 72B | 32K | Medium | Excellent |
| Llama-3.1-8B | 8B | 128K | Fast | Very Good |
| Gemma-2-9B | 9B | 8K | Fast | Good |

**Our Choice**: Qwen2.5-72B (best reasoning for complex analysis)

### Confidence Calibration Logic

**Problem**: LLMs can be overconfident or underconfident.

**Solution**: Hybrid scoring
```python
if similarity_score > 0.85 and llm_confidence == "High":
    final_confidence = "High"  # Both agree
elif similarity_score >= 0.70:
    final_confidence = "Medium"  # Moderate match
else:
    final_confidence = "Low"  # Poor match
```

**Benefits:**
- Prevents false positives (LLM says "High" but poor match)
- Increases reliability (two independent signals)
- Provides actionable confidence to analysts

---

## 🐛 Troubleshooting

### Common Issues

#### 1. `ModuleNotFoundError: No module named 'faiss'`

**Cause**: FAISS not installed

**Solution**:
```bash
pip install faiss-cpu
```

**Note**: Use `faiss-cpu` (not `faiss` or `faiss-gpu` unless you have CUDA)

---

#### 2. `Vector store not found at vector_store`

**Cause**: FAISS index not built

**Solution**:
```bash
python build_index.py
```

Make sure `data/mitre_clean_rag_ready.json` exists.

---

#### 3. `HUGGINGFACE_API_TOKEN not found`

**Cause**: `.env` file missing or incorrect

**Solution**:
1. Create `.env` in project root
2. Add: `HUGGINGFACE_API_TOKEN="hf_..."`
3. Restart application

**Verify token:**
```bash
python -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('HUGGINGFACE_API_TOKEN'))"
```

---

#### 4. `Rate limit exceeded` (HuggingFace API)

**Cause**: Free tier has rate limits

**Solutions**:
- Wait 1 minute and retry
- Upgrade to Pro account ($9/month)
- Use local models (Ollama + LLaMA)

---

#### 5. `JSON decode error` (LLM response)

**Cause**: LLM returned invalid JSON

**Why**: Low-quality models or high temperature

**Solutions**:
- Use better model (Qwen, GPT-4, Claude)
- Set `temperature=0.0`
- Add more explicit JSON formatting to prompt

---

#### 6. Memory error on Streamlit Cloud

**Cause**: Vector store too large or dependencies too heavy

**Solutions**:
1. Ensure `vector_store/` is pre-built (commit it)
2. Check `requirements.txt` doesn't include `torch` or heavy packages
3. Use `sentence-transformers` without local models

---

#### 7. Slow performance (>10 seconds per query)

**Causes**:
- API latency (HuggingFace)
- Large FAISS index
- Network issues

**Solutions**:
- Use regional API endpoints (if available)
- Switch to smaller LLM (Mistral-7B)
- Implement local embedding model
- Add response caching

---

### Debug Mode

Enable detailed logging:

**In `streamlit_app.py`**, add at top:
```python
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
```

**Run with debug output:**
```bash
streamlit run streamlit_app.py --logger.level=debug 2>&1 | tee debug.log
```

---

## ❓ FAQ

### Q1: Can I use this offline?
**A**: Partially. You need internet for HuggingFace API calls. For fully offline:
- Use local LLM (Ollama + LLaMA)
- Use local embedding model (already included in `sentence-transformers`)

### Q2: How accurate is the technique classification?
**A**: Depends on:
- Log quality (specific vs. vague)
- MITRE data coverage
- LLM understanding
- **Estimated accuracy**: 70-85% for well-described events

### Q3: Can I add custom attack techniques?
**A**: Yes! Edit `data/mitre_clean_rag_ready.json`, add your technique, and rebuild index.

### Q4: Is this production-ready?
**A**: It's a **production-grade prototype**, suitable for:
- Internal SOC tools
- Analyst training
- Demo/POC environments

**For mission-critical use:**
- Add authentication
- Use enterprise LLM (Azure OpenAI)
- Implement audit logging
- Add human-in-the-loop verification

### Q5: What's the cost?
**A**: 
- **HuggingFace Free Tier**: ~1000 requests/month (sufficient for testing)
- **HuggingFace Pro**: $9/month (unlimited API)
- **Streamlit Cloud**: Free for public apps

**Estimated cost for 1000 queries/month**: $0-9

### Q6: Can it analyze multiple logs at once?
**A**: Current version processes one log at a time. Batch processing can be added (see Development section).

### Q7: How often should I update MITRE data?
**A**: MITRE ATT&CK is updated quarterly. Check [attack.mitre.org](https://attack.mitre.org) for releases.

### Q8: Can I use GPT-4 or Claude instead?
**A**: Yes! Replace `HuggingFaceEndpoint` with:
- `ChatOpenAI` (from `langchain_openai`)
- `ChatAnthropic` (from `langchain_anthropic`)

### Q9: What's the difference between similarity score and confidence?
**A**:
- **Similarity**: How closely the log matches the technique (vector distance)
- **Confidence**: LLM's certainty in its classification (calibrated with similarity)

### Q10: Can this replace human analysts?
**A**: No. This is an **augmentation tool** to:
- Speed up initial triage
- Suggest techniques to investigate
- Provide mitigation starting points

**Always** have human analysts review and validate results.

---

## 🎓 Learning Resources

### RAG & Vector Databases
- [LangChain Documentation](https://python.langchain.com/docs/get_started/introduction)
- [FAISS GitHub](https://github.com/facebookresearch/faiss)
- [Vector Databases Explained](https://www.pinecone.io/learn/vector-database/)

### LLMs & Embeddings
- [HuggingFace Course](https://huggingface.co/learn/nlp-course/)
- [Sentence Transformers](https://www.sbert.net/)
- [Prompt Engineering Guide](https://www.promptingguide.ai/)

### MITRE ATT&CK
- [MITRE ATT&CK Framework](https://attack.mitre.org/)
- [ATT&CK Navigator](https://mitre-attack.github.io/attack-navigator/)
- [Getting Started with ATT&CK](https://attack.mitre.org/resources/getting-started/)

### Streamlit Development
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Streamlit Cheat Sheet](https://cheat-sheet.streamlit.app/)
- [Streamlit Gallery](https://streamlit.io/gallery)

---

## 🤝 Contributing

Contributions are welcome! Here's how to contribute:

### Reporting Issues
1. Check [existing issues](https://github.com/your-repo/issues)
2. Provide detailed description, error logs, and steps to reproduce
3. Include system info (OS, Python version, package versions)

### Submitting Pull Requests

**Fork → Branch → Commit → Test → PR**

1. **Fork the repository**
2. **Create a feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make changes** and commit:
   ```bash
   git commit -m "Add: description of changes"
   ```
4. **Run tests** (if applicable):
   ```bash
   python test_quick.py
   ```
5. **Push and create PR**:
   ```bash
   git push origin feature/your-feature-name
   ```

### Development Guidelines
- Follow PEP 8 style guide
- Add docstrings to functions
- Update README if adding features
- Test on Windows, macOS, and Linux (if possible)

### Ideas for Contributions
- [ ] Batch log processing
- [ ] Export results to JSON/CSV
- [ ] Integration with SIEM systems (Splunk, Elastic)
- [ ] Custom MITRE data sources (e.g., CAR analytics)
- [ ] Multi-language support (non-English logs)
- [ ] API endpoint (FastAPI wrapper)
- [ ] Dark mode UI
- [ ] Historical analysis dashboard

---

## 📝 License

**MIT License**

Copyright (c) 2026 [Vanshdeep Singh Lamba ]

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

---



## 🙏 Acknowledgments

This project leverages amazing open-source tools and frameworks:

- **MITRE Corporation** - ATT&CK Framework
- **Facebook AI** - FAISS vector search
- **HuggingFace** - Model hosting and APIs
- **LangChain** - RAG framework
- **Streamlit** - Web UI framework
- **Sentence Transformers** - Embedding models
- **Qwen Team** - Qwen2.5 LLM

Special thanks to the cybersecurity and AI/ML communities for sharing knowledge and tools.

---

## 📈 Project Stats & Milestones

- **Version**: 1.0.0
- **Release Date**: 12th February 2026
- **MITRE Techniques**: 200+
- **Supported Platforms**: Windows, Linux, macOS, Android, iOS, Network
- **Languages**: Python
- **Code Lines**: ~600
- **Response Time**: ~2-5 seconds
- **Memory Footprint**: ~100-150 MB

---


<div align="center">

**⭐ Star this repo if you find it useful! ⭐**

**Built with ❤️ by the Cybersecurity & AI Community**

[🏠 Home](README.md) • [📖 Docs](#) • [🐛 Issues](../../issues) • [💬 Discussions](../../discussions)

</div>
