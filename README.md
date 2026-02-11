# 🛡️ MITRE SOC Copilot

**Streamlit-based RAG System** for analyzing security logs and mapping them to MITRE ATT&CK techniques using AI-powered intelligence.

---

## 🎯 Features

- **Streamlit-Only Architecture**: No backend server required
- **HuggingFace API Integration**: Memory-efficient (no local models)
- **FAISS Vector Search**: Fast semantic similarity retrieval
- **Production-Ready**: Optimized for Streamlit Cloud free tier (<512MB)
- **Intelligent Scoring**: 
  - Similarity scores from FAISS
  - Rule-based severity classification
  - Confidence calibration

---

## 🏗️ Architecture

```
Streamlit App
    ↓
HuggingFace API Embeddings (sentence-transformers/all-MiniLM-L6-v2)
    ↓
FAISS Vector Store (built at startup)
    ↓
HuggingFace API LLM (mistralai/Mistral-7B-Instruct-v0.3)
    ↓
Structured JSON Output
```

---

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd mitre-intel-agent
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment

Create a `.env` file:

```bash
HUGGINGFACE_API_TOKEN="your_token_here"
```

Get your free HuggingFace API token at: https://huggingface.co/settings/tokens

### 4. Run Application

```bash
streamlit run streamlit_app.py
```

The app will open automatically at `http://localhost:8501`

---

## 📦 Project Structure

```
mitre-intel-agent/
├── streamlit_app.py              # Main application (all-in-one)
├── requirements.txt              # Python dependencies
├── .env                          # Environment variables (API token)
├── .gitignore                    # Git ignore rules
├── data/
│   └── mitre_clean_rag_ready.json # MITRE ATT&CK knowledge base
└── README.md                     # This file
```

---

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `HUGGINGFACE_API_TOKEN` | HuggingFace API token for embeddings and LLM | ✅ Yes |

### Models Used

- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (via HuggingFace API)
- **LLM**: `mistralai/Mistral-7B-Instruct-v0.3` (via HuggingFace API)

---

## 💡 Usage

1. **Enter a security log** in the text area
2. **Click "🔍 Analyze Log"**
3. **View results**:
   - Matched MITRE ATT&CK technique
   - Severity classification (High/Medium/Low)
   - Confidence score
   - Similarity score
   - AI-generated reasoning
   - Recommended mitigations

### Example Input

```
Failed login attempt from IP 192.168.1.100 user admin at 2024-01-15 10:30:45
```

### Example Output

```json
{
  "technique_id": "T1110.001",
  "technique_name": "Brute Force: Password Guessing",
  "similarity_score": 0.89,
  "severity": "High",
  "confidence": "High",
  "reasoning": "Multiple failed login attempts indicate password guessing attack",
  "mitigation": "Implement account lockout policies..."
}
```

---

## 🌐 Deploy to Streamlit Cloud

### Prerequisites
- GitHub account
- HuggingFace account with API token

### Steps

1. **Push to GitHub**:
```bash
git add .
git commit -m "Initial commit"
git push origin main
```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub repository
   - Set main file: `streamlit_app.py`
   - Add secret: `HUGGINGFACE_API_TOKEN = "your_token"`
   - Click "Deploy"

3. **Done!** Your app is live on Streamlit Cloud (free tier compatible)

---

## 🔒 Security

- ✅ No API keys hardcoded
- ✅ Environment variables via `.env`
- ✅ `.gitignore` configured properly
- ✅ Input validation implemented
- ✅ Safe error handling (no internal details exposed)

---

## 📊 Memory Optimization

This project is optimized for Streamlit Cloud free tier:

- **No local models**: All embeddings and LLM via HuggingFace API
- **Cached pipeline**: FAISS built once at startup with `@st.cache_resource`
- **Lightweight dependencies**: No PyTorch, no sentence-transformers locally
- **Memory footprint**: ~200-300MB (well under 512MB limit)

---

## 🛠️ Development

### Add New Features

All logic is in `streamlit_app.py`:
- Validation: `validate_input()`
- Severity: `classify_severity()`
- Confidence: `calibrate_confidence()`
- Analysis: `analyze_log()`

### Update MITRE Data

Replace `data/mitre_clean_rag_ready.json` with updated JSON and restart the app.

---

## 📝 License

MIT License - Feel free to use and modify for your projects.

---

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📧 Support

For issues or questions, please open a GitHub issue.

---

**Built with ❤️ using Streamlit, LangChain, and HuggingFace**
