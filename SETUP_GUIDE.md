# 🚀 Production Setup Guide

## Architecture Overview

This project uses a **prebuilt FAISS vector store** for production deployment:

```
┌─────────────────────────────────────────────┐
│ OFFLINE (One-Time Local Build)             │
│  build_index.py                             │
│  ├─ Load MITRE data                         │
│  ├─ Generate embeddings (sentence-trans)   │
│  ├─ Build FAISS index                       │
│  └─ Save to vector_store/                   │
└─────────────────────────────────────────────┘
                    │
                    ▼
         vector_store/ (commit to repo)
                    │
                    ▼
┌─────────────────────────────────────────────┐
│ PRODUCTION (Streamlit Cloud)               │
│  streamlit_app.py                           │
│  ├─ Load prebuilt FAISS (no embeddings)    │
│  ├─ Use HuggingFace API for LLM only       │
│  └─ Fast startup (< 2 seconds)             │
└─────────────────────────────────────────────┘
```

---

## Step 1: Build Vector Store Locally

### Install Development Dependencies

```bash
pip install -r requirements-dev.txt
```

This includes:
- `torch` (for sentence-transformers)
- `sentence-transformers` (for embeddings)
- All LangChain dependencies

### Run Build Script

```bash
python build_index.py
```

**Expected output:**
```
============================================================
MITRE ATT&CK Vector Store Builder
============================================================

[1/4] Loading MITRE ATT&CK data...
✅ Loaded 691 MITRE techniques

[2/4] Creating LangChain documents...
✅ Created 691 documents

[3/4] Initializing embedding model...
⏳ Downloading sentence-transformers/all-MiniLM-L6-v2...
✅ Embedding model loaded

[4/4] Building FAISS vector store...
⏳ Generating embeddings for all documents...
✅ Vector store saved to: vector_store/

============================================================
✅ SUCCESS!
============================================================

Generated files:
  - vector_store/index.faiss (XXX KB)
  - vector_store/index.pkl (XXX KB)
```

### Verify Vector Store

```bash
# Check files exist
ls vector_store/

# Should show:
# index.faiss
# index.pkl
```

---

## Step 2: Commit Vector Store to Repository

The `vector_store/` directory **MUST** be committed to your repo for production deployment.

```bash
# Add vector store
git add vector_store/

# Commit
git commit -m "Add prebuilt FAISS vector store"

# Push to GitHub
git push origin main
```

**Important:** The `.gitignore` has been updated to **NOT ignore** `vector_store/`

---

## Step 3: Deploy to Streamlit Cloud

### Prerequisites
- GitHub repository with code + vector_store/
- HuggingFace API token

### Deployment Steps

1. **Go to:** https://share.streamlit.io

2. **Click:** "New app"

3. **Configure:**
   - Repository: `YOUR_USERNAME/mitre-intel-agent`
   - Branch: `main`
   - Main file: `streamlit_app.py`

4. **Add Secret:**
   ```toml
   HUGGINGFACE_API_TOKEN = "hf_your_token_here"
   ```

5. **Click:** "Deploy"

6. **Wait:** ~2-3 minutes

7. **Done!** App is live at: `https://your-app.streamlit.app`

---

## Step 4: Test Production App

### Startup Time
- **Expected:** < 2 seconds (loading prebuilt FAISS)
- **No embedding generation** at startup
- **No torch** in production

### Test Log
```
Failed login attempt from IP 192.168.1.100 user admin at 2024-01-15 10:30:45
```

### Expected Response
```json
{
  "technique_id": "T1110.001",
  "technique_name": "Brute Force: Password Guessing",
  "similarity_score": 0.89,
  "severity": "High",
  "confidence": "High",
  "reasoning": "...",
  "mitigation": "..."
}
```

---

## Production Requirements

### Memory Usage
- **FAISS index:** ~2-3 MB
- **App runtime:** ~150-200 MB
- **Total:** < 512 MB ✅ (Streamlit Cloud free tier compatible)

### Dependencies (production)
```
python-dotenv
streamlit
langchain
langchain-community
langchain-huggingface
huggingface-hub
faiss-cpu
```

**NOT included in production:**
- ❌ torch
- ❌ sentence-transformers

---

## Troubleshooting

### ❌ "Vector store not found"
**Solution:** Run `python build_index.py` and commit `vector_store/` to repo

### ❌ "build_index.py fails"
**Solution:** Install dev dependencies: `pip install -r requirements-dev.txt`

### ❌ "Streamlit app crashes on startup"
**Solution:** Check `vector_store/` is committed to GitHub repo

### ❌ "LLM inference fails"
**Solution:** Verify `HUGGINGFACE_API_TOKEN` is set in Streamlit Cloud secrets

---

## File Structure

```
mitre-intel-agent/
├── build_index.py              ← Run once locally
├── streamlit_app.py            ← Production app
├── requirements.txt            ← Production deps (no torch)
├── requirements-dev.txt        ← Dev deps (includes torch)
├── .env                        ← HuggingFace API token
├── .gitignore                  ← Updated (vector_store NOT ignored)
├── data/
│   └── mitre_clean_rag_ready.json
└── vector_store/              ← COMMIT THIS TO REPO
    ├── index.faiss            ← Prebuilt FAISS index
    └── index.pkl              ← Metadata
```

---

## Updating MITRE Data

If you update `data/mitre_clean_rag_ready.json`:

1. Run locally:
   ```bash
   python build_index.py
   ```

2. Commit updated vector store:
   ```bash
   git add vector_store/
   git commit -m "Update MITRE data"
   git push origin main
   ```

3. Streamlit Cloud will auto-redeploy

---

## Performance Comparison

| Metric | Old (API Embeddings) | New (Prebuilt) |
|--------|---------------------|----------------|
| **Startup Time** | ~30-60 seconds | ~2 seconds |
| **API Calls** | ~691 (startup) | 0 (startup) |
| **Memory** | ~400 MB | ~200 MB |
| **Dependencies** | torch via API | No torch |
| **Cost** | API rate limits | Free |

---

## ✅ Checklist

Before deploying:

- [ ] Run `python build_index.py` locally
- [ ] Verify `vector_store/` directory exists
- [ ] Commit `vector_store/` to GitHub
- [ ] Push to GitHub
- [ ] Set `HUGGINGFACE_API_TOKEN` in Streamlit Cloud
- [ ] Deploy to Streamlit Cloud
- [ ] Test with sample log

---

**🎉 Your production-grade RAG system is ready!**
