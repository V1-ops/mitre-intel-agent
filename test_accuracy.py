#!/usr/bin/env python3
"""
Test accuracy of MITRE SOC Copilot
Tests with various security scenarios to measure accuracy
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from dotenv import load_dotenv
import json
import re

# Load environment
load_dotenv()
api_token = os.getenv("HUGGINGFACE_API_TOKEN")

if not api_token:
    print("❌ HUGGINGFACE_API_TOKEN not found")
    exit(1)

print("=" * 70)
print("🛡️  MITRE SOC Copilot - Accuracy Test Suite")
print("=" * 70)

# Test cases: (log_text, expected_technique_id, test_name)
test_cases = [
    (
        "Multiple failed SSH login attempts detected from 203.0.113.47 for user 'root' within 60 seconds. Authentication logs show 15 failed attempts using different passwords.",
        "T1110.001",
        "Brute Force - Password Guessing"
    ),
    (
        "Windows Event 4698: A scheduled task was created. Task Name: \\Microsoft\\Windows\\UpdateOrchestrator\\Backup. Creator: SYSTEM. Action: C:\\Windows\\Temp\\update.exe",
        "T1053.005",
        "Scheduled Task/Cron"
    ),
    (
        "Security alert: Process C:\\Windows\\System32\\svchost.exe attempted to inject code into explorer.exe using WriteProcessMemory and CreateRemoteThread APIs",
        "T1055",
        "Process Injection"
    ),
    (
        "User admin successfully logged in from unusual geographic location (Shanghai, China) at 3 AM. Account has never logged in from this location before.",
        "T1078.001",
        "Valid Accounts - Local Accounts"
    ),
    (
        "WHOIS query detected. Attacker performing reconnaissance on company's registered domain: domain.com from IP 192.168.1.100",
        "T1589",
        "Gather Victim Identity Information"
    ),
    (
        "Registry modification detected: HKEY_LOCAL_MACHINE\\System\\CurrentControlSet\\Services\\Tcpip\\Parameters. DisableUXWFE set to 1",
        "T1562.008",
        "Disable Windows Event Logging"
    ),
    (
        "Powershell Empire C2 beacon detected. Callback to 10.0.0.5:8080 established. Shell commands being executed",
        "T1071.001",
        "Application Layer Protocol - Web Protocols"
    ),
    (
        "Credential Access: SAM database dumped using mimikatz. Hashes extracted from ntds.dit via VSS copy",
        "T1003.004",
        "OS Credential Dumping - LSA Secrets"
    ),
]

print(f"\n📋 Running {len(test_cases)} accuracy test cases...\n")

# Load FAISS locally
print("🔄 Loading FAISS index...")
try:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    vector_store = FAISS.load_local(
        "vector_store",
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )
    print("✅ FAISS loaded\n")
except Exception as e:
    print(f"❌ Error loading FAISS: {e}")
    exit(1)

# Test each case
correct = 0
results = []

for i, (log_text, expected_id, test_name) in enumerate(test_cases, 1):
    print(f"[Test {i}/{len(test_cases)}] {test_name}")
    print(f"  Log: {log_text[:70]}...")
    
    try:
        # Get vector search results
        docs = vector_store.similarity_search_with_score(log_text, k=3)
        
        if not docs:
            print(f"  ❌ No results from vector search")
            results.append({
                "test": test_name,
                "expected": expected_id,
                "got": "NO_RESULTS",
                "correct": False,
                "similarity": 0
            })
            continue
        
        # Get top result
        top_doc, distance = docs[0]
        got_id = top_doc.metadata.get("technique_id", "UNKNOWN")
        got_name = top_doc.metadata.get("name", "UNKNOWN")
        similarity = round(1 / (1 + distance), 3)
        
        is_correct = (got_id == expected_id)
        
        if is_correct:
            print(f"  ✅ CORRECT: {got_id} - {got_name} (similarity: {similarity})")
            correct += 1
        else:
            print(f"  ❌ WRONG: Got {got_id}, expected {expected_id} (similarity: {similarity})")
        
        results.append({
            "test": test_name,
            "expected": expected_id,
            "got": got_id,
            "correct": is_correct,
            "similarity": similarity,
            "name": got_name
        })
        
    except Exception as e:
        print(f"  ❌ Error: {str(e)[:60]}")
        results.append({
            "test": test_name,
            "expected": expected_id,
            "got": "ERROR",
            "correct": False,
            "similarity": 0
        })
    
    print()

# Summary
accuracy = (correct / len(test_cases)) * 100
print("=" * 70)
print("📊 VECTOR SEARCH ACCURACY RESULTS")
print("=" * 70)
print(f"\n✅ Correct: {correct}/{len(test_cases)}")
print(f"📈 Accuracy: {accuracy:.1f}%\n")

# Breakdown
print("Detailed Results:")
for result in results:
    status = "✅" if result["correct"] else "❌"
    print(f"  {status} {result['test']}")
    print(f"     Expected: {result['expected']}")
    print(f"     Got: {result['got']}")
    if result.get("similarity"):
        print(f"     Similarity: {result['similarity']}")
    print()

# Save results
with open("accuracy_results.json", "w") as f:
    json.dump({
        "accuracy": accuracy,
        "correct": correct,
        "total": len(test_cases),
        "results": results
    }, f, indent=2)

print(f"📄 Results saved to accuracy_results.json")
print("\n" + "=" * 70)
print("Note: This tests ONLY vector search accuracy (FAISS retrieval)")
print("The full system also uses LLM analysis which can improve/worsen results")
print("=" * 70)
