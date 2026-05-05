#!/usr/bin/env python3
"""
Direct FAISS accuracy test - minimal dependencies
"""
import os
import json
import sys

print("=" * 70)
print("🛡️  MITRE SOC Copilot - Vector Search Accuracy Test")
print("=" * 70)

# Test cases
test_cases = [
    (
        "Multiple failed SSH login attempts detected from 203.0.113.47 for user 'root'. 15 failed attempts using different passwords.",
        "T1110.001",
        "Brute Force - Password Guessing"
    ),
    (
        "Windows scheduled task created: C:\\Windows\\Temp\\update.exe",
        "T1053.005",
        "Scheduled Task"
    ),
    (
        "Process injection detected: WriteProcessMemory and CreateRemoteThread APIs used",
        "T1055",
        "Process Injection"
    ),
    (
        "User logged in from unusual geographic location - Shanghai, China at 3 AM",
        "T1078.001",
        "Valid Accounts"
    ),
    (
        "WHOIS query detected - reconnaissance on company domain",
        "T1589",
        "Gather Victim Identity Information"
    ),
    (
        "SAM database dumped using mimikatz. Hashes extracted from ntds.dit",
        "T1003.004",
        "OS Credential Dumping"
    ),
    (
        "Powershell Empire C2 beacon callback to 10.0.0.5:8080",
        "T1071.001",
        "Application Layer Protocol"
    ),
    (
        "Registry modification: DisableUXWFE set to 1 - Windows event logging disabled",
        "T1562.008",
        "Disable Windows Event Logging"
    ),
]

print(f"\n📋 Testing {len(test_cases)} cases...\n")

try:
    # Try native FAISS import
    print("🔄 Loading FAISS index (native)...")
    import faiss
    import numpy as np
    import pickle
    
    # Load index and metadata
    index = faiss.read_index("vector_store/index.faiss")
    
    with open("vector_store/index.pkl", "rb") as f:
        metadata_dict = pickle.load(f)
    
    # Load embeddings model - try sentence-transformers without the problematic LangChain layer
    print("🔄 Loading embedding model...")
    from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("✅ Model loaded\n")
    
    # Test
    correct = 0
    results = []
    
    for i, (log_text, expected_id, test_name) in enumerate(test_cases, 1):
        print(f"[Test {i}/{len(test_cases)}] {test_name}")
        print(f"  Log: {log_text[:60]}...")
        
        try:
            # Embed query
            query_vector = model.encode([log_text])
            query_vector = np.array(query_vector, dtype=np.float32)
            
            # Search
            distances, indices = index.search(query_vector, k=3)
            
            if len(indices[0]) > 0:
                top_idx = indices[0][0]
                distance = distances[0][0]
                
                # Get metadata
                doc_key = list(metadata_dict.keys())[top_idx] if top_idx < len(metadata_dict) else None
                
                if doc_key in metadata_dict:
                    metadata = metadata_dict[doc_key]
                    got_id = metadata.get("technique_id", "UNKNOWN")
                    got_name = metadata.get("name", "UNKNOWN")
                else:
                    got_id = "NOT_FOUND"
                    got_name = "NOT_FOUND"
                
                # Calculate similarity (note: FAISS returns L2 distance, convert to similarity)
                similarity = 1.0 / (1.0 + distance)
                
                is_correct = (got_id == expected_id)
                
                if is_correct:
                    print(f"  ✅ CORRECT: {got_id} (similarity: {similarity:.3f})")
                    correct += 1
                else:
                    print(f"  ❌ WRONG: Got {got_id}, expected {expected_id} (similarity: {similarity:.3f})")
                    print(f"     Got: {got_name}")
                
                results.append({
                    "test": test_name,
                    "expected": expected_id,
                    "got": got_id,
                    "correct": is_correct,
                    "similarity": round(similarity, 3)
                })
            else:
                print(f"  ❌ No results")
                results.append({
                    "test": test_name,
                    "expected": expected_id,
                    "got": "NO_RESULTS",
                    "correct": False,
                    "similarity": 0
                })
        
        except Exception as e:
            print(f"  ❌ Error: {str(e)[:50]}")
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
    print("📊 VECTOR SEARCH ACCURACY (FAISS Retrieval Only)")
    print("=" * 70)
    print(f"\n✅ Correct: {correct}/{len(test_cases)}")
    print(f"📈 Vector Search Accuracy: {accuracy:.1f}%\n")
    
    # Breakdown
    print("Detailed Results:")
    print("-" * 70)
    for result in results:
        status = "✅" if result["correct"] else "❌"
        print(f"{status} {result['test']:<40} Similarity: {result['similarity']:.3f}")
        if not result["correct"]:
            print(f"   Expected: {result['expected']:<20} Got: {result['got']}")
    
    # Save results
    with open("accuracy_results.json", "w") as f:
        json.dump({
            "phase": "vector_search_only",
            "accuracy": accuracy,
            "correct": correct,
            "total": len(test_cases),
            "results": results,
            "notes": "This tests ONLY the FAISS vector retrieval accuracy. The full system adds LLM refinement which can improve/worsen results."
        }, f, indent=2)
    
    print(f"\n📄 Full results saved to accuracy_results.json")
    
    print("\n" + "=" * 70)
    print("📝 INTERPRETATION:")
    print("=" * 70)
    print(f"Vector Search Accuracy: {accuracy:.1f}%")
    print("\nNote: This measures ONLY the semantic search accuracy.")
    print("The full system also runs LLM analysis which:")
    print("  • Can improve accuracy by using context")
    print("  • Can hurt accuracy through hallucination")
    print("=" * 70)
    
except ImportError as e:
    print(f"❌ Missing required package: {e}")
    print("\nTo fix, run:")
    print("  pip install faiss-cpu sentence-transformers")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
