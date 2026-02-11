import streamlit as st
import requests
import json

# ============================================
# PAGE CONFIGURATION
# ============================================
st.set_page_config(
    page_title="MITRE SOC Copilot",
    page_icon="🛡️",
    layout="centered"
)

# ============================================
# HEADER SECTION
# ============================================
st.title("🛡️ MITRE SOC Copilot")
st.markdown(
    """
    **Hybrid RAG System** using the MITRE ATT&CK framework  
    Analyzes security logs and maps them to attack techniques with AI-powered intelligence.
    """
)

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
        # Show loading spinner
        with st.spinner("🔄 Analyzing log with AI..."):
            try:
                # Send POST request to FastAPI backend
                response = requests.post(
                    "http://127.0.0.1:8000/analyze",
                    json={"log": log_input},
                    timeout=30
                )
                
                # Check if request was successful
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check for validation error
                    if "error" in data:
                        st.error(data["error"])
                    else:
                        # ============================================
                        # DISPLAY RESULTS
                        # ============================================
                        st.success("✅ Analysis Complete")
                        
                        # Technique ID and Name
                        st.subheader(f"{data.get('technique_id', 'N/A')}: {data.get('technique_name', 'N/A')}")
                        
                        # Metrics for Severity, Confidence, and Similarity
                        m_col1, m_col2, m_col3 = st.columns(3)
                        
                        severity = data.get('severity', 'Medium')
                        m_col1.metric("Severity", severity)
                        
                        confidence = data.get('confidence', 'N/A')
                        m_col2.metric("Confidence", confidence)
                        
                        similarity = data.get('similarity_score', 0.0)
                        m_col3.metric("Similarity Score", f"{similarity:.2f}")

                        st.divider()
                        
                        # Reasoning
                        st.markdown("**Reasoning:**")
                        st.info(data.get('reasoning', 'No reasoning provided'))
                        
                        # Recommended Mitigation
                        st.markdown("**Recommended Mitigation:**")
                        mitigation = data.get('mitigation', '')
                        if mitigation and mitigation.strip() and mitigation != "No mitigation available":
                            st.success(mitigation)
                        else:
                            st.warning("No mitigation information available")
                        
                        # Expandable section for raw JSON
                        with st.expander("📄 View Raw JSON Response"):
                            st.json(data)
                
                else:
                    # Handle non-200 responses
                    st.error(f"❌ Error: Received status code {response.status_code}")
                    st.error(response.text)
                    
            except requests.exceptions.ConnectionError:
                st.error("❌ **Connection Error:** Cannot connect to the backend API.")
                st.info("Make sure the FastAPI server is running at http://127.0.0.1:8000")
                st.code("uvicorn test_api:app --reload", language="bash")
                
            except requests.exceptions.Timeout:
                st.error("❌ **Timeout Error:** The request took too long to complete.")
                
            except requests.exceptions.RequestException as e:
                st.error(f"❌ **Request Error:** {str(e)}")
                
            except json.JSONDecodeError:
                st.error("❌ **JSON Error:** Could not parse the response from the API.")
                
            except Exception as e:
                st.error(f"❌ **Unexpected Error:** {str(e)}")

# ============================================
# FOOTER SECTION
# ============================================
st.divider()
st.caption("Powered by MITRE ATT&CK Framework • RAG Architecture • AI Analysis")
