
"""
Optimized Streamlit UI for Pesticide Recommendation Chatbot
KEY CHANGES:
1. Removed st.rerun() to prevent fade effect
2. Direct message append without full page reload
"""

import streamlit as st
import requests
import uuid
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

API_BASE_URL = "http://localhost:8000"

st.set_page_config(
    page_title="Pesticide Advisor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
    .history-item {
        font-size: 0.85rem;
        padding: 0.3rem;
        margin: 0.2rem 0;
        border-left: 3px solid #4CAF50;
        padding-left: 0.5rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-size: 0.85rem;
        font-weight: bold;
    }
    .status-online {
        background-color: #4CAF50;
        color: white;
    }
    .status-offline {
        background-color: #f44336;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# API Helper Functions
def check_api_health():
    """Check if API is online"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200, response.json()
    except Exception as e:
        logger.error(f"API health check failed: {e}")
        return False, None

def send_chat_message(query: str, user_id: str):
    """Send chat message to API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={"query": query, "user_id": user_id},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.Timeout:
        raise Exception("Request timed out. Please try again.")
    except requests.exceptions.RequestException as e:
        raise Exception(f"API request failed: {str(e)}")

def get_conversation_history(user_id: str):
    """Get conversation history from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/history/{user_id}", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to get history: {e}")
        return {"messages": [], "total_messages": 0}

def clear_conversation_history(user_id: str):
    """Clear conversation history via API"""
    try:
        response = requests.post(f"{API_BASE_URL}/clear-history/{user_id}", timeout=5)
        response.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"Failed to clear history: {e}")
        return False

def get_session_state(user_id: str):
    """Get session state from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/session/{user_id}", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to get session state: {e}")
        return {}

def get_database_stats():
    """Get database statistics from API"""
    try:
        response = requests.get(f"{API_BASE_URL}/database/stats", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        return None

# Session Initialization
if "user_id" not in st.session_state:
    st.session_state.user_id = str(uuid.uuid4())
    logger.info(f"Generated new user ID: {st.session_state.user_id}")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "api_online" not in st.session_state:
    st.session_state.api_online = False

# SIDEBAR
with st.sidebar:
    st.title("🌾 Pesticide Advisor")
    
    # API Status
    api_online, health_data = check_api_health()
    st.session_state.api_online = api_online
    
    if api_online:
        st.markdown('<span class="status-badge status-online">🟢 API Online</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-badge status-offline">🔴 API Offline</span>', unsafe_allow_html=True)
        st.error("Please start the FastAPI server:\n```bash\npython app.py\n```")
    
    st.markdown("---")
    
    # Current Session State (only if API is online)
    if api_online:
        st.subheader("📋 Current Session")
        session_state = get_session_state(st.session_state.user_id)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Crop", session_state.get("crop") or "Not set")
        with col2:
            st.metric("Pest", session_state.get("pest_name") or "Not set")
        
        if session_state.get("application_type"):
            st.metric("Application", session_state.get("application_type"))
        
        st.markdown("---")
    
    # Action Buttons - OPTIMIZED: Use key to prevent full rerun
    if st.button("🗑️ Clear Chat", use_container_width=True, disabled=not api_online, key="clear_btn"):
        if clear_conversation_history(st.session_state.user_id):
            st.session_state.messages = []
            st.success("Chat cleared!")
            # Only rerun on clear action (intentional)
            st.rerun()
    
    if api_online:
        st.markdown("---")
        
        # Conversation History
        st.subheader("💬 Recent History")
        history_data = get_conversation_history(st.session_state.user_id)
        messages = history_data.get("messages", [])
        
        st.caption(f"Last {len(messages)} messages")
        
        if messages:
            for msg in messages[-6:]:
                role_emoji = "👤" if msg['role'] == 'user' else "🤖"
                preview = msg['content'][:60] + "..." if len(msg['content']) > 60 else msg['content']
                st.markdown(f"<div class='history-item'>{role_emoji} {preview}</div>", unsafe_allow_html=True)
        else:
            st.caption("No messages yet")
        
        st.markdown("---")
        
        # Database Stats
        st.subheader("📊 Database Info")
        db_stats = get_database_stats()
        
        if db_stats:
            st.caption(f"🌱 {db_stats['total_crops']} Crops")
            st.caption(f"🐛 {db_stats['total_pests']} Pests")
            st.caption(f"💧 {db_stats['total_applications']} Application Types")
        
        st.markdown("---")
    
    st.caption(f"User ID: {st.session_state.user_id[:8]}...")
    st.caption("Built with using Groq Api & FastAPI")

# MAIN CHAT INTERFACE
st.title("🌾 Pesticide Recommendation Assistant")

if not api_online:
    st.error("""
    ⚠️ **API Server is Offline**
    
    Please start the FastAPI server first:
    
    ```bash
    python app.py
    ```
    
    Then refresh this page.
    """)
    st.stop()

st.markdown("Ask me about pesticide solutions for your crops!")

# Display Chat Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input - OPTIMIZED: No rerun after response
if prompt := st.chat_input("Ask about pesticides... (e.g., 'grapes powdery mildew')"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response via API
    with st.chat_message("assistant"):
        with st.spinner("🤔 Thinking..."):
            try:
                # Call API
                result = send_chat_message(prompt, st.session_state.user_id)
                response = result['response']
                
                # Display response
                st.markdown(response)
                
                # Add assistant response to messages
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Show session state update if changed (optional, collapsed)
                session_state = result.get('session_state', {})
                if session_state.get('crop') or session_state.get('pest_name'):
                    with st.expander("📊 Session Updated", expanded=False):
                        st.json(session_state)
                
            except Exception as e:
                error_msg = f"❌ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
                logger.error(f"Chat error: {e}")
    
    # REMOVED st.rerun() - This was causing the fade effect!
    # Streamlit will automatically update the chat without full page reload

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    💡 <b>Tips:</b> Be specific about your crop and pest problem for best results<br>
    Say "not sure" if you don't know the pest name to see all solutions
</div>
""", unsafe_allow_html=True)

# Debug Info (collapsible)
with st.expander("🔧 Debug Info", expanded=False):
    st.json({
        "user_id": st.session_state.user_id,
        "api_base_url": API_BASE_URL,
        "api_online": api_online,
        "messages_count": len(st.session_state.messages)
    })