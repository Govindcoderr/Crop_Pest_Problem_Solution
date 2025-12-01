# 🌾 Pesticide Recommendation Chatbot

AI-powered assistant for pesticide recommendations using AWS Bedrock, structured data, and RAG fallback.

---

## 📁 Project Structure

```
project/
├── knowledge_base/
│   └── pesticide_recommendations.md
├── chroma_db/                    # Auto-created
├── .env
├── requirements.txt
├── corrector.py
├── database.py
├── session_manager.py
├── chatbot.py
└── app_ui.py
```

---

## ⚙️ Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Credentials

Create a `.env` file:

```env

API_ACCESS_KEY_ID=your_access_key

```

### 3. Add Your Data

Place `pesticide_recommendations.md` in the `knowledge_base/` folder.

---

## 🚀 Run the Application

### Option 1: Start API + UI (Recommended)

**Terminal 1 - Start FastAPI Backend:**
```bash
python app.py
```
API will be available at: **http://localhost:8000**

**Terminal 2 - Start Streamlit UI:**
```bash
streamlit run app_ui.py
```
UI will be available at: **http://localhost:8501**

### Option 2: API Only (For Testing)

```bash
python app.py
```

Then visit: **http://localhost:8000/docs** for interactive API documentation

---

## 🧪 Testing the API

### Using cURL

**Health Check:**
```bash
curl http://localhost:8000/health
```

**Chat Request:**
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "grapes powdery mildew", "user_id": "test-user-123"}'
```

**Get History:**
```bash
curl http://localhost:8000/history/test-user-123
```

**Clear History:**
```bash
curl -X POST http://localhost:8000/clear-history/test-user-123
```

**Get Database Stats:**
```bash
curl http://localhost:8000/database/stats
```

**Get All Crops:**
```bash
curl http://localhost:8000/database/crops
```

**Get Pests for Crop:**
```bash
curl "http://localhost:8000/database/pests?crop=grapes"
```

**Get Solutions:**
```bash
curl "http://localhost:8000/database/solutions?crop=grapes&pest=powdery%20mildew"
```

### Using Postman/Thunder Client

Import these endpoints:
- `POST /chat` - Main chat endpoint
- `GET /history/{user_id}` - Get conversation history
- `POST /clear-history/{user_id}` - Clear history
- `GET /session/{user_id}` - Get session state
- `DELETE /session/{user_id}` - Delete session
- `GET /database/stats` - Database statistics
- `GET /database/crops` - All crops
- `GET /database/pests?crop=grapes` - Pests for crop
- `GET /database/applications` - All application types
- `GET /database/solutions?crop=grapes&pest=powdery mildew` - Get solutions

---

## 💡 Usage Examples

### Example 1: Complete Query
```
User: "grapes powdery mildew"
Bot: [Shows all solutions for Grapes - Powdery mildew]
```

### Example 2: Incomplete Query
```
User: "I have grape problem"
Bot: "What pest problem in Grapes? Options: Powdery mildew, Downy mildew, ..."
User: "powdery mildew"
Bot: [Shows solutions]
```

### Example 3: Uncertainty
```
User: "grapes"
Bot: "What pest problem in Grapes?"
User: "not sure"
Bot: [Shows ALL solutions for Grapes]
```

### Example 4: Context Carryover
```
User: "grapes powdery mildew"
Bot: [Shows solutions]
User: "what about foliar spray"
Bot: [Shows only foliar spray solutions for Grapes + Powdery mildew]
```

---

## 🏗️ Architecture

### Components

1. **TextCorrector** (`corrector.py`)
   - Fixes typos using AWS Nova Lite
   - "graeps" → "grapes"

2. **PesticideDatabase** (`database.py`)
   - Parses markdown table → structured dict
   - Fast O(1) lookups for crops/pests/solutions
   - Fuzzy matching for user queries

3. **SessionState** (`session_manager.py`)
   - Tracks conversation context
   - Stores last 10 messages
   - Manages multi-turn flow

4. **PesticideChatbot** (`chatbot.py`)
   - LLM-based intent detection
   - Structured data routing
   - RAG fallback for general questions

5. **Streamlit UI** (`app_ui.py`)
   - Clean chat interface
   - Session state display
   - Conversation history

### Flow

```
User Query
    ↓
Text Correction (LLM MODEL)
    ↓
Intent Detection (LLM with context)
    ↓
    ├─→ Structured Lookup (DB)
    ├─→ Ask Follow-up Question
    └─→ RAG Fallback (ChromaDB)
    ↓
Response Generation
    ↓
Update Session State
```

---

## 🎯 Key Features

✅ **Typo Correction**: Automatic spelling fixes
✅ **Smart Intent Detection**: Context-aware understanding
✅ **Multi-turn Conversations**: Remembers context
✅ **Uncertainty Handling**: "I don't know" → shows all options
✅ **Fuzzy Matching**: Handles variations ("graep" → "Grapes")
✅ **Session State**: Tracks crop/pest across messages
✅ **RAG Fallback**: Handles general questions
✅ **Clean UI**: Simple Streamlit interface

---

## 🛠️ Troubleshooting

### Issue: "Chatbot not initialized"
- Check API  credentials in `.env`

### Issue: "No solutions found"
- Check if `pesticide_recommendations.md` exists
- Verify table format (must have header + separator)

### Issue: "Embedding error"
- Ensure Cohere model is available in your region
- Check AWS Bedrock model access

---

## 📊 Models Used

| Component | Model | Purpose |
|-----------|-------|---------|
| Text Correction | `openai/gpt-oss-20b` | Fix typos |
| Chat | `USED GROQ LLM MODEL ` | Intent detection, responses |
| Embeddings | `USED LOCAL EMBED` | RAG fallback |

---

## 🔧 Customization

### Change Models

Edit in `chatbot.py`:
```python
self.chat_model = "apac.amazon.nova-lite-v1:0"
self.embedding_model = "cohere.embed-english-v3"
```

### Adjust History Length

Edit in `session_manager.py`:
```python
# Change from 10 to any number
if len(self.last_10_messages) > 10:
```

### Modify Fuzzy Match Threshold

Edit in `database.py`:
```python
def _fuzzy_match(self, query: str, candidates: List[str], threshold: float = 0.6):
```

---

## 📝 License

MIT License - Feel free to use and modify!

---

## 🤝 Contributing

Issues and PRs welcome!

## 📊 Architecture Overview

┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT UI (app_ui.py)                  │
│                    Port: 8501                                 │
│   - User interface                                            │
│   - Makes HTTP requests to API                                │
│   - Displays responses                                        │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP Requests
                         │ (localhost:8000)
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                  FASTAPI BACKEND (app.py)                     │
│                    Port: 8000                                 │
│                                                               │
│   Endpoints:                                                  │
│   ├─ POST /chat              (Main chat)                     │
│   ├─ GET /history/{user_id}  (Conversation history)          │
│   ├─ POST /clear-history     (Clear history)                 │
│   ├─ GET /session/{user_id}  (Session state)                 │
│   ├─ DELETE /session         (Delete session)                │
│   ├─ GET /database/stats     (Database info)                 │
│   ├─ GET /database/crops     (All crops)                     │
│   ├─ GET /database/pests     (All pests)                     │
│   └─ GET /database/solutions (Get solutions)                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                CHATBOT LOGIC (chatbot.py)                     │
│   - PesticideChatbot class                                    │
│   - Intent detection                                          │
│   - Session management                                        │
│   - Response generation                                       │
└────────────────────────┬────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ↓                ↓                ↓
┌──────────────┐ ┌──────────────┐ ┌──────────────┐
│  Database    │ │  Corrector   │ │  Session     │
│  (database.py)│ │(corrector.py)│ │ (session_    │
│              │ │              │ │  manager.py) │
└──────────────┘ └──────────────┘ └──────────────┘