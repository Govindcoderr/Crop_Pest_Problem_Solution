"""
FastAPI Backend for Pesticide Recommendation Chatbot
"""

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
from datetime import datetime
import uvicorn
import logging

from chatbot import PesticideChatbot
from session_manager import SessionState

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
chatbot_instance: Optional[PesticideChatbot] = None
user_sessions: Dict[str, SessionState] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for startup and shutdown"""
    global chatbot_instance
    try:
        logger.info("=" * 80)
        logger.info("INITIALIZING PESTICIDE CHATBOT")
        logger.info("=" * 80)
        
        chatbot_instance = PesticideChatbot(
            knowledge_base_path="knowledge_base/pesticide_recommendations.md"
        )
        
        logger.info("✅ Chatbot initialized successfully!")
        logger.info(f"✅ Loaded {len(chatbot_instance.database.all_crops)} crops")
        logger.info(f"✅ Loaded {len(chatbot_instance.database.all_pests)} pests")
        logger.info(f"✅ Loaded {len(chatbot_instance.database.all_applications)} application types")
        logger.info("=" * 80)
    except Exception as e:
        logger.error(f"❌ Error initializing chatbot: {str(e)}")
        chatbot_instance = None
    
    yield
    
    logger.info("Shutting down Pesticide Chatbot API...")


# Initialize FastAPI Application
app = FastAPI(
    title="Pesticide Recommendation Chatbot API",
    description="REST API for pesticide recommendations with multi-turn conversations",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ChatRequest(BaseModel):
    """Request model for chat queries"""
    query: str = Field(..., description="User's query", min_length=1)
    user_id: str = Field(..., description="Unique user identifier")


class SessionStateResponse(BaseModel):
    """Response model for session state"""
    crop: Optional[str] = None
    problem_type: Optional[str] = None
    pest_name: Optional[str] = None
    application_type: Optional[str] = None
    pending_question: Optional[str] = None


class MessageHistory(BaseModel):
    """Individual message in history"""
    role: str
    content: str
    timestamp: str


class ChatResponse(BaseModel):
    """Response model for chat queries"""
    response: str = Field(..., description="Bot's response")
    session_state: SessionStateResponse = Field(..., description="Current session state")
    message_count: int = Field(..., description="Number of messages in history")


class HistoryResponse(BaseModel):
    """Response model for conversation history"""
    messages: List[MessageHistory] = Field(..., description="Last 10 messages")
    total_messages: int = Field(..., description="Total number of messages")


class DatabaseStatsResponse(BaseModel):
    """Response model for database statistics"""
    total_crops: int
    total_pests: int
    total_applications: int
    top_crops: List[str]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_or_create_session(user_id: str) -> SessionState:
    """Get existing session or create new one for user"""
    if user_id not in user_sessions:
        user_sessions[user_id] = SessionState()
        logger.info(f"Created new session for user: {user_id}")
    return user_sessions[user_id]


def session_to_response(session: SessionState) -> SessionStateResponse:
    """Convert SessionState to response model"""
    return SessionStateResponse(
        crop=session.crop,
        problem_type=session.problem_type,
        pest_name=session.pest_name,
        application_type=session.application_type,
        pending_question=session.pending_question
    )


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/", tags=["System"])
async def root():
    """Root endpoint"""
    return {
        "message": "Pesticide Recommendation Chatbot API",
        "version": "1.0.0",
        "status": "active" if chatbot_instance else "inactive",
        "docs": "/docs"
    }


@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint"""
    if not chatbot_instance:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    return {
        "status": "healthy",
        "chatbot_initialized": True,
        "database_stats": {
            "crops": len(chatbot_instance.database.all_crops),
            "pests": len(chatbot_instance.database.all_pests),
            "applications": len(chatbot_instance.database.all_applications)
        },
        "active_sessions": len(user_sessions),
        "timestamp": datetime.now().isoformat()
    }


@app.post("/chat", response_model=ChatResponse, tags=["Chat"])
async def chat(request: ChatRequest):
    """
    Main chat endpoint - process user queries with multi-turn conversation
    
    - **query**: User's input message
    - **user_id**: Unique identifier for the user (maintains session context)
    """
    if not chatbot_instance:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    try:
        # Get or create session for user
        session = get_or_create_session(request.user_id)
        
        logger.info(f"Chat request from user {request.user_id}: '{request.query}'")
        
        # Process query
        response, updated_session = chatbot_instance.chat(request.query, session)
        
        # Update session
        user_sessions[request.user_id] = updated_session
        
        logger.info(f"Response generated successfully for user {request.user_id}")
        
        return ChatResponse(
            response=response,
            session_state=session_to_response(updated_session),
            message_count=len(updated_session.last_10_messages)
        )
    
    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.get("/history/{user_id}", response_model=HistoryResponse, tags=["Chat"])
async def get_history(user_id: str):
    """
    Get conversation history for a user
    
    - **user_id**: User identifier
    """
    if user_id not in user_sessions:
        return HistoryResponse(messages=[], total_messages=0)
    
    session = user_sessions[user_id]
    
    messages = [
        MessageHistory(
            role=msg['role'],
            content=msg['content'],
            timestamp=msg['timestamp']
        )
        for msg in session.last_10_messages
    ]
    
    return HistoryResponse(
        messages=messages,
        total_messages=len(messages)
    )


@app.post("/clear-history/{user_id}", tags=["Chat"])
async def clear_history(user_id: str):
    """
    Clear conversation history for a user
    
    - **user_id**: User identifier
    """
    if user_id in user_sessions:
        user_sessions[user_id].reset()
        logger.info(f"Cleared history for user: {user_id}")
        return {
            "status": "success",
            "message": f"History cleared for user {user_id}",
            "timestamp": datetime.now().isoformat()
        }
    
    return {
        "status": "info",
        "message": f"No session found for user {user_id}",
        "timestamp": datetime.now().isoformat()
    }


@app.delete("/session/{user_id}", tags=["Chat"])
async def delete_session(user_id: str):
    """
    Delete entire session for a user
    
    - **user_id**: User identifier
    """
    if user_id in user_sessions:
        del user_sessions[user_id]
        logger.info(f"Deleted session for user: {user_id}")
        return {
            "status": "success",
            "message": f"Session deleted for user {user_id}",
            "timestamp": datetime.now().isoformat()
        }
    
    return {
        "status": "info",
        "message": f"No session found for user {user_id}",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/session/{user_id}", response_model=SessionStateResponse, tags=["Chat"])
async def get_session_state(user_id: str):
    """
    Get current session state for a user
    
    - **user_id**: User identifier
    """
    if user_id not in user_sessions:
        return SessionStateResponse()
    
    session = user_sessions[user_id]
    return session_to_response(session)


@app.get("/database/stats", response_model=DatabaseStatsResponse, tags=["Database"])
async def get_database_stats():
    """
    Get database statistics
    """
    if not chatbot_instance:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    db = chatbot_instance.database
    
    return DatabaseStatsResponse(
        total_crops=len(db.all_crops),
        total_pests=len(db.all_pests),
        total_applications=len(db.all_applications),
        top_crops=db.get_most_common_crops(limit=10)
    )


@app.get("/database/crops", tags=["Database"])
async def get_all_crops():
    """
    Get list of all available crops
    """
    if not chatbot_instance:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    return {
        "crops": chatbot_instance.database.all_crops,
        "total": len(chatbot_instance.database.all_crops)
    }


@app.get("/database/pests", tags=["Database"])
async def get_all_pests(crop: Optional[str] = None):
    """
    Get list of all pests (optionally filtered by crop)
    
    - **crop**: Optional crop name to filter pests
    """
    if not chatbot_instance:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    db = chatbot_instance.database
    
    if crop:
        # Fuzzy match crop
        matched_crop = db.fuzzy_match_crop(crop)
        if not matched_crop:
            raise HTTPException(status_code=404, detail=f"Crop '{crop}' not found")
        
        pests = db.get_pests(matched_crop)
        return {
            "crop": matched_crop,
            "pests": pests,
            "total": len(pests)
        }
    
    return {
        "pests": db.all_pests,
        "total": len(db.all_pests)
    }


@app.get("/database/applications", tags=["Database"])
async def get_all_applications():
    """
    Get list of all application types
    """
    if not chatbot_instance:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    return {
        "application_types": chatbot_instance.database.all_applications,
        "total": len(chatbot_instance.database.all_applications)
    }


@app.get("/database/solutions", tags=["Database"])
async def get_solutions(
    crop: str,
    pest: Optional[str] = None,
    application: Optional[str] = None
):
    """
    Get solutions for specific crop/pest/application combination
    
    - **crop**: Crop name (required)
    - **pest**: Pest name (optional)
    - **application**: Application type (optional)
    """
    if not chatbot_instance:
        raise HTTPException(status_code=503, detail="Chatbot not initialized")
    
    db = chatbot_instance.database
    
    # Fuzzy match crop
    matched_crop = db.fuzzy_match_crop(crop)
    if not matched_crop:
        raise HTTPException(status_code=404, detail=f"Crop '{crop}' not found")
    
    # Fuzzy match pest if provided
    matched_pest = None
    if pest:
        matched_pest = db.fuzzy_match_pest(pest, matched_crop)
        if not matched_pest:
            raise HTTPException(status_code=404, detail=f"Pest '{pest}' not found for {matched_crop}")
    
    # Fuzzy match application if provided
    matched_app = None
    if application:
        matched_app = db.fuzzy_match_application_type(application)
        if not matched_app:
            raise HTTPException(status_code=404, detail=f"Application type '{application}' not found")
    
    solutions = db.get_solutions(matched_crop, matched_pest, matched_app)
    
    return {
        "crop": matched_crop,
        "pest": matched_pest,
        "application_type": matched_app,
        "solutions": solutions,
        "total": len(solutions)
    }


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )