"""
Session Manager - Track conversation state and history
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
import re


class SessionState:
    """
    Manage conversation state for multi-turn interactions
    """
    
    def __init__(self):
        self.crop: Optional[str] = None
        self.problem_type: Optional[str] = None
        self.pest_name: Optional[str] = None
        self.application_type: Optional[str] = None
        
        # Conversation tracking
        self.last_10_messages: List[Dict[str, Any]] = []
        self.pending_question: Optional[str] = None  # What the bot is waiting for
        self.last_bot_message: Optional[str] = None
    
    def update_from_entities(self, entities: Dict[str, Optional[str]]):
        """Update session state from extracted entities"""
        if entities.get('crop'):
            self.crop = entities['crop']
        
        if entities.get('problem_type'):
            self.problem_type = entities['problem_type']
        
        if entities.get('pest_name'):
            self.pest_name = entities['pest_name']
        
        if entities.get('application_type'):
            self.application_type = entities['application_type']
    
    def is_complete(self) -> bool:
        """Check if we have enough info to provide solutions"""
        return self.crop is not None and self.pest_name is not None
    
    def get_missing_fields(self) -> List[str]:
        """Get list of missing required fields"""
        missing = []
        if not self.crop:
            missing.append('crop')
        if not self.pest_name:
            missing.append('pest_name')
        return missing
    
    def add_message(self, role: str, content: str):
        """
        Add message to history (last 10 only)
        Auto-strip reasoning tags before storing
        """
        cleaned_content = self._strip_reasoning(content)
        
        self.last_10_messages.append({
            'role': role,
            'content': cleaned_content,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 10
        if len(self.last_10_messages) > 10:
            self.last_10_messages = self.last_10_messages[-10:]
        
        # Track last bot message for context
        if role == 'assistant':
            self.last_bot_message = cleaned_content
    
    def _strip_reasoning(self, text: str) -> str:
        """Remove reasoning tags from text"""
        text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<thought>.*?</thought>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        return text.strip()
    
    def get_history_for_llm(self, max_messages: int = 6) -> str:
        """
        Format recent history for LLM context
        Returns last N messages as formatted string
        """
        recent = self.last_10_messages[-max_messages:] if len(self.last_10_messages) > max_messages else self.last_10_messages
        
        if not recent:
            return ""
        
        formatted = []
        for msg in recent:
            role = "User" if msg['role'] == 'user' else "Assistant"
            formatted.append(f"{role}: {msg['content']}")
        
        return "\n".join(formatted)
    
    def get_last_bot_message(self) -> str:
        """Get the last message from the bot"""
        return self.last_bot_message or ""
    
    def get_last_user_message(self) -> str:
        """Get the last message from the user"""
        for msg in reversed(self.last_10_messages):
            if msg['role'] == 'user':
                return msg['content']
        return ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dict for serialization"""
        return {
            'crop': self.crop,
            'problem_type': self.problem_type,
            'pest_name': self.pest_name,
            'application_type': self.application_type,
            'pending_question': self.pending_question
        }
    
    def reset(self):
        """Reset all state (for new conversation)"""
        self.crop = None
        self.problem_type = None
        self.pest_name = None
        self.application_type = None
        self.last_10_messages = []
        self.pending_question = None
        self.last_bot_message = None