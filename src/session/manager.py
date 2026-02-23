# src/session/manager.py
"""
Conversation session management
"""

import time
from typing import List, Dict, Any, Optional

class ConversationSession:
    def __init__(self, session_id: str):
        """
        Initialize conversation session
        
        Args:
            session_id: Unique session identifier
        """
        self.session_id = session_id
        self.history: List[Dict[str, Any]] = []
        self.emergency_type: Optional[str] = None
        self.protocol_state: Dict[str, Any] = {}
        self.context_data: Dict[str, Any] = {}
        self.analysis: Dict[str, Any] = {}
        self.created_at = time.time()
        
    def add_exchange(self, user_msg: str, assistant_msg: str):
        """
        Add user-assistant exchange to history
        
        Args:
            user_msg: User's message
            assistant_msg: Assistant's response
        """
        self.history.append({
            "role": "user",
            "content": user_msg,
            "timestamp": time.time()
        })
        self.history.append({
            "role": "assistant",
            "content": assistant_msg,
            "timestamp": time.time()
        })
        
    def get_context(self, last_n: int = 5) -> List[Dict[str, Any]]:
        """
        Get last N exchanges for context
        
        Args:
            last_n: Number of exchanges to retrieve
            
        Returns:
            List of message dictionaries
        """
        return self.history[-last_n*2:] if self.history else []
        
    def update_protocol_state(self, key: str, value: Any):
        """
        Update protocol state
        
        Args:
            key: State key
            value: State value
        """
        self.protocol_state[key] = value
        
    def get_turn_count(self) -> int:
        """Get number of conversation turns"""
        return len(self.history) // 2
