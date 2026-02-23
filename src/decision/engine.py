# src/decision/engine.py
"""
Decision tree navigation engine
"""

import yaml
import os
from typing import Dict, Any, Optional, List
from pathlib import Path

class DecisionEngine:
    def __init__(self, config: dict):
        """
        Initialize decision engine
        
        Args:
            config: Decision engine configuration
        """
        self.tree_dir = config.get("tree_dir", "decision_trees")
        self.trees: Dict[str, Dict] = {}
        
        # Load all decision trees
        self._load_trees()
        
    def _load_trees(self):
        """Load all YAML decision trees from directory"""
        if not os.path.exists(self.tree_dir):
            print(f"⚠️ Decision tree directory not found: {self.tree_dir}")
            return
            
        tree_files = Path(self.tree_dir).glob("*.yaml")
        for tree_file in tree_files:
            try:
                with open(tree_file, 'r') as f:
                    tree_data = yaml.safe_load(f)
                tree_id = tree_data.get("tree_id")
                if tree_id:
                    self.trees[tree_id.replace("_protocol", "")] = tree_data
                    print(f"✅ Loaded decision tree: {tree_id}")
            except Exception as e:
                print(f"❌ Error loading {tree_file}: {e}")
                
        if self.trees:
            print(f"✅ Decision engine initialized with {len(self.trees)} trees")
        else:
            print("⚠️ No decision trees loaded")
            
    def navigate(self, emergency_type: str, session, user_response: str = None) -> Dict[str, Any]:
        """
        Navigate decision tree based on conversation state
        
        Args:
            emergency_type: Type of emergency (bleeding, burn, etc.)
            session: Current conversation session
            user_response: User's latest response
            
        Returns:
            Dictionary with action, message, tags, etc.
        """
        # Get tree for emergency type
        tree = self.trees.get(emergency_type)
        if not tree:
            return {
                "action": "fallback",
                "message": "I'll help you with this emergency. Tell me more about what's happening.",
                "next_node_id": None,
                "completed": False
            }
        
        nodes = tree.get("nodes", {})
        
        # Get current node from session state
        current_node_id = session.protocol_state.get("current_node", "root")
        current_node = nodes.get(current_node_id, nodes.get("root"))
        
        if not current_node:
            return self._fallback_response()
        
        # If user just responded, find next node
        if user_response and current_node.get("type") == "question":
            next_node_id = self._find_next_node(current_node, user_response)
            if next_node_id:
                current_node_id = next_node_id
                current_node = nodes.get(next_node_id)
                
        # Update session
        session.update_protocol_state("current_node", current_node_id)
        
        # Build response based on node type
        node_type = current_node.get("type", "instruction")
        
        if node_type == "question":
            return {
                "action": "ask",
                "message": current_node.get("question", "Can you tell me more?"),
                "next_node_id": current_node_id,
                "completed": False,
                "critical": current_node.get("critical", False)
            }
            
        elif node_type == "instruction":
            return {
                "action": "instruct",
                "message": current_node.get("instruction", "Follow these steps carefully."),
                "rag_tags": current_node.get("rag_tags", []),
                "next_node_id": current_node.get("next"),
                "completed": current_node.get("terminal", False),
                "critical": current_node.get("critical", False)
            }
            
        elif node_type == "escalate":
            return {
                "action": "escalate",
                "message": current_node.get("message", "⚠️ CALL EMERGENCY SERVICES"),
                "completed": True,
                "critical": True
            }
            
        return self._fallback_response()
        
    def _find_next_node(self, node: Dict, user_response: str) -> Optional[str]:
        """Find next node based on user response matching"""
        user_lower = user_response.lower()
        options = node.get("options", [])
        
        for option in options:
            match_keywords = option.get("response_match", [])
            if any(keyword.lower() in user_lower for keyword in match_keywords):
                return option.get("next")
                
        # Default to first option if no match
        if options:
            return options[0].get("next")
            
        return None
        
    def _fallback_response(self) -> Dict[str, Any]:
        """Fallback response when tree navigation fails"""
        return {
            "action": "ask",
            "message": "I understand. Can you describe what's happening in more detail?",
            "next_node_id": None,
            "completed": False
        }
