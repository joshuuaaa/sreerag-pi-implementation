# src/decision/engine.py
"""
Decision tree navigation engine
"""

import yaml
import os
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger("decision.engine")

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
            logger.warning("Decision tree directory not found: %s", self.tree_dir)
            return
            
        tree_files = Path(self.tree_dir).glob("*.yaml")
        for tree_file in tree_files:
            try:
                with open(tree_file, 'r') as f:
                    tree_data = yaml.safe_load(f)
                tree_id = tree_data.get("tree_id")
                if tree_id:
                    self.trees[tree_id.replace("_protocol", "")] = tree_data
                    logger.info("Loaded decision tree: %s", tree_id)
            except Exception as e:
                logger.error("Error loading %s: %s", tree_file, e)
                
        if self.trees:
            logger.info("Decision engine initialized with %s trees", len(self.trees))
        else:
            logger.warning("No decision trees loaded")
            
    def navigate(
        self,
        emergency_type: str,
        session,
        user_response: str = None,
        secondary_conditions: list = None,
    ) -> Dict[str, Any]:
        """
        Navigate decision tree based on conversation state.

        Args:
            emergency_type:       Type of emergency (bleeding, burn, etc.)
            session:              Current conversation session
            user_response:        User's latest response
            secondary_conditions: Other detected condition types (e.g. ['fracture'])

        Returns:
            Dictionary with action, message, tags, etc.
        """
        secondary_conditions = secondary_conditions or []
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
        start_node = tree.get("start", "root")
        current_node_id = session.protocol_state.get("current_node", start_node)
        current_node = nodes.get(current_node_id, nodes.get(start_node, nodes.get("root")))

        if not current_node:
            return self._fallback_response()

        # ── Contraindication check ────────────────────────────────────────────
        # When the primary tree's default nodes would give harmful instructions
        # due to a secondary condition, divert to a complication-specific branch.
        _PRESSURE_NODES = {
            start_node, "root", "assess_severity", "severe_bleeding",
            "moderate_bleeding", "find_materials",
        }
        if emergency_type == "bleeding" and not session.protocol_state.get("complication_checked"):
            session.update_protocol_state("complication_checked", True)
            if "fracture" in secondary_conditions and "open_fracture_bleeding" in nodes \
                    and current_node_id in _PRESSURE_NODES:
                session.update_protocol_state("current_node", "open_fracture_bleeding")
                current_node_id = "open_fracture_bleeding"
                current_node = nodes["open_fracture_bleeding"]
            elif "chest_injury" in secondary_conditions and "chest_wound_bleeding" in nodes \
                    and current_node_id in _PRESSURE_NODES:
                session.update_protocol_state("current_node", "chest_wound_bleeding")
                current_node_id = "chest_wound_bleeding"
                current_node = nodes["chest_wound_bleeding"]

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
            # 'question' field (new format) or 'text' field (old format)
            message = (
                current_node.get("question")
                or current_node.get("text")
                or "Can you tell me more?"
            )
            return {
                "action": "ask",
                "message": message,
                "next_node_id": current_node_id,
                "completed": False,
                "critical": current_node.get("critical", False)
            }
            
        elif node_type in ("instruction", "action"):
            # 'instruction' (new format) uses 'instruction' field
            # 'action' (old format) uses 'text' field
            message = (
                current_node.get("instruction")
                or current_node.get("text")
                or "Follow these steps carefully."
            )

            # Advance session to the next node so the conversation keeps moving
            next_id = current_node.get("next")
            next_node = nodes.get(next_id) if next_id else None

            # Keep the next question separate — the orchestrator appends it
            # directly to the response WITHOUT passing it through the LLM
            next_question = None
            if next_node and next_node.get("type") == "question":
                next_question = next_node.get("question") or next_node.get("text")
                session.update_protocol_state("current_node", next_id)
            elif next_id and not current_node.get("terminal", False):
                session.update_protocol_state("current_node", next_id)
            # else: terminal node — leave current_node as-is

            return {
                "action": "instruct",
                "message": message,
                "next_question": next_question,
                "rag_tags": current_node.get("rag_tags", []),
                "next_node_id": next_id,
                "completed": current_node.get("terminal", False),
                "critical": current_node.get("critical", False)
            }
            
        elif node_type == "escalate":
            message = (
                current_node.get("message")
                or current_node.get("text")
                or "⚠️ CRITICAL. Act now: protect airway, support breathing, control bleeding, prevent shock. Send someone for any available help."
            )
            return {
                "action": "escalate",
                "message": message,
                "completed": True,
                "critical": True
            }
            
        return self._fallback_response()
        
    def _find_next_node(self, node: Dict, user_response: str) -> Optional[str]:
        """Find next node based on user response matching.
        
        Handles two option formats:
          - Old format (dict): {yes/True: 'node_id', no/False: 'node_id', 'keyword': 'node_id'}
          - New format (list): [{response_match: [...], next: 'node_id'}, ...]
        """
        user_lower = (user_response or "").lower()
        options = node.get("options", [])

        # --- Old format: options is a dict ---
        if isinstance(options, dict):
            yes_words = {"yes", "yeah", "yep", "yup", "correct", "affirmative", "y", "true"}
            no_words  = {"no", "nope", "nah", "negative", "n", "false", "not"}
            first_value = None
            for key, next_node_id in options.items():
                if first_value is None:
                    first_value = next_node_id
                if key is True:
                    if any(w in user_lower.split() for w in yes_words):
                        return next_node_id
                elif key is False:
                    if any(w in user_lower.split() for w in no_words):
                        return next_node_id
                elif isinstance(key, str):
                    if key.lower() in user_lower:
                        return next_node_id
            # No keyword matched — return first option as default
            return first_value

        # --- New format: options is a list of dicts ---
        for option in (options or []):
            if not isinstance(option, dict):
                logger.warning("Skipping non-dict option entry: %r", option)
                continue
            match_keywords = option.get("response_match", []) or []
            if isinstance(match_keywords, str):
                match_keywords = [match_keywords]
            if any(str(k).lower() in user_lower for k in match_keywords if k is not None):
                return option.get("next")

        # Default to first list option if no keyword matched
        if options and isinstance(options, list):
            first = options[0]
            if isinstance(first, dict):
                return first.get("next")

        return None
        
    def _fallback_response(self) -> Dict[str, Any]:
        """Fallback response when tree navigation fails"""
        return {
            "action": "ask",
            "message": "I understand. Can you describe what's happening in more detail?",
            "next_node_id": None,
            "completed": False
        }
