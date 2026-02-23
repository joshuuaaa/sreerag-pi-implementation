# src/orchestrator.py
"""
Main orchestrator - coordinates all AI components
"""

import time
from typing import Dict, Any, Optional
from src.session.manager import ConversationSession
from src.analyzer.situation_analyzer import SituationAnalyzer
from src.decision.engine import DecisionEngine
from src.llm.engine import LLMEngine
from src.rag.engine import RAGEngine
from src.prompt.styles import build_prompt

class IntelligentOrchestrator:
    def __init__(self, config: dict):
        """
        Initialize orchestrator with all AI components
        
        Args:
            config: Full system configuration
        """
        print("🔄 Initializing AI Orchestrator...")
        
        # Initialize components
        self.llm = LLMEngine(config.get("llm", {}))
        self.rag = RAGEngine(config.get("rag", {}))
        self.decision = DecisionEngine(config.get("decision", {}))
        self.analyzer = SituationAnalyzer()
        
        # Get prompt style
        app_config = config.get("app", {})
        self.prompt_style = app_config.get("prompt_style", "warm")
        self.system_prompt = build_prompt(self.prompt_style)
        
        # Single session (no persistence)
        self.current_session: Optional[ConversationSession] = None
        
        print("✅ Orchestrator ready")
        
    def start_session(self) -> str:
        """
        Start new conversation session
        
        Returns:
            Initial greeting message
        """
        self.current_session = ConversationSession(session_id=str(time.time()))
        return "I'm here to help with emergency situations. Press the button and tell me what's happening."
        
    def reset_session(self):
        """Reset conversation (triple press)"""
        self.current_session = None
        
    def process_message(self, user_message: str) -> Dict[str, Any]:
        """
        Process user message with full situational analysis
        
        Args:
            user_message: User's spoken input
            
        Returns:
            Response dictionary with text, LCD display, state, analysis
        """
        # Create session if needed
        if not self.current_session:
            self.current_session = ConversationSession(session_id=str(time.time()))
            
        session = self.current_session
        
        # STEP 1: Analyze situation
        analysis = self.analyzer.analyze(
            conversation_history=session.history,
            new_message=user_message
        )
        
        print(f"📊 Analysis: {analysis['phase']} | Conditions: {[c['type'] for c in analysis['conditions']]}")
        
        # STEP 2: Determine response based on phase
        if analysis["phase"] == "initial_assessment":
            response = self._handle_initial_assessment(user_message, analysis)
            
        elif analysis["phase"] == "clarification":
            response = self._handle_clarification(user_message, analysis)
            
        elif analysis["phase"] == "condition_confirmation":
            response = self._handle_confirmation(user_message, analysis)
            
        else:  # active_guidance
            response = self._handle_guidance(user_message, analysis)
            
        # STEP 3: Update session
        session.add_exchange(user_message, response["response"])
        session.analysis = analysis
        
        # Add analysis to response
        response["analysis"] = analysis
        
        return response
        
    def _handle_initial_assessment(self, message: str, analysis: Dict) -> Dict[str, Any]:
        """Handle first message - understand situation"""
        
        if not analysis["conditions"]:
            # Unclear situation
            return {
                "response": "I'm here to help. Can you describe what's wrong? Is someone injured?",
                "lcd_display": "Need details",
                "state": "assessing"
            }
        
        # Conditions detected - check critical info
        primary = analysis["primary_condition"]
        
        if "patient_conscious" not in analysis["context"]:
            return {
                "response": f"I understand there's a {primary} situation. First, is the person conscious and breathing?",
                "lcd_display": "Check conscious",
                "state": "assessing"
            }
        
        # Move to confirmation
        return self._handle_confirmation(message, analysis)
        
    def _handle_clarification(self, message: str, analysis: Dict) -> Dict[str, Any]:
        """Ask clarifying questions"""
        
        # Simple clarification
        return {
            "response": "I need more information. What exactly happened? Who is affected?",
            "lcd_display": "Clarifying",
            "state": "assessing"
        }
        
    def _handle_confirmation(self, message: str, analysis: Dict) -> Dict[str, Any]:
        """Confirm conditions and prioritize"""
        
        conditions = analysis["conditions"]
        if not conditions:
            return self._handle_clarification(message, analysis)
            
        primary = conditions[0]
        
        # Check for multiple serious conditions
        if len(conditions) > 1:
            cond1 = conditions[0]["type"]
            cond2 = conditions[1]["type"]
            return {
                "response": f"I see {cond1} and {cond2}. I'll guide you through the {cond1} first. Are you ready?",
                "lcd_display": f"{cond1}+more",
                "state": "confirming"
            }
        
        # Single condition - move to guidance
        return self._handle_guidance(message, analysis)
        
    def _handle_guidance(self, message: str, analysis: Dict) -> Dict[str, Any]:
        """Provide step-by-step guidance"""
        
        primary_condition = analysis["primary_condition"]
        
        if not primary_condition:
            return {
                "response": "Tell me what's happening so I can help you properly.",
                "lcd_display": "Assessing",
                "state": "assessing"
            }
        
        # Check for critical situations
        if self._is_critical(analysis):
            return {
                "response": "This is life-threatening. Call 911 immediately if you haven't already. While waiting, I'll guide you through critical steps.",
                "lcd_display": "CALL 911 NOW",
                "state": "critical"
            }
        
        # Get decision tree guidance
        decision_result = self.decision.navigate(
            emergency_type=primary_condition,
            session=self.current_session,
            user_response=message
        )
        
        # Get RAG context if available
        rag_context = ""
        if decision_result.get("rag_tags"):
            rag_docs = self.rag.retrieve(
                query=f"{primary_condition} {message}",
                tags=decision_result["rag_tags"],
                top_k=2
            )
            if rag_docs:
                rag_context = "\n".join([doc["content"][:200] for doc in rag_docs])
        
        # Build context-aware LLM prompt
        response_text = self._generate_response(
            user_message=message,
            decision_guidance=decision_result["message"],
            rag_context=rag_context,
            analysis=analysis
        )
        
        return {
            "response": response_text,
            "lcd_display": primary_condition[:12],
            "state": "guiding"
        }
        
    def _generate_response(self, user_message: str, decision_guidance: str, 
                          rag_context: str, analysis: Dict) -> str:
        """Generate LLM response with full context"""
        
        # Build conversation history
        history_text = ""
        if self.current_session:
            recent = self.current_session.get_context(last_n=3)
            history_text = "\n".join([
                f"{msg['role'].upper()}: {msg['content']}"
                for msg in recent
            ])
        
        # Build prompt
        primary = analysis.get("primary_condition", "emergency")
        severity = ""
        if analysis.get("conditions"):
            severity = analysis["conditions"][0].get("severity", "")
            
        prompt = f"""{self.system_prompt}

SITUATION:
- Emergency type: {primary}
- Severity: {severity}
- Patient conscious: {analysis['context'].get('patient_conscious', 'unknown')}

RECENT CONVERSATION:
{history_text if history_text else 'First interaction'}

USER JUST SAID: {user_message}

PROTOCOL GUIDANCE:
{decision_guidance}

MEDICAL REFERENCE:
{rag_context if rag_context else 'No additional reference'}

Provide your response (under 40 words, calm and clear):

RESPONSE:"""

        # Generate with LLM
        response = self.llm.generate(prompt, max_tokens=100)
        
        # Fallback if empty
        if not response or len(response.strip()) < 5:
            response = decision_guidance
            
        return response
        
    def _is_critical(self, analysis: Dict) -> bool:
        """Check if situation is immediately life-threatening"""
        
        # Unconscious patient
        if analysis["context"].get("patient_conscious") == False:
            return True
            
        # Critical severity or high priority conditions
        for condition in analysis["conditions"]:
            if condition["severity"] == "critical" and condition["priority"] >= 9:
                return True
                
        # Specific critical conditions
        critical_types = ["choking", "unconscious", "shock"]
        for condition in analysis["conditions"]:
            if condition["type"] in critical_types:
                return True
                
        return False
