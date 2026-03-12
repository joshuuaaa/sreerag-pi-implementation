# src/orchestrator.py
"""
Main orchestrator - coordinates all AI components
"""

import time
import logging
from typing import Dict, Any, Optional
from src.session.manager import ConversationSession
from src.analyzer.situation_analyzer import SituationAnalyzer
from src.decision.engine import DecisionEngine
from src.llm.engine import LLMEngine
from src.rag.engine import RAGEngine
from src.prompt.styles import build_prompt

logger = logging.getLogger("orchestrator")

class IntelligentOrchestrator:
    def __init__(self, config: dict):
        """
        Initialize orchestrator with all AI components
        
        Args:
            config: Full system configuration
        """
        logger.info("Initializing AI Orchestrator")
        
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
        
        logger.info("Orchestrator ready")
        
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
        
        logger.debug(
            "Analysis: phase=%s conditions=%s",
            analysis["phase"],
            [c["type"] for c in analysis["conditions"]],
        )
        
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
        
    def _llm_respond(self, situation: str, task: str, max_tokens: int = 100) -> str:
        """Minimal LLM call with situation context and a task instruction."""
        # Inject any active contraindication warnings even on short LLM calls
        if self.current_session and self.current_session.analysis:
            analysis = self.current_session.analysis
            primary = analysis.get("primary_condition", "")
            secondary_types = [c["type"] for c in analysis.get("conditions", [])[1:]]
            _CONTRAINDICATIONS = {
                ("bleeding", "fracture"): "COMPLICATION: open fracture. Do NOT instruct direct pressure over bone. Gentle pressure around edges only. Immobilise limb.",
                ("bleeding", "chest_injury"): "COMPLICATION: open chest wound. Do NOT apply solid direct pressure — 3-sided seal only.",
                ("bleeding", "head_injury"): "COMPLICATION: head injury. Do NOT tilt head/neck. Pressure around skull only, not over deformity.",
                ("burn", "smoke_inhalation"): "COMPLICATION: smoke inhalation. Airway is absolute priority before burn treatment. Move to fresh air first.",
                ("smoke_inhalation", "burn"): "COMPLICATION: burns present. Treat airway first, cool burn only once breathing stable.",
                ("drowning", "hypothermia"): "COMPLICATION: cold-water drowning. Do NOT stop CPR — continue until rewarmed. Remove wet clothes.",
                ("head_injury", "seizure"): "COMPLICATION: seizure after head injury is serious. Do NOT restrain. Protect head. Recovery position after seizure stops.",
                ("shock", "fracture"): "COMPLICATION: fracture likely causing internal blood loss. Immobilise limb immediately. Do NOT let person stand.",
            }
            for sec in secondary_types:
                warning = _CONTRAINDICATIONS.get((primary, sec))
                if warning:
                    situation = f"\u26a0\ufe0f {warning}\n{situation}"
                    break
        content = f"{situation}\n\n{task}"
        response = self.llm.generate_chat(
            system_prompt=self.system_prompt,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
        )
        return response.strip() if response and len(response.strip()) >= 5 else ""

    def _handle_initial_assessment(self, message: str, analysis: Dict) -> Dict[str, Any]:
        """Handle first message - understand situation"""

        if not analysis["conditions"]:
            resp = self._llm_respond(
                situation=f'Person said: "{message}"\nNo clear emergency detected yet.',
                task="Required question to ask: Can you describe what happened? Is someone injured or unwell?",
            ) or "Can you describe what happened? Is someone injured?"
            return {"response": resp, "lcd_display": "Need details", "state": "assessing"}

        primary  = analysis["primary_condition"]
        severity = analysis["conditions"][0].get("severity", "") if analysis["conditions"] else ""

        if "patient_conscious" not in analysis["context"]:
            resp = self._llm_respond(
                situation=f'Person said: "{message}"\nDetected: {primary} ({severity}).',
                task="Required question to ask: Is the person conscious and breathing?",
            ) or f"Okay — {primary}. Is the person conscious and breathing?"
            return {"response": resp, "lcd_display": "Check conscious", "state": "assessing"}

        return self._handle_confirmation(message, analysis)

    def _handle_clarification(self, message: str, analysis: Dict) -> Dict[str, Any]:
        """Ask clarifying questions"""
        resp = self._llm_respond(
            situation=f'Person said: "{message}"\nSituation unclear.',
            task="Required question to ask: What exactly happened and who is affected?",
        ) or "I need a bit more detail — what exactly happened?"
        return {"response": resp, "lcd_display": "Clarifying", "state": "assessing"}

    def _handle_confirmation(self, message: str, analysis: Dict) -> Dict[str, Any]:
        """Confirm conditions and prioritize"""

        conditions = analysis["conditions"]
        if not conditions:
            return self._handle_clarification(message, analysis)

        if len(conditions) > 1:
            cond1 = conditions[0]["type"]
            cond2 = conditions[1]["type"]
            # If this pair has a known clinical contraindication, skip the
            # "ready?" confirmation and jump straight to guided treatment —
            # that triggers both the decision-tree complication branch and
            # the LLM contraindication warning injection.
            _DANGEROUS_PAIRS = {
                ("bleeding", "fracture"), ("bleeding", "chest_injury"),
                ("bleeding", "head_injury"), ("burn", "smoke_inhalation"),
                ("smoke_inhalation", "burn"), ("drowning", "hypothermia"),
                ("head_injury", "seizure"), ("shock", "fracture"),
            }
            if (cond1, cond2) in _DANGEROUS_PAIRS:
                return self._handle_guidance(message, analysis)
            resp = self._llm_respond(
                situation=f'Person said: "{message}"\nTwo emergencies: {cond1} and {cond2}.',
                task=f"Required question: I'll guide you through {cond1} first — are you ready?",
            ) or f"I can see two issues — {cond1} and {cond2}. I'll take you through {cond1} first. Ready?"
            return {"response": resp, "lcd_display": f"{cond1}+more", "state": "confirming"}

        return self._handle_guidance(message, analysis)
        
    def _handle_guidance(self, message: str, analysis: Dict) -> Dict[str, Any]:
        """Provide step-by-step guidance driven by decision tree, delivered by LLM."""

        primary_condition = analysis["primary_condition"]

        if not primary_condition:
            return {
                "response": "Tell me what's happening so I can help you properly.",
                "lcd_display": "Assessing",
                "state": "assessing"
            }

        # Navigate decision tree — pass secondary conditions so the engine
        # can detect contraindications (e.g. fracture + bleeding = no direct pressure)
        secondary = [c["type"] for c in analysis["conditions"][1:]] if analysis["conditions"] else []
        decision_result = self.decision.navigate(
            emergency_type=primary_condition,
            session=self.current_session,
            user_response=message,
            secondary_conditions=secondary,
        )

        action        = decision_result.get("action", "instruct")
        step_text     = decision_result.get("message", "")
        next_question = decision_result.get("next_question")
        is_critical   = decision_result.get("critical", False) or self._is_critical(analysis)

        # RAG context for instruction nodes
        rag_context = ""
        if decision_result.get("rag_tags"):
            rag_docs = self.rag.retrieve(
                query=f"{primary_condition} {message}",
                tags=decision_result["rag_tags"],
                top_k=2
            )
            if rag_docs:
                rag_context = "\n".join([doc["content"][:200] for doc in rag_docs])

        response_text = self._generate_response(
            user_message=message,
            action=action,
            step_text=step_text,
            next_question=next_question,
            rag_context=rag_context,
            analysis=analysis,
        )

        state = "critical" if is_critical else "guiding"
        lcd   = "CRITICAL" if is_critical else primary_condition[:12]
        return {"response": response_text, "lcd_display": lcd, "state": state}
        
    def _generate_response(self, user_message: str, action: str,
                           step_text: str, next_question: Optional[str],
                           rag_context: str, analysis: Dict) -> str:
        """LLM delivers the decision tree step naturally, grounded in situation analysis."""

        primary   = analysis.get("primary_condition", "emergency")
        severity  = analysis["conditions"][0].get("severity", "") if analysis.get("conditions") else ""
        conscious = analysis["context"].get("patient_conscious", "unknown")

        situation = (
            f'Situation: {primary}, severity: {severity}, patient conscious: {conscious}.\n'
            f'Person just said: "{user_message}"'
        )

        # ── Clinical contraindication warnings ───────────────────────────────
        # Keyed as (primary, secondary) → warning injected into LLM situation.
        # Prevents the LLM from giving advice that is correct in isolation but
        # dangerous when a second condition is also present.
        _CONTRAINDICATIONS = {
            ("bleeding", "fracture"): (
                "⚠️ COMPLICATION: open fracture at wound site. "
                "Do NOT apply direct firm pressure over bone or protruding tissue. "
                "Apply gentle pressure AROUND the wound edges only. "
                "Immobilise the limb in the position found. Do NOT push bone back in."
            ),
            ("bleeding", "chest_injury"): (
                "⚠️ COMPLICATION: chest injury present. "
                "If there is an open chest wound, do NOT seal it with a solid dressing — "
                "use a 3-sided occlusive seal (tape 3 sides only, leave 1 open). "
                "Do NOT apply firm direct pressure to a sucking chest wound."
            ),
            ("bleeding", "head_injury"): (
                "⚠️ COMPLICATION: head injury also present. "
                "Do NOT tilt or move the head or neck to access the wound. "
                "Apply gentle pressure AROUND — not directly over — any skull deformity. "
                "Keep the person completely still."
            ),
            ("bleeding", "fracture"): (  # also covers spinal via fracture tree
                "⚠️ COMPLICATION: possible spinal/fracture injury. "
                "Do NOT reposition the patient to control bleeding. "
                "Apply pressure from the current position only. Keep spine neutral."
            ),
            ("burn", "smoke_inhalation"): (
                "⚠️ COMPLICATION: smoke inhalation present. "
                "Airway is the absolute priority — check for singed nose hairs, "
                "hoarse voice, or stridor BEFORE treating the burn. "
                "Move to fresh air first if not already done."
            ),
            ("smoke_inhalation", "burn"): (
                "⚠️ COMPLICATION: burns also present. "
                "Treat the airway first. Only cool burns once the person is in "
                "fresh air and breathing is stable."
            ),
            ("drowning", "hypothermia"): (
                "⚠️ COMPLICATION: hypothermia with drowning. "
                "Do NOT assume death — cold-water drowning victims can survive "
                "prolonged submersion. Continue CPR until the patient is rewarmed. "
                "Remove wet clothing and insulate while continuing CPR."
            ),
            ("head_injury", "seizure"): (
                "⚠️ COMPLICATION: seizure following head injury — this is serious. "
                "Do NOT restrain seizure movements. Protect the head from further impact. "
                "Keep the airway open; place in recovery position after seizure stops."
            ),
            ("shock", "fracture"): (
                "⚠️ COMPLICATION: fracture (possible femur) may be causing internal "
                "blood loss driving the shock. Immobilise the fractured limb immediately "
                "to reduce internal bleeding. Do NOT let the person stand or walk."
            ),
            ("cpr", "fracture"): (
                "⚠️ NOTE: rib fractures may occur during CPR — this is expected and "
                "acceptable. Do NOT stop CPR because of cracking sounds or suspected "
                "rib fracture. Continue compressions at full depth."
            ),
        }

        secondary_types = [c["type"] for c in analysis.get("conditions", [])[1:]]
        for sec in secondary_types:
            warning = _CONTRAINDICATIONS.get((primary, sec))
            if warning:
                situation += f"\n{warning}"
                break  # one warning per turn keeps the prompt focused

        if rag_context:
            situation += f"\nMedical reference (use only if helpful): {rag_context[:300]}"

        # Tell LLM exactly what to deliver this turn
        if action == "ask":
            task = (
                f"Acknowledge what they said briefly, then ask this question naturally:\n"
                f"{step_text}"
            )
        else:  # instruct / action / escalate
            task = (
                f"Acknowledge what they said briefly, then give this instruction naturally:\n"
                f"{step_text}"
            )
            if next_question:
                task += (
                    f"\nThen end your response by asking this follow-up question naturally:\n"
                    f"{next_question}"
                )

        # Include last 2 turns for continuity (fewer turns → less multi-turn drift)
        messages = []
        if self.current_session:
            for msg in self.current_session.get_context(last_n=2):
                messages.append({"role": msg["role"], "content": msg["content"]})

        content = f"{situation}\n\n{task}"
        if messages and messages[-1]["role"] == "user":
            messages[-1]["content"] = content
        else:
            messages.append({"role": "user", "content": content})

        response = self.llm.generate_chat(
            system_prompt=self.system_prompt,
            messages=messages,
            max_tokens=75,
        )

        if not response or len(response.strip()) < 5:
            # Fallback: raw step + question
            response = step_text
            if next_question:
                response += f" {next_question}"

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
