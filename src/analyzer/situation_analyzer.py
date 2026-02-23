# src/analyzer/situation_analyzer.py
"""
Situation analysis and condition detection
"""

from typing import List, Dict, Any

class SituationAnalyzer:
    def __init__(self):
        """Initialize situation analyzer with condition patterns"""
        self.condition_patterns = {
            "bleeding": {
                "keywords": ["bleed", "blood", "cut", "wound", "gash", "laceration"],
                "severity_indicators": {
                    "critical": ["spurting", "won't stop", "pool", "soaked", "gushing"],
                    "serious": ["lots", "heavy", "deep", "flowing"],
                    "moderate": ["steady", "medium"],
                    "minor": ["small", "little", "scratch", "oozing"]
                },
                "priority": 9
            },
            "fracture": {
                "keywords": ["broken", "fracture", "bone", "snap", "can't move", "deformed"],
                "severity_indicators": {
                    "critical": ["compound", "bone sticking", "white bone", "exposed"],
                    "serious": ["deformed", "swollen", "intense pain", "can't bear weight"],
                    "moderate": ["painful", "hurts to move"]
                },
                "priority": 6
            },
            "burn": {
                "keywords": ["burn", "burned", "fire", "hot", "scald", "blister"],
                "severity_indicators": {
                    "critical": ["charred", "white", "large area", "face", "airway"],
                    "serious": ["blistering", "deep", "red raw"],
                    "moderate": ["red", "painful"]
                },
                "priority": 7
            },
            "choking": {
                "keywords": ["choking", "can't breathe", "throat", "swallowed", "gagging"],
                "priority": 10
            },
            "shock": {
                "keywords": ["pale", "cold", "clammy", "weak pulse", "dizzy", "confused"],
                "severity_indicators": {
                    "critical": ["unconscious", "no pulse", "blue lips"],
                    "serious": ["very pale", "rapid breathing", "weak"]
                },
                "priority": 10
            },
            "unconscious": {
                "keywords": ["unconscious", "passed out", "unresponsive", "not breathing"],
                "priority": 10
            }
        }
        
    def analyze(self, conversation_history: List[Dict], new_message: str) -> Dict[str, Any]:
        """
        Analyze situation from conversation
        
        Args:
            conversation_history: List of previous messages
            new_message: Current message
            
        Returns:
            Analysis dictionary with conditions, phase, context
        """
        # Combine all conversation text
        all_text = " ".join([msg["content"] for msg in conversation_history])
        all_text += " " + new_message
        all_text_lower = all_text.lower()
        
        # Detect conditions
        detected_conditions = []
        for condition_type, config in self.condition_patterns.items():
            if self._matches_keywords(all_text_lower, config["keywords"]):
                severity = self._assess_severity(all_text_lower, config.get("severity_indicators", {}))
                detected_conditions.append({
                    "type": condition_type,
                    "severity": severity,
                    "priority": config["priority"],
                    "confidence": 0.8
                })
        
        # Sort by priority (highest first)
        detected_conditions.sort(key=lambda x: x["priority"], reverse=True)
        
        # Extract context from conversation
        context = self._extract_context(all_text_lower, conversation_history)
        
        # Determine conversation phase
        turn_count = len(conversation_history) // 2
        phase = self._determine_phase(turn_count, detected_conditions, context)
        
        return {
            "conditions": detected_conditions,
            "primary_condition": detected_conditions[0]["type"] if detected_conditions else None,
            "context": context,
            "phase": phase,
            "turn_count": turn_count
        }
        
    def _matches_keywords(self, text: str, keywords: List[str]) -> bool:
        """Check if any keyword matches text"""
        return any(keyword in text for keyword in keywords)
        
    def _assess_severity(self, text: str, severity_indicators: Dict) -> str:
        """Determine severity based on indicators"""
        if not severity_indicators:
            return "unknown"
            
        for severity, keywords in severity_indicators.items():
            if any(keyword in text for keyword in keywords):
                return severity
        return "moderate"
        
    def _extract_context(self, text: str, history: List[Dict]) -> Dict[str, Any]:
        """Extract contextual information"""
        context = {}
        
        # Patient consciousness
        if any(word in text for word in ["conscious", "awake", "responsive", "talking", "alert"]):
            context["patient_conscious"] = True
        elif any(word in text for word in ["unconscious", "unresponsive", "passed out", "not breathing"]):
            context["patient_conscious"] = False
            
        # Body location
        body_parts = ["head", "arm", "leg", "chest", "back", "hand", "foot", "neck", "face", "stomach"]
        for part in body_parts:
            if part in text:
                context["location"] = part
                break
                
        # Time indicators
        if any(word in text for word in ["just happened", "just now", "now"]):
            context["timing"] = "immediate"
        elif any(word in text for word in ["minutes ago", "few minutes"]):
            context["timing"] = "recent"
            
        return context
        
    def _determine_phase(self, turn_count: int, conditions: List[Dict], context: Dict) -> str:
        """Determine conversation phase"""
        if turn_count == 0:
            return "initial_assessment"
        elif turn_count == 1 and not conditions:
            return "clarification"
        elif turn_count <= 2 and conditions:
            return "condition_confirmation"
        else:
            return "active_guidance"
