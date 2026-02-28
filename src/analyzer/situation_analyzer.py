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
            },

            # ── disaster / austere-environment scenarios ─────────────────
            "hypothermia": {
                "keywords": ["hypothermia", "very cold", "freezing", "shivering", "cold exposure", "wet and cold", "frostbite"],
                "severity_indicators": {
                    "critical": ["not shivering", "confused", "drowsy", "unconscious", "stumbling", "slurred"],
                    "serious": ["violent shivering", "can't stop shivering", "numb", "clumsy"],
                    "moderate": ["shivering", "cold", "chilled"],
                },
                "priority": 8,
            },
            "heat_illness": {
                "keywords": ["heatstroke", "heat stroke", "heat exhaustion", "overheating", "too hot", "sun", "hot and dizzy", "no sweating"],
                "severity_indicators": {
                    "critical": ["confused", "seizure", "passed out", "unconscious", "hot dry", "not sweating"],
                    "serious": ["vomiting", "very weak", "faint", "headache"],
                    "moderate": ["dizzy", "nausea", "cramps", "thirst"],
                },
                "priority": 8,
            },
            "dehydration": {
                "keywords": ["dehydration", "no water", "thirsty", "dry mouth", "dark urine", "not peeing", "diarrhea"],
                "severity_indicators": {
                    "critical": ["confused", "unconscious", "can't keep fluids down", "not peeing all day"],
                    "serious": ["very dizzy", "very weak", "sunken", "rapid heartbeat"],
                    "moderate": ["thirst", "dry", "headache"],
                },
                "priority": 6,
            },
            "smoke_inhalation": {
                "keywords": ["smoke inhalation", "smoke", "fire", "soot", "burning", "coughing", "wheezing", "carbon monoxide"],
                "severity_indicators": {
                    "critical": ["can't breathe", "blue", "confused", "passed out", "unconscious"],
                    "serious": ["wheezing", "hoarse", "burned nose", "soot in mouth"],
                    "moderate": ["cough", "sore throat", "headache"],
                },
                "priority": 9,
            },
            "seizure": {
                "keywords": ["seizure", "seizing", "convulsion", "shaking", "fit", "epilepsy"],
                "severity_indicators": {
                    "critical": ["more than 5", "won't stop", "repeated", "not waking", "injured"],
                    "serious": ["first seizure", "pregnant", "diabetes", "water"],
                    "moderate": ["shaking", "jerking"],
                },
                "priority": 9,
            },

            # ── trauma and injury scenarios ──────────────────────────────
            "head_injury": {
                "keywords": ["head injury", "hit head", "head wound", "skull", "concussion", "head trauma", "head laceration", "scalp"],
                "severity_indicators": {
                    "critical": ["unconscious", "not waking", "unequal pupils", "fluid from ear", "fluid from nose", "repeated vomiting", "confusion getting worse", "seizure", "skull deformity"],
                    "serious": ["confused", "vomiting", "headache", "dazed", "dizzy"],
                    "moderate": ["bump", "bruise"],
                },
                "priority": 9,
            },
            "drowning": {
                "keywords": ["drowning", "drowned", "nearly drowned", "pulled from water", "water rescue", "underwater"],
                "severity_indicators": {
                    "critical": ["not breathing", "unconscious", "blue", "no pulse"],
                    "serious": ["coughing", "gasping", "confused", "barely conscious"],
                    "moderate": ["swallowed water"],
                },
                "priority": 10,
            },
            "chest_injury": {
                "keywords": ["chest injury", "chest wound", "chest pain", "rib", "sucking chest", "open chest", "pneumothorax", "chest hit"],
                "severity_indicators": {
                    "critical": ["can't breathe", "blue", "sucking wound", "trachea shifted", "gurgling", "not breathing"],
                    "serious": ["severe chest pain", "breathing difficulty", "coughing blood"],
                    "moderate": ["rib pain", "sharp chest pain"],
                },
                "priority": 9,
            },
            "amputation": {
                "keywords": ["amputation", "amputated", "limb cut off", "finger cut off", "hand cut off", "arm cut off", "leg cut off", "severed"],
                "severity_indicators": {
                    "critical": ["spurting", "won't stop", "pool of blood", "shock"],
                    "serious": ["heavy bleeding", "tourniquet"],
                    "moderate": ["controlled"],
                },
                "priority": 10,
            },
            "snake_bite": {
                "keywords": ["snake bite", "snakebite", "bitten by snake", "snake attack", "viper", "cobra", "adder"],
                "severity_indicators": {
                    "critical": ["can't breathe", "collapsing", "paralysis", "blurred vision", "drooping eyelids"],
                    "serious": ["swelling", "pain", "nausea", "weakness"],
                    "moderate": ["bite", "small swelling"],
                },
                "priority": 9,
            },
            "poisoning": {
                "keywords": ["poison", "poisoned", "overdose", "ingested", "swallowed chemical", "toxic", "carbon monoxide", "fumes", "contaminated"],
                "severity_indicators": {
                    "critical": ["unconscious", "not breathing", "seizure", "blue"],
                    "serious": ["vomiting", "confused", "difficulty breathing", "severe pain"],
                    "moderate": ["nausea", "dizziness"],
                },
                "priority": 9,
            },
            "postpartum_haemorrhage": {
                "keywords": ["postpartum", "after childbirth", "after delivery", "after birth", "bleeding after baby", "uterus", "postnatal bleeding", "gave birth"],
                "severity_indicators": {
                    "critical": ["soaking", "heavy bleeding", "pale", "dizzy", "collapsing", "shock"],
                    "serious": ["lots of blood", "pad soaked", "bleeding won't stop"],
                    "moderate": ["some bleeding", "more than usual"],
                },
                "priority": 10,
            },
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
