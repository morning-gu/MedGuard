"""
MedGuard: Medical Rule-Based Filter

Fast rule-based filtering for common medical safety patterns.
Provides quick rejection of obvious safety violations before model inference.
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import re
import logging

logger = logging.getLogger(__name__)


@dataclass
class RuleResult:
    """Result of rule-based check."""
    is_blocked: bool
    risk_score: float
    reason: Optional[str]
    category: Optional[str]
    severity: str
    matched_rules: List[str]


class MedicalRuleFilter:
    """
    Rule-based filter for medical safety checking.
    
    Provides fast, deterministic filtering for well-known safety patterns
    before invoking more expensive model-based classification.
    """
    
    def __init__(self):
        """Initialize the rule-based filter."""
        self._compile_patterns()
        
        # Risk scores for different categories
        self.category_risk_scores = {
            "medication_safety": 0.75,
            "drug_interaction": 0.80,
            "contraindication": 0.85,
            "dosage_error": 0.90,
            "mental_health": 0.95,
            "emergency": 1.0,
            "privacy": 0.70,
        }
        
        # Severity mapping
        self.severity_map = {
            "emergency": "critical",
            "mental_health": "critical",
            "dosage_error": "high",
            "contraindication": "high",
            "drug_interaction": "medium",
            "medication_safety": "medium",
            "privacy": "low",
        }
    
    def _compile_patterns(self):
        """Compile regex patterns for rule matching."""
        
        # Emergency/crisis patterns
        self.emergency_patterns = [
            re.compile(r'\b(?:kill\s+myself|suicide|want\s+to\s+die|end\s+my\s+life)\b', re.IGNORECASE),
            re.compile(r'\b(?:overdose|intentionally\s+take\s+too\s+many)\b', re.IGNORECASE),
            re.compile(r'\b(?:chest\s+pain|can\'?t\s+breathe|heart\s+attack|stroke\s+symptoms)\b', re.IGNORECASE),
        ]
        
        # Medication safety patterns
        self.medication_patterns = [
            re.compile(r'\b(?:take\s+\d+\s+(?:pills?|tablets?|capsules?)\s+(?:per|a)\s+day)\b', re.IGNORECASE),
            re.compile(r'\b(?:double\s+(?:the\s+)?dose|twice\s+the\s+recommended)\b', re.IGNORECASE),
            re.compile(r'\b(?:mix|combine|take\s+with)\s+.*\b(?:alcohol|wine|beer)\b', re.IGNORECASE),
        ]
        
        # Drug interaction keywords
        self.interaction_keywords = [
            "drug interaction", "interact with", "combine with", "take together",
            "adverse reaction", "side effect", "contraindicated"
        ]
        
        # High-risk medication combinations
        self.dangerous_combinations = [
            ("warfarin", "aspirin"),
            ("warfarin", "ibuprofen"),
            ("lisinopril", "potassium"),
            ("metformin", "alcohol"),
            ("ssri", "maoi"),
            ("benzodiazepine", "opioid"),
            ("statin", "grapefruit"),
        ]
        
        # Contraindication patterns
        self.contraindication_patterns = [
            re.compile(r'\b(?:pregnant|pregnancy|breastfeeding|breast-feeding)\b', re.IGNORECASE),
            re.compile(r'\b(?:child|children|pediatric|under\s+18|minor)\b', re.IGNORECASE),
            re.compile(r'\b(?:elderly|senior|over\s+65|kidney\s+disease|liver\s+disease)\b', re.IGNORECASE),
        ]
        
        # Dosage error patterns
        self.dosage_patterns = [
            re.compile(r'\b(?:\d{3,}\s*(?:mg|ml|tablet|pill))\b', re.IGNORECASE),  # Very high doses
            re.compile(r'\b(?:every\s+\d+\s*hours?)\b', re.IGNORECASE),  # Frequent dosing
            re.compile(r'\b(?:crush|snort|inject|dissolve)\b', re.IGNORECASE),  # Abuse routes
        ]
        
        # Privacy violation patterns
        self.privacy_patterns = [
            re.compile(r'\b(?:ssn|social\s+security)[:\s]*\d', re.IGNORECASE),
            re.compile(r'\b(?:mrn|medical\s+record)[:\s]*[A-Z0-9]', re.IGNORECASE),
            re.compile(r'\b(?:dob|date\s+of\s+birth)[:\s]*\d', re.IGNORECASE),
        ]
    
    def check(self, text: str) -> RuleResult:
        """
        Check text against rule-based patterns.
        
        Args:
            text: Input text to analyze
            
        Returns:
            RuleResult with blocking decision
        """
        matched_rules = []
        highest_risk = 0.0
        category = None
        reason = None
        
        # Check emergency patterns (highest priority)
        for i, pattern in enumerate(self.emergency_patterns):
            if pattern.search(text):
                matched_rules.append(f"emergency_{i}")
                if self.category_risk_scores["emergency"] > highest_risk:
                    highest_risk = self.category_risk_scores["emergency"]
                    category = "emergency"
                    reason = "Detected potential emergency or crisis situation"
        
        # Check mental health crisis
        mental_health_phrases = ["suicide", "kill myself", "want to die", "end my life"]
        for phrase in mental_health_phrases:
            if phrase.lower() in text.lower():
                matched_rules.append("mental_health_crisis")
                if self.category_risk_scores["mental_health"] > highest_risk:
                    highest_risk = self.category_risk_scores["mental_health"]
                    category = "mental_health"
                    reason = "Detected mental health crisis indicators"
        
        # Check medication safety
        for i, pattern in enumerate(self.medication_patterns):
            if pattern.search(text):
                matched_rules.append(f"medication_{i}")
                if self.category_risk_scores["medication_safety"] > highest_risk:
                    highest_risk = self.category_risk_scores["medication_safety"]
                    category = "medication_safety"
                    reason = "Detected potential medication safety issue"
        
        # Check dangerous drug combinations
        text_lower = text.lower()
        for drug1, drug2 in self.dangerous_combinations:
            if drug1.lower() in text_lower and drug2.lower() in text_lower:
                matched_rules.append(f"interaction_{drug1}_{drug2}")
                if self.category_risk_scores["drug_interaction"] > highest_risk:
                    highest_risk = self.category_risk_scores["drug_interaction"]
                    category = "drug_interaction"
                    reason = f"Detected potentially dangerous combination: {drug1} + {drug2}"
        
        # Check contraindications
        for i, pattern in enumerate(self.contraindication_patterns):
            if pattern.search(text):
                matched_rules.append(f"contraindication_{i}")
                if self.category_risk_scores["contraindication"] > highest_risk:
                    highest_risk = self.category_risk_scores["contraindication"]
                    category = "contraindication"
                    reason = "Detected potential contraindication scenario"
        
        # Check dosage errors
        for i, pattern in enumerate(self.dosage_patterns):
            if pattern.search(text):
                matched_rules.append(f"dosage_{i}")
                if self.category_risk_scores["dosage_error"] > highest_risk:
                    highest_risk = self.category_risk_scores["dosage_error"]
                    category = "dosage_error"
                    reason = "Detected potential dosage error or abuse pattern"
        
        # Check privacy violations
        for i, pattern in enumerate(self.privacy_patterns):
            if pattern.search(text):
                matched_rules.append(f"privacy_{i}")
                if self.category_risk_scores["privacy"] > highest_risk:
                    highest_risk = self.category_risk_scores["privacy"]
                    category = "privacy"
                    reason = "Detected potential privacy violation (PHI in text)"
        
        # Determine if blocked
        is_blocked = highest_risk >= 0.7  # Threshold for automatic blocking
        
        # Get severity
        severity = self.severity_map.get(category, "low")
        
        return RuleResult(
            is_blocked=is_blocked,
            risk_score=highest_risk,
            reason=reason,
            category=category,
            severity=severity,
            matched_rules=matched_rules
        )
    
    def get_rule_explanation(self, rule_name: str) -> str:
        """Get human-readable explanation for a matched rule."""
        
        explanations = {
            "emergency_0": "Crisis/self-harm language detected",
            "emergency_1": "Overdose intent detected",
            "emergency_2": "Medical emergency symptoms detected",
            "mental_health_crisis": "Mental health crisis indicators",
            "medication_0": "High-frequency dosing pattern",
            "medication_1": "Dose escalation pattern",
            "medication_2": "Mixing medication with alcohol",
            "dosage_0": "Extremely high dose detected",
            "dosage_1": "Frequent dosing schedule",
            "dosage_2": "Non-standard administration route",
            "privacy_0": "Social Security Number pattern",
            "privacy_1": "Medical Record Number pattern",
            "privacy_2": "Date of Birth pattern",
        }
        
        return explanations.get(rule_name, f"Rule matched: {rule_name}")


# Convenience function
def quick_check(text: str) -> RuleResult:
    """
    Quick rule-based safety check.
    
    Args:
        text: Text to check
        
    Returns:
        RuleResult
    """
    filter = MedicalRuleFilter()
    return filter.check(text)
