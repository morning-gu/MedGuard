"""
MedGuard: Medical Safety Guardrail

A specialized safety guardrail system for healthcare AI applications.
"""

__version__ = "0.1.0"
__author__ = "MedGuard Team"
__email__ = "medguard@example.org"

from .pipeline import MedGuardPipeline, SafetyResult, TokenRisk, check_safety
from .privacy_ner import PrivacyNER, PHIEntity, anonymize_phi
from .medical_rules import MedicalRuleFilter, RuleResult, quick_check

__all__ = [
    # Main pipeline
    "MedGuardPipeline",
    "SafetyResult",
    "TokenRisk",
    "check_safety",
    
    # Privacy NER
    "PrivacyNER",
    "PHIEntity",
    "anonymize_phi",
    
    # Rule-based filter
    "MedicalRuleFilter",
    "RuleResult",
    "quick_check",
]
