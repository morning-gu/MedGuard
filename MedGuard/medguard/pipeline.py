"""
MedGuard: Medical Safety Guardrail Pipeline

Core pipeline implementation for medical safety checking with token-level attribution.
"""

from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class TokenRisk:
    """Represents risk associated with a specific token."""
    token: str
    position: List[int]  # [start, end] character positions
    risk_score: float
    reason: str
    category: str


@dataclass
class SafetyResult:
    """Result of safety check."""
    is_blocked: bool
    risk_score: float
    reason: Optional[str]
    suggested_response: Optional[str]
    token_risks: List[TokenRisk]
    category: Optional[str] = None
    severity: str = "low"  # low, medium, high, critical
    
    def to_dict(self) -> Dict:
        return {
            "is_blocked": self.is_blocked,
            "risk_score": self.risk_score,
            "reason": self.reason,
            "suggested_response": self.suggested_response,
            "token_risks": [
                {
                    "token": tr.token,
                    "position": tr.position,
                    "risk_score": tr.risk_score,
                    "reason": tr.reason,
                    "category": tr.category
                }
                for tr in self.token_risks
            ],
            "category": self.category,
            "severity": self.severity
        }


class MedGuardPipeline:
    """
    Main pipeline for medical safety checking.
    
    Combines rule-based filtering, semantic classification, and token-level
    risk attribution for comprehensive medical safety analysis.
    """
    
    def __init__(
        self,
        model_name: str = "medguard-7b-qlora",
        device: str = "cuda",
        quantization: str = "int4",
        risk_threshold: float = 0.7,
        include_attribution: bool = True
    ):
        """
        Initialize the MedGuard pipeline.
        
        Args:
            model_name: Name of the model to use
            device: Device to run inference on ('cuda' or 'cpu')
            quantization: Quantization strategy ('int4', 'int8', 'fp16')
            risk_threshold: Threshold above which content is blocked
            include_attribution: Whether to include token-level attribution
        """
        self.model_name = model_name
        self.device = device
        self.quantization = quantization
        self.risk_threshold = risk_threshold
        self.include_attribution = include_attribution
        
        # Components initialized lazily
        self._model = None
        self._tokenizer = None
        self._rule_filter = None
        self._privacy_ner = None
        
        logger.info(f"Initialized MedGuardPipeline with model={model_name}")
    
    def _load_model(self):
        """Load model and tokenizer (lazy initialization)."""
        if self._model is None:
            # Placeholder for actual model loading
            # In production, this would load from HuggingFace or local weights
            logger.info(f"Loading model {self.model_name}...")
            # self._model = AutoModelForCausalLM.from_pretrained(...)
            # self._tokenizer = AutoTokenizer.from_pretrained(...)
            pass
    
    def _load_components(self):
        """Load supporting components."""
        if self._rule_filter is None:
            from .medical_rules import MedicalRuleFilter
            self._rule_filter = MedicalRuleFilter()
        
        if self._privacy_ner is None:
            from .privacy_ner import PrivacyNER
            self._privacy_ner = PrivacyNER()
    
    def check(
        self,
        text: str,
        include_attribution: Optional[bool] = None
    ) -> SafetyResult:
        """
        Check text for medical safety risks.
        
        Args:
            text: Input text to analyze
            include_attribution: Override default attribution setting
            
        Returns:
            SafetyResult with risk assessment
        """
        if include_attribution is None:
            include_attribution = self.include_attribution
        
        # Load components if needed
        self._load_model()
        self._load_components()
        
        # Step 1: Rule-based filtering (fast path)
        rule_result = self._rule_filter.check(text)
        if rule_result.is_blocked:
            return SafetyResult(
                is_blocked=True,
                risk_score=rule_result.risk_score,
                reason=rule_result.reason,
                suggested_response=self._get_safe_response(rule_result.category),
                token_risks=[],
                category=rule_result.category,
                severity=rule_result.severity
            )
        
        # Step 2: Privacy check
        privacy_entities = self._privacy_ner.detect(text)
        if privacy_entities:
            # De-identify and continue processing
            text_clean = self._privacy_ner.anonymize(text)
        else:
            text_clean = text
        
        # Step 3: Semantic classification (model-based)
        semantic_result = self._semantic_check(text_clean)
        
        # Step 4: Token-level attribution (if enabled and risk detected)
        token_risks = []
        if include_attribution and semantic_result.risk_score > 0.3:
            token_risks = self._compute_token_attribution(text_clean, semantic_result)
        
        # Determine if blocked
        is_blocked = semantic_result.risk_score >= self.risk_threshold
        
        # Map risk score to severity
        if semantic_result.risk_score >= 0.9:
            severity = "critical"
        elif semantic_result.risk_score >= 0.75:
            severity = "high"
        elif semantic_result.risk_score >= 0.5:
            severity = "medium"
        else:
            severity = "low"
        
        return SafetyResult(
            is_blocked=is_blocked,
            risk_score=semantic_result.risk_score,
            reason=semantic_result.reason if is_blocked else None,
            suggested_response=self._get_safe_response(semantic_result.category) if is_blocked else None,
            token_risks=token_risks,
            category=semantic_result.category,
            severity=severity
        )
    
    def _semantic_check(self, text: str) -> SafetyResult:
        """
        Perform semantic safety classification using the model.
        
        This is a placeholder - in production, this would call the fine-tuned model.
        """
        # Placeholder implementation
        # In production: run model inference, get risk scores
        
        # Simulated result for demonstration
        risk_keywords = [
            ("overdose", 0.85, "medication_safety"),
            ("suicide", 0.92, "mental_health"),
            ("drug interaction", 0.78, "drug_interaction"),
            ("contraindicated", 0.72, "contraindication"),
        ]
        
        text_lower = text.lower()
        max_risk = 0.0
        category = None
        reason = None
        
        for keyword, risk, cat in risk_keywords:
            if keyword in text_lower:
                if risk > max_risk:
                    max_risk = risk
                    category = cat
                    reason = f"Detected potential {cat.replace('_', ' ')} risk"
        
        return SafetyResult(
            is_blocked=max_risk >= self.risk_threshold,
            risk_score=max_risk,
            reason=reason,
            suggested_response=None,
            token_risks=[],
            category=category
        )
    
    def _compute_token_attribution(
        self,
        text: str,
        semantic_result: SafetyResult
    ) -> List[TokenRisk]:
        """
        Compute token-level risk attribution.
        
        Uses gradient saliency and attention rollout to identify risky tokens.
        """
        # Placeholder implementation
        # In production: compute gradients and attention weights
        
        token_risks = []
        
        # Simple heuristic for demonstration
        risk_terms = {
            "aspirin": ("NSAID medication", "medication_safety"),
            "warfarin": ("anticoagulant", "drug_interaction"),
            "pregnant": ("special population", "contraindication"),
            "child": ("pediatric patient", "contraindication"),
            "overdose": ("excessive dosage", "medication_safety"),
        }
        
        words = text.split()
        pos = 0
        for word in words:
            word_lower = word.lower().strip(".,!?;:")
            if word_lower in risk_terms:
                reason, category = risk_terms[word_lower]
                token_risks.append(TokenRisk(
                    token=word,
                    position=[pos, pos + len(word)],
                    risk_score=0.75,  # Simplified
                    reason=reason,
                    category=category
                ))
            pos += len(word) + 1
        
        return token_risks
    
    def _get_safe_response(self, category: Optional[str]) -> str:
        """Generate safe response template based on risk category."""
        
        templates = {
            "medication_safety": (
                "⚠️ Medication Safety Notice: This query involves potential medication risks. "
                "Please consult a qualified healthcare professional or pharmacist for personalized advice. "
                "Do not make medication changes without medical supervision."
            ),
            "drug_interaction": (
                "⚠️ Drug Interaction Alert: Potential drug interaction detected. "
                "Combining medications can have serious consequences. Please consult your doctor or pharmacist "
                "before taking multiple medications together."
            ),
            "contraindication": (
                "⚠️ Contraindication Warning: This medication or treatment may not be safe for your situation. "
                "Certain conditions, age groups, or circumstances can make treatments dangerous. "
                "Please seek professional medical evaluation."
            ),
            "mental_health": (
                "🆘 Mental Health Support: Your message suggests you may be experiencing distress. "
                "Please reach out to a mental health professional or crisis hotline immediately. "
                "You are not alone - help is available."
            ),
            "privacy": (
                "🔒 Privacy Notice: Your message contains personal information. "
                "For your privacy and security, please avoid sharing identifiable health information. "
                "Consult healthcare providers through secure, official channels."
            ),
            None: (
                "⚠️ Safety Notice: This query requires professional medical judgment. "
                "AI systems cannot replace licensed healthcare providers. "
                "Please consult a qualified medical professional for accurate diagnosis and treatment."
            )
        }
        
        return templates.get(category, templates[None])
    
    def batch_check(
        self,
        texts: List[str],
        batch_size: int = 8
    ) -> List[SafetyResult]:
        """
        Check multiple texts for safety risks.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for processing
            
        Returns:
            List of SafetyResult objects
        """
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            for text in batch:
                result = self.check(text)
                results.append(result)
        return results


# Convenience function for quick usage
def check_safety(text: str, threshold: float = 0.7) -> SafetyResult:
    """
    Quick safety check with default settings.
    
    Args:
        text: Text to check
        threshold: Risk threshold for blocking
        
    Returns:
        SafetyResult
    """
    guard = MedGuardPipeline(risk_threshold=threshold)
    return guard.check(text)
