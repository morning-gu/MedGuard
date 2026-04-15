"""
MedGuard: Medical Privacy Entity Recognition

Hybrid approach combining rule-based patterns and neural NER for 
detecting and anonymizing Protected Health Information (PHI).
"""

import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PHIEntity:
    """Represents a detected PHI entity."""
    text: str
    start: int
    end: int
    entity_type: str
    confidence: float
    detection_method: str  # 'rule' or 'ner'


class PrivacyNER:
    """
    Privacy entity recognition for medical text de-identification.
    
    Combines rule-based pattern matching with neural NER for comprehensive
    PHI detection and anonymization.
    """
    
    def __init__(self):
        """Initialize the Privacy NER system."""
        self._compile_patterns()
        self._ner_model = None  # Lazy loading
        
        # PHI entity types
        self.entity_types = [
            "PATIENT_NAME",
            "PHYSICIAN_NAME", 
            "PHONE_NUMBER",
            "EMAIL",
            "DATE_OF_BIRTH",
            "MEDICAL_RECORD_NUMBER",
            "INSURANCE_ID",
            "SOCIAL_SECURITY_NUMBER",
            "ADDRESS",
            "HOSPITAL_NAME"
        ]
        
        # Replacement templates
        self.replacement_templates = {
            "PATIENT_NAME": "[PATIENT REDACTED]",
            "PHYSICIAN_NAME": "[PROVIDER REDACTED]",
            "PHONE_NUMBER": "[PHONE REDACTED]",
            "EMAIL": "[EMAIL REDACTED]",
            "DATE_OF_BIRTH": "[DOB REDACTED]",
            "MEDICAL_RECORD_NUMBER": "[MRN REDACTED]",
            "INSURANCE_ID": "[INSURANCE REDACTED]",
            "SOCIAL_SECURITY_NUMBER": "[SSN REDACTED]",
            "ADDRESS": "[ADDRESS REDACTED]",
            "HOSPITAL_NAME": "[FACILITY REDACTED]"
        }
    
    def _compile_patterns(self):
        """Compile regex patterns for rule-based PHI detection."""
        
        self.patterns = {
            # Phone numbers (various formats)
            "PHONE_NUMBER": [
                re.compile(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b'),
                re.compile(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b'),
            ],
            
            # Email addresses
            "EMAIL": [
                re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            ],
            
            # Social Security Numbers (US)
            "SOCIAL_SECURITY_NUMBER": [
                re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
                re.compile(r'\b\d{3}\s\d{2}\s\d{4}\b'),
            ],
            
            # Medical Record Numbers (common patterns)
            "MEDICAL_RECORD_NUMBER": [
                re.compile(r'\b(?:MRN|MR#|Medical Record)[:\s]*[A-Z0-9]{6,10}\b', re.IGNORECASE),
                re.compile(r'\b[A-Z]{2,3}\d{5,8}\b'),  # e.g., MR123456
            ],
            
            # Dates of Birth (context-aware)
            "DATE_OF_BIRTH": [
                re.compile(r'\b(?:DOB|Date of Birth|Birth Date)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b', re.IGNORECASE),
                re.compile(r'\bborn[:\s]*(\w+\s+\d{1,2},?\s+\d{4})\b', re.IGNORECASE),
            ],
            
            # Insurance IDs
            "INSURANCE_ID": [
                re.compile(r'\b(?:BCBS|AETNA|UHC|CIGNA|HUMANA)[A-Z0-9]{6,12}\b', re.IGNORECASE),
                re.compile(r'\b(?:Policy|Insurance|Member)\s*(?:ID|Number)[:\s]*[A-Z0-9]{8,15}\b', re.IGNORECASE),
            ],
            
            # HIPAA dates (years in the past)
            "DATE": [
                re.compile(r'\b(?:\d{1,2}[/-]){2}\d{2,4}\b'),
                re.compile(r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b', re.IGNORECASE),
            ],
            
            # Ages over 89 (HIPAA safe harbor)
            "AGE": [
                re.compile(r'\b(?:age|aged|yo)\s*:?[\s\-]*(\d{3,})\b', re.IGNORECASE),
                re.compile(r'\b(\d{3,})[\s\-]*(?:year old|yo|years)\b', re.IGNORECASE),
            ],
        }
        
        # Name patterns (require context)
        self.name_patterns = [
            re.compile(r'(?:patient|pt\.?|patient name)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', re.IGNORECASE),
            re.compile(r'(?:doctor|dr\.?|physician|provider)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', re.IGNORECASE),
            re.compile(r'(?:name|named)[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)', re.IGNORECASE),
        ]
    
    def detect(self, text: str) -> List[PHIEntity]:
        """
        Detect PHI entities in text.
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of detected PHIEntity objects
        """
        entities = []
        
        # Step 1: Rule-based detection
        rule_entities = self._rule_detect(text)
        entities.extend(rule_entities)
        
        # Step 2: Neural NER detection (if available)
        ner_entities = self._ner_detect(text)
        entities.extend(ner_entities)
        
        # Step 3: Merge overlapping entities (keep highest confidence)
        entities = self._merge_overlapping(entities)
        
        # Sort by position
        entities.sort(key=lambda e: e.start)
        
        return entities
    
    def _rule_detect(self, text: str) -> List[PHIEntity]:
        """Detect PHI using rule-based patterns."""
        entities = []
        
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    matched_text = match.group(0)
                    
                    # Skip if group capture exists
                    if match.lastindex and match.lastindex >= 1:
                        try:
                            matched_text = match.group(1)
                            start = match.start(1)
                            end = match.end(1)
                        except:
                            start = match.start()
                            end = match.end()
                    else:
                        start = match.start()
                        end = match.end()
                    
                    entities.append(PHIEntity(
                        text=matched_text,
                        start=start,
                        end=end,
                        entity_type=entity_type,
                        confidence=0.85,  # High confidence for rule-based
                        detection_method='rule'
                    ))
        
        # Detect names with context
        for pattern in self.name_patterns:
            for match in pattern.finditer(text):
                name_text = match.group(1)
                
                # Determine entity type based on context
                context = match.group(0).lower()
                if any(word in context for word in ['patient', 'pt']):
                    entity_type = 'PATIENT_NAME'
                elif any(word in context for word in ['doctor', 'dr', 'physician', 'provider']):
                    entity_type = 'PHYSICIAN_NAME'
                else:
                    entity_type = 'PATIENT_NAME'
                
                entities.append(PHIEntity(
                    text=name_text,
                    start=match.start(1),
                    end=match.end(1),
                    entity_type=entity_type,
                    confidence=0.75,
                    detection_method='rule'
                ))
        
        return entities
    
    def _ner_detect(self, text: str) -> List[PHIEntity]:
        """Detect PHI using neural NER model."""
        # Placeholder for neural NER
        # In production: load BioBERT or similar model fine-tuned on i2b2/CLEF
        
        if self._ner_model is None:
            # Lazy loading
            try:
                # from transformers import pipeline
                # self._ner_model = pipeline("ner", model="medguard-ner")
                pass
            except Exception as e:
                logger.warning(f"Could not load NER model: {e}")
                return []
        
        # Simulated NER results (placeholder)
        return []
    
    def _merge_overlapping(self, entities: List[PHIEntity]) -> List[PHIEntity]:
        """Merge overlapping entities, keeping highest confidence."""
        if not entities:
            return []
        
        # Sort by start position, then by confidence (descending)
        entities.sort(key=lambda e: (e.start, -e.confidence))
        
        merged = []
        current = entities[0]
        
        for entity in entities[1:]:
            if entity.start < current.end:
                # Overlapping - keep higher confidence
                if entity.confidence > current.confidence:
                    current = entity
            else:
                # Non-overlapping - add current and move to next
                merged.append(current)
                current = entity
        
        merged.append(current)
        return merged
    
    def anonymize(
        self,
        text: str,
        replacement_map: Optional[Dict[str, str]] = None
    ) -> str:
        """
        Anonymize PHI in text by replacing with redaction markers.
        
        Args:
            text: Input text containing PHI
            replacement_map: Optional custom replacement mapping
            
        Returns:
            Anonymized text with PHI replaced
        """
        entities = self.detect(text)
        
        if not entities:
            return text
        
        # Use custom replacements or defaults
        replacements = replacement_map or self.replacement_templates
        
        # Replace from end to start to preserve positions
        result = text
        for entity in reversed(entities):
            replacement = replacements.get(
                entity.entity_type,
                f"[{entity.entity_type} REDACTED]"
            )
            result = result[:entity.start] + replacement + result[entity.end:]
        
        return result
    
    def get_phi_summary(self, text: str) -> Dict:
        """
        Get summary of PHI detected in text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with PHI statistics
        """
        entities = self.detect(text)
        
        summary = {
            "total_entities": len(entities),
            "entity_counts": {},
            "entity_types_found": [],
            "high_risk": False
        }
        
        for entity in entities:
            count = summary["entity_counts"].get(entity.entity_type, 0)
            summary["entity_counts"][entity.entity_type] = count + 1
            
            if entity.entity_type not in summary["entity_types_found"]:
                summary["entity_types_found"].append(entity.entity_type)
            
            # Mark as high risk if sensitive entities found
            if entity.entity_type in ["SOCIAL_SECURITY_NUMBER", "MEDICAL_RECORD_NUMBER", "DATE_OF_BIRTH"]:
                summary["high_risk"] = True
        
        return summary


# Convenience function
def anonymize_phi(text: str) -> str:
    """
    Quick PHI anonymization.
    
    Args:
        text: Text containing potential PHI
        
    Returns:
        Anonymized text
    """
    ner = PrivacyNER()
    return ner.anonymize(text)
