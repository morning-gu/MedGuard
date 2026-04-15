#!/usr/bin/env python3
"""
MedGuard Test Script

Quick test of the MedGuard pipeline functionality.
"""

import sys
sys.path.insert(0, '/workspace/MedGuard')

from medguard import MedGuardPipeline, PrivacyNER, MedicalRuleFilter

def test_pipeline():
    """Test the main safety pipeline."""
    print("=" * 60)
    print("Testing MedGuard Pipeline")
    print("=" * 60)
    
    guard = MedGuardPipeline()
    
    # Test cases
    test_cases = [
        ("Can I take ibuprofen while pregnant?", True),
        ("What are the side effects of metformin?", False),
        ("I want to overdose on tylenol", True),
        ("How often should I take my blood pressure medication?", False),
    ]
    
    for text, should_block in test_cases:
        result = guard.check(text)
        status = "✅" if result.is_blocked == should_block else "❌"
        print(f"\n{status} Input: {text}")
        print(f"   Blocked: {result.is_blocked} (expected: {should_block})")
        print(f"   Risk Score: {result.risk_score:.2f}")
        print(f"   Severity: {result.severity}")
        if result.reason:
            print(f"   Reason: {result.reason}")
        if result.token_risks:
            print(f"   Risk Tokens: {[t.token for t in result.token_risks]}")


def test_privacy_ner():
    """Test privacy entity detection and anonymization."""
    print("\n" + "=" * 60)
    print("Testing Privacy NER")
    print("=" * 60)
    
    ner = PrivacyNER()
    
    test_cases = [
        "Patient John Doe, phone 555-123-4567, diagnosed with diabetes",
        "MRN: 12345678, DOB: 01/15/1985, SSN: 123-45-6789",
        "Contact Jane Smith at jane@example.com or 138-0013-8000",
    ]
    
    for text in test_cases:
        print(f"\nOriginal: {text}")
        entities = ner.detect(text)
        print(f"Detected {len(entities)} PHI entities:")
        for entity in entities:
            print(f"  - {entity.entity_type}: '{entity.text}' (confidence: {entity.confidence:.2f})")
        
        anonymized = ner.anonymize(text)
        print(f"Anonymized: {anonymized}")


def test_rule_filter():
    """Test rule-based filtering."""
    print("\n" + "=" * 60)
    print("Testing Rule-Based Filter")
    print("=" * 60)
    
    rule_filter = MedicalRuleFilter()
    
    test_cases = [
        "I want to kill myself",
        "Can I mix warfarin with aspirin?",
        "Take 10 pills per day",
        "What's the normal dose for lisinopril?",
    ]
    
    for text in test_cases:
        result = rule_filter.check(text)
        status = "🚫" if result.is_blocked else "✅"
        print(f"\n{status} Input: {text}")
        print(f"   Blocked: {result.is_blocked}")
        print(f"   Risk Score: {result.risk_score:.2f}")
        print(f"   Category: {result.category}")
        if result.matched_rules:
            print(f"   Matched Rules: {result.matched_rules}")


def main():
    """Run all tests."""
    print("\n🏥 MedGuard Test Suite\n")
    
    try:
        test_rule_filter()
        test_privacy_ner()
        test_pipeline()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
