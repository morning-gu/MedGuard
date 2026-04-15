# MedGuard Python SDK

[![PyPI version](https://badge.fury.io/py/medguard.svg)](https://badge.fury.io/py/medguard)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Python SDK for MedGuard - Medical Safety Guardrail for LLMs.

## Installation

```bash
pip install medguard
```

## Quick Start

```python
from medguard import MedGuardPipeline, PrivacyNER

# Initialize the guardrail
guard = MedGuardPipeline(model_name="medguard-7b-qlora")

# Check a medical query
result = guard.check("Can I take aspirin if I'm pregnant?")

if result.is_blocked:
    print(f"🚫 Blocked: {result.reason}")
    print(f"Suggested: {result.suggested_response}")
else:
    print("✅ Safe to proceed")

# Get token-level risk attribution
print(f"Risk tokens: {result.token_risks}")
```

## API Reference

### MedGuardPipeline

Main class for medical safety checking.

#### Constructor

```python
MedGuardPipeline(
    model_name: str = "medguard-7b-qlora",
    device: str = "cuda",
    quantization: str = "int4",
    risk_threshold: float = 0.7,
    include_attribution: bool = True
)
```

**Parameters:**
- `model_name`: Model to use for inference
- `device`: Device for inference ('cuda' or 'cpu')
- `quantization`: Quantization strategy ('int4', 'int8', 'fp16')
- `risk_threshold`: Threshold above which content is blocked (0.0-1.0)
- `include_attribution`: Whether to include token-level risk attribution

#### Methods

##### check(text, include_attribution=None)

Check text for medical safety risks.

**Parameters:**
- `text`: Input text to analyze
- `include_attribution`: Override default attribution setting

**Returns:** `SafetyResult` object

**Example:**
```python
result = guard.check("I want to overdose on tylenol")
print(result.is_blocked)  # True
print(result.risk_score)  # 0.95
print(result.severity)    # "critical"
```

##### batch_check(texts, batch_size=8)

Check multiple texts efficiently.

**Parameters:**
- `texts`: List of input texts
- `batch_size`: Batch size for processing

**Returns:** List of `SafetyResult` objects

### SafetyResult

Result of a safety check.

**Attributes:**
- `is_blocked` (bool): Whether content should be blocked
- `risk_score` (float): Risk score (0.0-1.0)
- `reason` (str|None): Explanation for blocking
- `suggested_response` (str|None): Safe response template
- `token_risks` (List[TokenRisk]): Token-level risk attribution
- `category` (str|None): Risk category
- `severity` (str): Severity level ("low", "medium", "high", "critical")

### TokenRisk

Represents risk associated with a specific token.

**Attributes:**
- `token` (str): The risky token
- `position` (List[int]): Character positions [start, end]
- `risk_score` (float): Risk score for this token
- `reason` (str): Why this token is risky
- `category` (str): Risk category

### PrivacyNER

Privacy entity recognition for PHI de-identification.

#### Methods

##### detect(text)

Detect PHI entities in text.

**Returns:** List of `PHIEntity` objects

##### anonymize(text)

Anonymize PHI in text.

**Returns:** Text with PHI replaced by redaction markers

**Example:**
```python
ner = PrivacyNER()
text = "Patient John Doe, MRN: 12345, DOB: 01/15/1985"
anonymized = ner.anonymize(text)
print(anonymized)
# Output: "Patient [PATIENT REDACTED], MRN: [MRN REDACTED], DOB: [DOB REDACTED]"
```

##### get_phi_summary(text)

Get summary of detected PHI.

**Returns:** Dictionary with PHI statistics

## Advanced Usage

### Custom Risk Threshold

```python
# More conservative (block more)
guard_strict = MedGuardPipeline(risk_threshold=0.5)

# More permissive (block less)
guard_lenient = MedGuardPipeline(risk_threshold=0.85)
```

### Without Token Attribution

```python
# Faster inference without attribution
result = guard.check(text, include_attribution=False)
```

### Batch Processing

```python
texts = [
    "Can I take ibuprofen while pregnant?",
    "What's the dose for metformin?",
    "I'm feeling suicidal"
]

results = guard.batch_check(texts)
for text, result in zip(texts, results):
    print(f"{text}: {'BLOCKED' if result.is_blocked else 'SAFE'}")
```

### Integration with LangChain

```python
from langchain_medguard import MedGuardSafetyChain

safety_chain = MedGuardSafetyChain(threshold=0.7)
safe_llm = safety_chain.wrap(your_llm)

response = safe_llm.invoke("Medical question here...")
```

## Error Handling

```python
from medguard import MedGuardPipeline, MedGuardError

guard = MedGuardPipeline()

try:
    result = guard.check("Your text here")
except MedGuardError as e:
    print(f"Safety check failed: {e}")
    # Handle error appropriately
```

## Performance Tips

1. **Reuse instances**: Create one `MedGuardPipeline` and reuse it
2. **Batch processing**: Use `batch_check()` for multiple texts
3. **Disable attribution**: Set `include_attribution=False` for faster inference
4. **Use rule filter**: Rules provide fast path for obvious cases

```python
# Optimal setup for high-throughput
guard = MedGuardPipeline(
    model_name="medguard-7b-qlora",
    device="cuda",
    quantization="int4",
    include_attribution=False  # Disable for speed
)

# Process many texts
results = guard.batch_check(large_text_list, batch_size=16)
```

## License

Apache License 2.0
