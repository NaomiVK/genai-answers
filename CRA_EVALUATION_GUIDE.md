# CRA Multi-Model Evaluation System

## Overview

This evaluation system assesses the quality and accuracy of multiple AI models' responses to Canada Revenue Agency (CRA) related questions. It uses Claude Opus 4.1 via OpenRouter as an expert judge to evaluate responses from various models, focusing specifically on Canadian federal tax information accuracy.

## How It Works

### 1. Data Input
The script reads from an Excel file (`genai-answers.xlsx`) containing:
- **Question column**: User questions about CRA services, tax matters, or account access
- **Model response columns**: Answers from different AI models (auto-detected as columns with >120 characters average length)

### 2. Evaluation Process

```mermaid
graph LR
    A[Excel Data] --> B[Question + Model Responses]
    B --> C[Claude Opus 4.1 Judge]
    C --> D[Structured Evaluation]
    D --> E[Multiple Output Formats]
```

Each evaluation batch:
1. Extracts a question and all model responses
2. Adds CRA-specific context to ensure proper interpretation
3. Sends to Claude Opus 4.1 with strict evaluation criteria
4. Receives structured JSON evaluation
5. Processes results into multiple output formats

### 3. Evaluation Criteria

The judge evaluates **ONLY factual correctness** regarding CRA information, not style or formatting.

#### Key Metrics:

- **Verdict**: Categorical assessment of the response
  - `correct` = Fully accurate and complete
  - `partially_correct` = Some correct information but has issues
  - `incorrect` = Materially wrong or misleading
  - `uncertain` = Judge lacks confidence in CRA-specific facts to assess

  *Note: Most evaluations show "uncertain" as Claude Opus is conservative when not completely sure about specific CRA facts.*

- **Correctness Score (0-1)**: Numerical assessment providing nuance even when verdict is uncertain
  - 1.0 = Fully correct & materially complete
  - 0.5 = Partially correct
  - 0.0 = Materially incorrect/misleading

  *Example: Verdict="uncertain", Score=0.8 means the judge isn't confident enough to call it "correct" but rates it 80% accurate based on known information.*

- **Critical Errors**: Statements that would materially mislead a taxpayer
  - Wrong eligibility requirements
  - Incorrect rates, amounts, dates, or forms
  - Wrong procedures or portals

- **Key Points Missing**: Important facts that SHOULD be present for a safe, correct answer

- **Unsupported Assertions**: Specific claims that cannot be justified from well-known CRA facts

## Output Files

### 1. `per_question_eval.jsonl`
Raw JSON Lines format with complete evaluation data for each question.

### 2. `summary_scores.csv`
High-level scoring summary:
```csv
QID | Question | Model | Verdict | Correctness | CriticalErrorCount | MissingCount
```

**Score Calculation**:
- `Correctness`: Direct score from evaluator (0-1)
- `CriticalErrorCount`: Count of critical errors identified
- `MissingCount`: Count of key missing points

### 3. `per_question_details.csv`
Detailed evaluation breakdown with consistency metrics:
```csv
QID | Question | Model | Verdict | Correctness | CriticalErrors | UnsupportedAssertions | KeyPointsCovered | KeyPointsMissing | Notes | ConsistencyScore | AgreementLevel | Contradictions | UniqueFacts
```

**Fields**:
- `CriticalErrors`: Semicolon-separated list of specific errors
- `UnsupportedAssertions`: Claims that can't be verified
- `KeyPointsCovered`: Important points correctly addressed
- `KeyPointsMissing`: Important omissions
- `Notes`: Additional evaluator observations
- `ConsistencyScore` (0-1): Measures how consistent model responses are for this question
  - 1.0 = Perfect consistency (all models scored similarly)
  - 0.5 = Moderate consistency
  - 0.0 = Very inconsistent responses
- `AgreementLevel`: Categorical assessment of model agreement
  - High: Standard deviation < 0.1 in scores
  - Medium: Standard deviation 0.1-0.25
  - Low: Standard deviation > 0.25
- `Contradictions`: Number of direct contradictions between models on this question
- `UniqueFacts`: Number of fact groups mentioned by only one model

### 4. `cross_model_flags.csv`
Cross-model comparison data with improved structure:
```csv
QID | Type | Model | MissingFromModels | Details
```

**Column Descriptions**:
- `QID`: Question identifier
- `Type`: Either "Unique Facts" or "Contradiction"
- `Model`:
  - For Unique Facts: The model that provided the unique information
  - For Contradictions: All models involved in the disagreement
- `MissingFromModels`:
  - For Unique Facts: Models that didn't mention these facts
  - For Contradictions: Always "N/A"
- `Details`:
  - For Unique Facts: The actual facts (semicolon-separated)
  - For Contradictions: The topic of disagreement only (no model positions)

**Cross-Model Analysis Types**:

1. **Unique Facts**: Information one model provides that others omit
   - Identifies comprehensive vs. minimal responses
   - Highlights unique insights or details
   - Example: "CRA phone number 1-800-959-8281; Security code required"

2. **Contradictions**: Direct disagreements between models
   - Tax rates, eligibility criteria
   - Procedures or portal names
   - Account types or identifiers
   - Example: "Business Number format" or "What constitutes the CRA account number"

## Example Output Insights

### Common Findings from Evaluations

1. **Phone Numbers**: Models often differ on whether to include CRA contact numbers
2. **Security Details**: Some models mention banking credentials aren't shared with CRA, others omit
3. **Business Number Format**: Major contradiction area - 9 digits vs. 15 characters debate
4. **Account Identifiers**: Disagreement about whether CRA has separate account numbers from SIN
5. **Sign-In Partner Changes**: Conflicting information about switching from Sign-In Partner to CRA user ID

### Interpretation Guide

### Correctness Scores
- **High Correctness Score (0.8-1.0)**: Generally reliable information
- **Medium Score (0.5-0.7)**: Some issues but core information likely correct
- **Low Score (0.0-0.4)**: Significant errors or missing critical information
- **Uncertain Verdict**: Judge being conservative; check correctness score for relative quality

### Consistency Metrics
- **High Consistency (0.8-1.0)**: Models agree on most facts and approaches
  - Indicates robust understanding of the topic
  - Lower risk of misinformation
- **Medium Consistency (0.5-0.8)**: Some variation in responses
  - May indicate ambiguity in the question
  - Review cross_model_flags for specific differences
- **Low Consistency (0.0-0.5)**: Significant disagreement between models
  - High risk area requiring human review
  - Check contradictions and unique facts for details

### Using Consistency for Quality Assurance
1. **High Agreement + High Scores**: Most reliable answers
2. **High Agreement + Low Scores**: All models struggling (knowledge gap)
3. **Low Agreement + Mixed Scores**: Some models have better information
4. **Low Agreement + Low Scores**: Problematic question or topic area

## Configuration

### Environment Variables
```bash
OPENROUTER_API_KEY=sk-or-v1-xxx  # Required for OpenRouter access
```

### Command Line Options
```bash
python evaluate_cra.py \
  --xlsx "genai-answers.xlsx" \      # Input Excel file
  --sheet "Sheet1" \                  # Specific sheet (optional)
  --models "GPT-4,Claude,Gemini" \    # Model columns (auto-detected if omitted)
  --model "anthropic/claude-opus-4.1" # Judge model
  --outdir "eval_out" \               # Output directory
  --max-rows 10                       # Limit rows for testing
```

## Scaling Considerations

### 1. Performance Optimization
- **Batch Processing**: Process multiple questions in parallel
- **Caching**: Cache evaluations by question+answer hash to avoid re-evaluation
- **Async API Calls**: Implement concurrent API requests with rate limiting

### 2. Cost Management
- **Sampling Strategy**: Evaluate representative samples rather than full datasets
- **Tiered Evaluation**: Use cheaper models for initial screening, Opus for final evaluation
- **Response Deduplication**: Skip evaluation of identical responses

### 3. Quality Improvements

#### Grounding with URLs
Enhance evaluation accuracy by:
```python
# Add official CRA URL validation
CRA_DOMAINS = [
    "canada.ca/en/revenue-agency",
    "canada.ca/fr/agence-revenu",
    "apps.cra-arc.gc.ca"
]

# Verify cited sources in responses
def validate_urls(response_text):
    urls = extract_urls(response_text)
    return [url for url in urls if any(domain in url for domain in CRA_DOMAINS)]
```

#### Fact Database Integration
```python
# Build knowledge base from official sources
CRA_FACTS = {
    "tax_rates": load_from_cra_api(),
    "deadlines": parse_cra_calendar(),
    "forms": scrape_form_directory(),
    "programs": extract_program_details()
}

# Cross-reference model claims
def verify_claim(claim, fact_db):
    return fact_db.lookup(claim.entity, claim.attribute)
```

### 4. Advanced Analytics

#### Temporal Analysis
Track model performance over time:
- Accuracy trends by model version
- Seasonal accuracy variations (tax season vs. off-season)
- Response consistency across updates

#### Error Pattern Detection
```python
# Identify systematic issues
def analyze_error_patterns(evaluations):
    patterns = {
        "rate_errors": [],
        "deadline_errors": [],
        "eligibility_errors": [],
        "portal_confusion": []
    }
    # Categorize and track error types
    return patterns
```

## Best Practices

### 1. Data Preparation
- Ensure questions are clear and unambiguous
- Include edge cases and complex scenarios
- Balance question types (accounts, taxes, benefits, deadlines)

### 2. Evaluation Consistency
- Use consistent judge model and parameters
- Document any prompt changes
- Version control evaluation criteria

### 3. Result Interpretation
- "Uncertain" verdicts indicate judge knowledge limitations
- Low scores may indicate outdated training data
- Cross-model contradictions highlight areas needing verification

## Future Enhancements

### 1. Multi-Language Support
Extend to French CRA content:
```python
JUDGE_PREAMBLE_FR = """Vous êtes un évaluateur expert de l'Agence du revenu du Canada (ARC)..."""
```

### 2. Automated Fact Checking
```python
# Real-time verification against CRA API
async def verify_fact(claim):
    cra_api_response = await cra_client.verify(claim)
    return cra_api_response.is_valid
```

### 3. Confidence Calibration
```python
# Add confidence scores to evaluations
"confidence_level": 0.95,  # Judge's confidence in evaluation
"evidence_quality": "strong",  # Based on verifiable sources
```

### 4. Interactive Dashboard
- Real-time evaluation monitoring
- Model comparison visualizations
- Error trend analysis
- Export capabilities for stakeholder reports

## Recent Improvements

### Consistency Metrics (Latest)
- Added ConsistencyScore (0-1) to measure response agreement across models
- AgreementLevel (High/Medium/Low) for quick assessment
- Contradiction and UniqueFacts counts in per_question_details.csv
- Helps identify questions where models disagree significantly

### Model Name Mapping
- Automatic correction when Claude Opus doesn't return proper model names
- Maps evaluations to actual model columns even when parsing issues occur
- Ensures consistent model identification across all output files

### Enhanced Cross-Model Analysis
- Cleaner CSV structure with clear Type column (Unique Facts vs. Contradictions)
- Details column contains only facts/topics, not model names
- Model involvement clearly shown in dedicated columns
- Cross-model flags now output to both CSV and Excel formats

### CRA Context Enhancement
- Explicit CRA context added to every evaluation prompt
- Judge preamble emphasizes Canadian federal tax administration
- Reduces ambiguity in question interpretation

## Troubleshooting

### Common Issues

1. **Model names showing as "model"**:
   - Now auto-corrected by matching evaluation order to model columns
   - Original issue: Occurs when Claude Opus doesn't extract model names from prompt

2. **Empty cross_model_flags**:
   - Normal when models agree on all points
   - Increase sample size for more contradictions
   - Check for subtle differences in phrasing that may not be flagged

3. **API Timeouts**:
   - Default timeout: 2 minutes per evaluation
   - Can increase with `--timeout` parameter
   - Consider using `--max-rows` for testing

4. **CSV formatting issues**:
   - Details field properly quoted when containing commas
   - Valid CSV format readable by pandas, Excel, and Google Sheets
   - Some basic CSV readers may struggle with embedded commas

### Debug Mode
Enable detailed logging:
```python
# Add to script
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Compliance & Ethics

- **Privacy**: Never include personal taxpayer information in evaluations
- **Accuracy**: Evaluations are assistive, not authoritative tax advice
- **Transparency**: Document evaluation criteria and limitations
- **Updates**: Regularly update fact bases for tax law changes

## Contact & Support

For questions or improvements to the evaluation system:
- Review official CRA documentation at canada.ca
- Consult latest tax legislation and bulletins
- Consider seasonal tax changes and updates

---

*Note: This evaluation system is designed for quality assurance of AI responses about CRA information. It does not replace professional tax advice or official CRA guidance.*