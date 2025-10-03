# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python-based evaluation system that assesses the quality and accuracy of multiple AI models' responses to Canada Revenue Agency (CRA) related questions. It uses Claude Opus 4.1 via OpenRouter as an expert judge to evaluate factual correctness of responses.

## High-Level Architecture

The system follows a pipeline architecture:
1. **Input Processing**: Reads Excel file with questions and model responses
2. **Two-Pass Evaluation Engine**:
   - **First Pass**: Initial evaluation by Claude Opus 4.1 judge with strict JSON schema
   - **Second Pass**: Self-review and validation of first pass evaluation for improved accuracy
3. **Cross-Model Analysis**: Identifies contradictions and unique facts across models
4. **Output Generation**: Creates multiple output formats (JSONL, CSV, XLSX) with different levels of detail

Key evaluation flow:
- Questions are hashed for consistent IDs
- Model responses are auto-detected (columns with >120 char average)
- Two-pass evaluation ensures accurate, validated assessments (enabled by default)
- Each evaluation includes per-model scoring, cross-model analysis, and confidence scores
- Results are written to `eval_out/` directory with multiple formats
- Retry logic with exponential backoff handles API failures gracefully

## Development Commands

### Setup Environment
```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Evaluation
```bash
# Basic evaluation with default settings (two-pass enabled by default)
python evaluate_cra.py

# Evaluate specific number of rows
python evaluate_cra.py --max-rows 10

# Use different Excel file
python evaluate_cra.py --xlsx your-data.xlsx

# Specify sheet and model columns
python evaluate_cra.py --xlsx genai-answers.xlsx --sheet Sheet1 --models ModelA,ModelB

# Use different judge model
python evaluate_cra.py --model anthropic/claude-opus-4.1

# Disable two-pass evaluation (faster but less accurate)
python evaluate_cra.py --no-two-pass

# Enable two-pass evaluation explicitly (this is the default)
python evaluate_cra.py --two-pass
```

### Environment Variables
Create a `.env` file with:
```
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

## Code Organization

- **evaluate_cra.py**: Main evaluation script with two-pass validation system
  - **Two-Pass Evaluation**: First pass generates initial assessment, second pass validates and refines it
  - **Few-Shot Prompting**: Includes example evaluations for better judge calibration
  - **Retry Logic**: Exponential backoff (1s, 2s, 4s) handles API failures gracefully
  - **Validation Logging**: Warns when judge responses need coercion/normalization
  - **Temperature**: 0.3 for balanced, nuanced evaluations
  - **Token Limit**: 10,000 tokens to accommodate complex multi-model evaluations
  - **Confidence Scoring**: Each evaluation includes final_confidence (0.0-1.0) from second pass

- **Output Files** (in `eval_out/`):
  - `per_question_eval.jsonl`: Raw evaluation data with confidence scores
  - `summary_scores.csv`: High-level metrics per model
  - `per_question_details.csv/xlsx`: Detailed breakdowns with consistency metrics and confidence
  - `cross_model_flags.csv/xlsx`: Cross-model contradictions and unique facts
  - `cra_evaluation_consolidated.xlsx`: Multi-sheet workbook with complete analysis including model responses

## Key Technical Details

- **Python Version**: 3.8+ (tested on 3.12.3)
- **API Integration**: Uses OpenRouter API with OpenAI SDK compatibility
- **JSON Schema**: Strict schema validation ensures consistent evaluation structure
- **Error Handling**: Comprehensive error handling with 3-attempt retry and exponential backoff
- **Two-Pass Validation**: Same model (Claude Opus 4.1) reviews its own work for self-consistency
- **Consistency Metrics**: Calculates agreement scores, contradiction counts, and unique fact identification
- **Confidence Tracking**: Every evaluation includes confidence score from second-pass review
- **Robustness**: Handles Excel columns beyond Z, validates all judge responses, logs coercion warnings

## Testing Approach

Currently no formal test suite. To verify changes:
1. Run with small dataset (1-2 rows): `python evaluate_cra.py --max-rows 2`
2. Check output files are created in `eval_out/`
3. Verify CSV/JSONL structure matches expected schema
4. For new features, consider adding unit tests under `tests/` using pytest

## Important Notes

- Never commit `.env` file or API keys
- Large artifacts (`eval_out/`, `.venv/`, `*.xlsx`) are git-ignored
- The system evaluates ONLY factual correctness regarding CRA information
- **Two-pass evaluation is enabled by default** for improved accuracy (use `--no-two-pass` to disable)
- Most evaluations show "uncertain" verdict as the judge is conservative about CRA-specific facts
- Correctness scores (0-1) provide nuance even when verdict is uncertain
- **EvalConfidence** column shows judge's confidence in the evaluation (0.0-1.0)
- Low confidence scores may indicate questions requiring human review

## Recent Improvements (2025-10-01)

1. **Two-Pass Evaluation System**: Judge reviews and validates its own assessments for improved accuracy
2. **Few-Shot Examples**: Judge prompt now includes calibration examples for better consistency
3. **Increased Token Limit**: Raised from 4,000 to 10,000 tokens for complex evaluations
4. **Temperature Adjustment**: Increased from 0.1 to 0.3 for more nuanced assessments
5. **Retry Logic**: Added exponential backoff (3 attempts) for API failure resilience
6. **Validation Logging**: Warns when judge responses require normalization
7. **Excel Column Fix**: Correctly handles columns beyond Z (AA, AB, etc.)
8. **Confidence Scoring**: All evaluations include final_confidence metric