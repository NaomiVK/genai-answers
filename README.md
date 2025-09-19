# CRA Multi-Model Evaluation System

A Python-based evaluation system that assesses the quality and accuracy of multiple AI models' responses to Canada Revenue Agency (CRA) related questions.

## Features

- **Multi-Model Evaluation**: Compare responses from multiple AI models simultaneously
- **Expert Judge**: Uses Claude Opus 4.1 via OpenRouter to evaluate factual correctness
- **Cross-Model Analysis**: Identifies contradictions and unique facts across models
- **Multiple Output Formats**: JSON, CSV, and Excel files for comprehensive analysis
- **Detailed Metrics**: Correctness scores, critical errors, and missing information tracking

## Quick Start

### Prerequisites

- Python 3.8+
- OpenRouter API key

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd genai-answers

# Install dependencies
pip install pandas openai tqdm openpyxl
```

### Configuration

Create a `.env` file with your API key:
```bash
OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

### Usage

```bash
# Evaluate with default settings
python evaluate_cra.py

# Evaluate specific number of rows
python evaluate_cra.py --max-rows 10

# Use a different Excel file
python evaluate_cra.py --xlsx "your-data.xlsx"
```

## Input Format

The script expects an Excel file with:
- **Question column**: CRA-related questions
- **Model response columns**: AI model responses (auto-detected)

## Output Files

The system generates four output files in the `eval_out/` directory:

1. **per_question_eval.jsonl**: Raw evaluation data in JSON Lines format
2. **summary_scores.csv**: High-level metrics for each model response
3. **per_question_details.csv**: Detailed evaluation breakdowns
4. **cross_model_flags.csv/xlsx**: Cross-model contradictions and unique facts

## Evaluation Metrics

- **Correctness Score (0-1)**: Numerical assessment of factual accuracy
- **Verdict**: Categorical assessment (correct/partially_correct/incorrect/uncertain)
- **Critical Errors**: Statements that would materially mislead users
- **Missing Information**: Important facts omitted from responses

## Documentation

See [CRA_EVALUATION_GUIDE.md](CRA_EVALUATION_GUIDE.md) for comprehensive documentation including:
- Detailed evaluation criteria
- Output file descriptions
- Interpretation guide
- Scaling strategies
- Troubleshooting tips

## License

[Specify your license here]

## Contributing

[Add contribution guidelines if needed]