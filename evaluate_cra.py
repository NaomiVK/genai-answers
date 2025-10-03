import os, json, argparse, hashlib, time
import pandas as pd
from tqdm import tqdm
from openai import OpenAI

DEFAULT_MODEL = "anthropic/claude-opus-4.1"  
REQUIRED_FIELDS = ["Question"]  

# Strict JSON schema for responses_api -> response_format.json_schema
EVAL_SCHEMA = {
    "name": "cra_llm_eval_v1",
    "schema": {
        "type": "object",
        "properties": {
            "question_id": {"type": "string"},
            "question": {"type": "string"},
            "per_model": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "verdict": {"type": "string", "enum": ["correct","partially_correct","incorrect","uncertain"]},
                        "correctness_score": {"type": "number", "minimum": 0, "maximum": 1},
                        "critical_errors": {"type": "array", "items": {"type": "string"}},
                        "unsupported_assertions": {"type": "array", "items": {"type": "string"}},
                        "key_points_covered": {"type": "array", "items": {"type": "string"}},
                        "key_points_missing": {"type": "array", "items": {"type": "string"}},
                        "notes": {"type": "string"}
                    },
                    "required": ["name","verdict","correctness_score","critical_errors","key_points_missing"]
                }
            },
            "cross_model": {
                "type": "object",
                "properties": {
                    "facts_unique_to_model": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "anchor_model": {"type": "string"},
                                "missing_in_models": {"type": "array", "items": {"type": "string"}},
                                "facts": {"type": "array", "items": {"type": "string"}}
                            },
                            "required": ["anchor_model","missing_in_models","facts"]
                        }
                    },
                    "contradictions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "topic": {"type": "string"},
                                "model_positions": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "position_summary": {"type": "string"}
                                        },
                                        "required": ["name","position_summary"]
                                    }
                                }
                            },
                            "required": ["topic","model_positions"]
                        }
                    }
                },
                "required": ["facts_unique_to_model","contradictions"]
            }
        },
        "required": ["question_id","question","per_model","cross_model"],
        "additionalProperties": False
    },
    "strict": True
}

# Schema for second-pass evaluation with confidence score
EVAL_SCHEMA_WITH_CONFIDENCE = {
    "name": "cra_llm_eval_with_confidence_v1",
    "schema": {
        "type": "object",
        "properties": {
            "question_id": {"type": "string"},
            "question": {"type": "string"},
            "per_model": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "verdict": {"type": "string", "enum": ["correct","partially_correct","incorrect","uncertain"]},
                        "correctness_score": {"type": "number", "minimum": 0, "maximum": 1},
                        "critical_errors": {"type": "array", "items": {"type": "string"}},
                        "unsupported_assertions": {"type": "array", "items": {"type": "string"}},
                        "key_points_covered": {"type": "array", "items": {"type": "string"}},
                        "key_points_missing": {"type": "array", "items": {"type": "string"}},
                        "notes": {"type": "string"}
                    },
                    "required": ["name","verdict","correctness_score","critical_errors","key_points_missing"]
                }
            },
            "cross_model": {
                "type": "object",
                "properties": {
                    "facts_unique_to_model": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "anchor_model": {"type": "string"},
                                "missing_in_models": {"type": "array", "items": {"type": "string"}},
                                "facts": {"type": "array", "items": {"type": "string"}}
                            },
                            "required": ["anchor_model","missing_in_models","facts"]
                        }
                    },
                    "contradictions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "topic": {"type": "string"},
                                "model_positions": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "position_summary": {"type": "string"}
                                        },
                                        "required": ["name","position_summary"]
                                    }
                                }
                            },
                            "required": ["topic","model_positions"]
                        }
                    }
                },
                "required": ["facts_unique_to_model","contradictions"]
            },
            "final_confidence": {"type": "number", "minimum": 0, "maximum": 1}
        },
        "required": ["question_id","question","per_model","cross_model","final_confidence"],
        "additionalProperties": False
    },
    "strict": True
}

def _load_env_from_dotenv_or_file():
    """Load environment variables from a .env file if present.

    Tries python-dotenv if installed; otherwise performs a minimal parse of
    KEY=VALUE lines in a local .env file. Existing env vars are not overridden.
    """
    
    try:
        from dotenv import load_dotenv  
        load_dotenv(override=False)
        return
    except Exception:
        pass

    # Minimal fallback parser
    path = ".env"
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                if "=" not in s:
                    continue
                key, val = s.split("=", 1)
                key = key.strip()
                val = val.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = val
    except Exception:
        # Non-fatal: if .env can't be read, continue with existing env
        pass

FEW_SHOT_EXAMPLE = """
EXAMPLE EVALUATION (for calibration):

Question: "What is the TFSA contribution limit for 2024?"

Model 1 (ChatGPT): "The TFSA contribution limit for 2024 is $7,000."
Model 2 (Claude): "The TFSA limit for 2024 is $6,500, same as 2023."

Expected JSON:
{
  "question_id": "example123",
  "question": "What is the TFSA contribution limit for 2024?",
  "per_model": [
    {
      "name": "ChatGPT",
      "verdict": "correct",
      "correctness_score": 1.0,
      "critical_errors": [],
      "unsupported_assertions": [],
      "key_points_covered": ["Correct 2024 TFSA limit of $7,000"],
      "key_points_missing": [],
      "notes": "Accurate and complete answer."
    },
    {
      "name": "Claude",
      "verdict": "incorrect",
      "correctness_score": 0.0,
      "critical_errors": ["States incorrect limit of $6,500 instead of $7,000"],
      "unsupported_assertions": [],
      "key_points_covered": [],
      "key_points_missing": ["Correct 2024 limit"],
      "notes": "Wrong amount could lead to over-contribution penalties."
    }
  ],
  "cross_model": {
    "facts_unique_to_model": [],
    "contradictions": [
      {
        "topic": "2024 TFSA contribution limit",
        "model_positions": [
          {"name": "ChatGPT", "position_summary": "States $7,000"},
          {"name": "Claude", "position_summary": "States $6,500"}
        ]
      }
    ]
  }
}
"""

JUDGE_PREAMBLE = """You are an expert evaluator of Canada Revenue Agency (CRA) information and Canadian federal tax matters.
ALL questions and answers relate to the Canada Revenue Agency (CRA), Canada's federal tax administration.
Judge ONLY factual correctness and coverage of information relevant to the question.
Do NOT grade style, length, tone, formatting, or verbosity.

Use only your training knowledge to evaluate. If you are not confident about CRA-specific facts, mark 'uncertain' and explain briefly.

Definitions:
- correctness_score (0..1): 1.0 = fully correct & materially complete; 0.5 = partially correct; 0.0 = materially incorrect/misleading.
- critical_errors: statements that would materially mislead a taxpayer (wrong eligibility, wrong rate/amount/date/form, wrong procedure/portal).
- key_points_missing: important facts that SHOULD be present for a correct, safe answer to this question but are absent in that model's answer.
- unsupported_assertions: claims that sound specific but you cannot justify from well-known CRA facts.
- cross_model.facts_unique_to_model: information covered by one model but omitted by specified others (not about length, strictly about content). Include anchor_model (who said it), missing_in_models (array of model names that didn't mention it), and facts (array of specific facts).
- cross_model.contradictions: places where models disagree about a fact/value/eligibility/procedure. Include topic (what they disagree about) and model_positions array with name and position_summary for EACH model's stance.

IMPORTANT: Each model's response is labeled with its name (e.g., "Model 1: ChatGPT4o"). You MUST use these exact model names in your per_model evaluations, not generic names like "model".
Evaluate all answers independently first (per_model), then produce cross_model diffs and contradictions.
Return ONLY JSON conforming to the provided schema. Do not include explanations outside JSON.

""" + FEW_SHOT_EXAMPLE

SECOND_PASS_PREAMBLE = """You are reviewing your own previous evaluation of CRA-related model responses.
Your task is to validate and refine your initial assessment.

Review your previous evaluation and:
1. Verify each critical_error is truly materially misleading (not just incomplete or stylistic)
2. Confirm correctness_scores accurately reflect factual accuracy (0.0 = wrong, 0.5 = partial, 1.0 = complete)
3. Double-check contradictions are real factual disagreements (not just different phrasing)
4. Validate unique facts are actually substantive differences (not just verbosity)
5. Assess your overall confidence in this evaluation

Return updated JSON in the same schema as before, with your refined assessment.

CRITICAL: You MUST add a top-level "final_confidence" field (type: number, 0.0-1.0) indicating your confidence:
- 1.0 = Very confident about all CRA facts and assessments
- 0.7-0.9 = High confidence with minor uncertainties
- 0.5-0.6 = Moderate confidence, some uncertainty exists
- 0.3-0.4 = Low confidence, significant uncertainty
- 0.0-0.2 = Very low confidence, needs human review

Example JSON structure with final_confidence:
{
  "question_id": "...",
  "question": "...",
  "per_model": [...],
  "cross_model": {...},
  "final_confidence": 0.85
}
"""

def _call_openrouter_api(client: OpenAI, model: str, prompt: str, system_preamble: str = None, max_retries: int = 3, use_confidence_schema: bool = False):
    """Call OpenRouter API using chat completions with retry logic.

    Returns the parsed JSON object according to EVAL_SCHEMA or EVAL_SCHEMA_WITH_CONFIDENCE.

    Args:
        use_confidence_schema: If True, uses schema with final_confidence field (for second pass)
    """

    json_instruction = "\n\nIMPORTANT: Return your response as a valid JSON object only, with no additional text or markdown formatting."

    if system_preamble is None:
        system_preamble = JUDGE_PREAMBLE

    # Note: OpenRouter doesn't support response_format with json_schema for all models
    # We rely on prompt-based JSON generation instead
    kwargs = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': system_preamble + json_instruction},
            {'role': 'user', 'content': prompt}
        ],
        'temperature': 0.3,  # Balanced temperature for nuanced evaluation
        'max_tokens': 10000  # Increased for complex multi-model evaluations
    }

    # Retry with exponential backoff
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(**kwargs)
            break  # Success, exit retry loop
        except Exception as e:
            if attempt == max_retries - 1:
                print(f"[ERROR] OpenRouter API call failed after {max_retries} attempts: {e}")
                raise

            # Exponential backoff: 2^attempt seconds (1s, 2s, 4s...)
            wait_time = 2 ** attempt
            print(f"[WARNING] API call failed (attempt {attempt + 1}/{max_retries}), retrying in {wait_time}s: {e}")
            time.sleep(wait_time)

    # Extract the response text
    try:
        text = resp.choices[0].message.content
    except Exception as e:
        raise SystemExit(f"Failed to extract response from OpenRouter API: {e}")

    def _try_parse_any_json(s: str):
        try:
            return json.loads(s)
        except Exception:
            pass
        # Strip Markdown fences
        ss = s.strip()
        if ss.startswith("```"):
            # remove first fence line and optional language tag
            ss = ss.split("\n", 1)[1] if "\n" in ss else ss
            if ss.endswith("```"):
                ss = ss.rsplit("```", 1)[0]
        try:
            return json.loads(ss)
        except Exception:
            pass
        # Heuristic: extract first balanced JSON object
        start = s.find('{')
        if start != -1:
            depth = 0
            for i in range(start, len(s)):
                ch = s[i]
                if ch == '{':
                    depth += 1
                elif ch == '}':
                    depth -= 1
                    if depth == 0:
                        candidate = s[start:i+1]
                        try:
                            return json.loads(candidate)
                        except Exception:
                            break
        raise SystemExit(
            "Model did not return valid JSON under current SDK capabilities."
            f"\nFirst 300 chars: {s[:300]}"
        )

    parsed = _try_parse_any_json(text)
    return parsed

def _coerce_to_eval_schema(parsed: dict, qid: str, question: str, model_cols):
    """Normalize loosely-structured JSON into EVAL_SCHEMA-compatible dict.

    Fills required keys and defaults when the model returns a near-miss.
    Logs warnings when coercion is needed.
    """
    coercion_applied = False

    out = {
        "question_id": qid,
        "question": question,
        "per_model": [],
        "cross_model": {
            "facts_unique_to_model": [],
            "contradictions": []
        },
    }

    # per_model normalization
    pm = parsed.get("per_model") if isinstance(parsed, dict) else None
    items = []
    if isinstance(pm, list):
        items = pm
    elif isinstance(pm, dict):
        # convert mapping to list with name
        print(f"[WARNING] QID {qid}: Coercing per_model from dict to list")
        coercion_applied = True
        for name, val in pm.items():
            if isinstance(val, dict):
                tmp = dict(val)
                tmp.setdefault("name", name)
                items.append(tmp)
    # If still empty, synthesize uncertain entries for provided model columns
    if not items and model_cols:
        print(f"[WARNING] QID {qid}: Judge returned no per_model data, synthesizing uncertain verdicts")
        coercion_applied = True
        for name in model_cols:
            items.append({"name": name, "verdict": "uncertain", "correctness_score": 0.0})

    normalized = []
    for idx, it in enumerate(items):
        name = it.get("name") or it.get("model") or "model"

        # If we got generic "model" but have the actual model names, use them
        if name == "model" and model_cols and idx < len(model_cols):
            print(f"[WARNING] QID {qid}: Judge used generic 'model' name, substituting {model_cols[idx]}")
            coercion_applied = True
            name = model_cols[idx]

        # Check if required fields are missing and log warnings
        if "verdict" not in it:
            print(f"[WARNING] QID {qid}, Model {name}: Missing 'verdict', defaulting to 'uncertain'")
            coercion_applied = True
        if "correctness_score" not in it:
            print(f"[WARNING] QID {qid}, Model {name}: Missing 'correctness_score', defaulting to 0.0")
            coercion_applied = True

        normalized.append({
            "name": name,
            "verdict": it.get("verdict", "uncertain"),
            "correctness_score": float(it.get("correctness_score", 0.0)),
            "critical_errors": list(it.get("critical_errors", [])),
            "unsupported_assertions": list(it.get("unsupported_assertions", [])),
            "key_points_covered": list(it.get("key_points_covered", [])),
            "key_points_missing": list(it.get("key_points_missing", [])),
            "notes": it.get("notes", ""),
        })
    out["per_model"] = normalized

    if coercion_applied:
        print(f"[WARNING] QID {qid}: Coercion was applied to normalize judge response")

    # cross_model normalization
    cm = parsed.get("cross_model") if isinstance(parsed, dict) else {}
    if isinstance(cm, dict):
        facts = cm.get("facts_unique_to_model", [])
        if isinstance(facts, list):
            out["cross_model"]["facts_unique_to_model"] = [
                {
                    "anchor_model": f.get("anchor_model", "unknown"),
                    "missing_in_models": list(f.get("missing_in_models", [])),
                    "facts": list(f.get("facts", [])),
                }
                for f in facts if isinstance(f, dict)
            ]
        cons = cm.get("contradictions", [])
        if isinstance(cons, list):
            out["cross_model"]["contradictions"] = [
                {
                    "topic": c.get("topic", "unspecified"),
                    "model_positions": [
                        {"name": mp.get("name", "model"), "position_summary": mp.get("position_summary", "")}
                        for mp in c.get("model_positions", []) if isinstance(mp, dict)
                    ],
                }
                for c in cons if isinstance(c, dict)
            ]

    # Preserve final_confidence if present (from second pass)
    if "final_confidence" in parsed:
        out["final_confidence"] = parsed["final_confidence"]

    return out

def detect_model_cols(df, explicit_models=None):
    if explicit_models:
        cols = []
        for m in explicit_models:
            if m not in df.columns:
                raise SystemExit(f"Model column not found: {m}")
            cols.append(m)
        return cols
    # heuristic: keep long-text columns other than the Question column
    qcol = "Question"
    texty = []
    for c in df.columns:
        if c == qcol: 
            continue
        s = df[c].dropna().astype(str)
        if len(s) == 0:
            continue
        avg = s.map(len).mean()
        if avg >= 120:  # long-ish = likely an answer column
            texty.append(c)
    if len(texty) < 2:
        raise SystemExit("Could not auto-detect multiple model answer columns. Pass --models explicitly.")
    return texty

def sha10(x: str) -> str:
    return hashlib.sha1(str(x).encode("utf-8")).hexdigest()[:10]

def make_prompt(qid, question, model_answers):
    # model_answers: list[{"name":..., "text":...}]
    context = "IMPORTANT CONTEXT: This question is about the Canada Revenue Agency (CRA) - Canada's federal tax agency. The user is asking about CRA services, accounts, or tax-related matters.\n\n"
    lines = [f"Question (QID={qid}):\n{question.strip()}\n"]
    for i, d in enumerate(model_answers, 1):
        lines.append(f"--- Model {i}: {d['name']} ---\n{str(d['text']).strip()}\n")
    task = "\n".join(lines)
    return context + task

def make_second_pass_prompt(qid, question, model_answers, first_pass_eval):
    """Create prompt for second-pass validation of initial evaluation."""
    original_prompt = make_prompt(qid, question, model_answers)

    first_pass_json = json.dumps(first_pass_eval, indent=2, ensure_ascii=False)

    second_pass_task = f"""{original_prompt}

---

PREVIOUS EVALUATION:
{first_pass_json}

Review the evaluation above and provide your refined assessment.
Ensure critical_errors are truly critical, correctness_scores are accurate, and contradictions are real.
Add a "final_confidence" field (0.0-1.0) at the top level indicating your confidence in this refined evaluation.
"""
    return second_pass_task

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="genai-answers.xlsx", help="Path to Excel workbook")
    ap.add_argument("--sheet", default=None, help="Sheet name (default: all)")
    ap.add_argument("--models", default=None,
                    help="Comma-separated model column names to evaluate (default: auto-detect long text columns)")
    ap.add_argument("--outdir", default="eval_out", help="Output directory")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Judge model (default: anthropic/claude-opus-4.1)")
    ap.add_argument("--max-rows", type=int, default=None, help="Evaluate at most this many rows per sheet")
    ap.add_argument("--two-pass", action="store_true", default=True, help="Enable two-pass evaluation for improved accuracy (default: True)")
    ap.add_argument("--no-two-pass", dest="two_pass", action="store_false", help="Disable two-pass evaluation")
    args = ap.parse_args()

    # Ensure .env is loaded so OPENAI_API_KEY/OPENAI_EVAL_MODEL are available
    _load_env_from_dotenv_or_file()

    os.makedirs(args.outdir, exist_ok=True)
    xls = pd.ExcelFile(args.xlsx)
    sheets = [args.sheet] if args.sheet else xls.sheet_names

    # Initialize OpenAI client for OpenRouter
    # Requires OPENROUTER_API_KEY in environment
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise SystemExit("OPENROUTER_API_KEY environment variable is required for OpenRouter")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key
    )

    jsonl_path = os.path.join(args.outdir, "per_question_eval.jsonl")
    sum_csv  = os.path.join(args.outdir, "summary_scores.csv")
    diff_csv = os.path.join(args.outdir, "cross_model_flags.csv")
    detail_csv = os.path.join(args.outdir, "per_question_details.csv")

    all_rows = []
    all_diffs = []
    all_details = []
    all_complete_analysis = []  # New comprehensive analysis data
    response_cache = {}  # Store QID+Model -> Response mapping
    # Track consistency metrics per question
    question_consistency = {}

    with open(jsonl_path, "w", encoding="utf-8") as w:
        for sheet in sheets:
            df = pd.read_excel(args.xlsx, sheet_name=sheet)
            if "Question" not in df.columns:
                raise SystemExit(f"Expected a 'Question' column in sheet '{sheet}'.")

            model_cols = detect_model_cols(df, [m.strip() for m in args.models.split(",")] if args.models else None)
            print(f"Sheet: {sheet}, detected model columns: {model_cols}")

            processed = 0
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{sheet}"):
                question = str(row["Question"])
                if not question or question.strip() == "" or all(pd.isna(row.get(c)) for c in model_cols):
                    continue

                qid = sha10(question + "|" + sheet)
                model_answers = [{"name": c, "text": row.get(c, "")} for c in model_cols]

                # Store model responses for later use
                for ma in model_answers:
                    response_cache[f"{qid}_{ma['name']}"] = str(ma['text']) if ma['text'] else ""

                # First pass evaluation
                prompt = make_prompt(qid, question, model_answers)
                parsed = _call_openrouter_api(client, args.model, prompt)
                parsed = _coerce_to_eval_schema(parsed, qid, question, [d["name"] for d in model_answers])

                # Second pass evaluation (validation and refinement)
                if args.two_pass:
                    second_pass_prompt = make_second_pass_prompt(qid, question, model_answers, parsed)
                    parsed_second = _call_openrouter_api(client, args.model, second_pass_prompt, system_preamble=SECOND_PASS_PREAMBLE, use_confidence_schema=True)
                    parsed_second = _coerce_to_eval_schema(parsed_second, qid, question, [d["name"] for d in model_answers])

                    # Extract final_confidence (should always be present now with strict schema)
                    final_confidence = parsed_second.get("final_confidence", 0.5)
                    if "final_confidence" not in parsed_second:
                        print(f"[WARNING] QID {qid}: Second pass did not return final_confidence field, defaulting to 0.5")
                        print(f"[DEBUG] Keys in parsed_second: {list(parsed_second.keys())}")
                    parsed = parsed_second  # Use refined evaluation
                    parsed["final_confidence"] = final_confidence  # Ensure it's in the output
                else:
                    parsed["final_confidence"] = None  # Mark as single-pass

                # write JSONL artifact
                w.write(json.dumps(parsed, ensure_ascii=False) + "\n")

                # Calculate consistency metrics for this question
                model_scores = [pm["correctness_score"] for pm in parsed["per_model"]]

                # Calculate consistency score (standard deviation of scores)
                import statistics
                if len(model_scores) > 1:
                    consistency_score = 1 - min(statistics.stdev(model_scores), 1.0)  # Normalize to 0-1

                    # Determine agreement level based on score variance
                    score_range = max(model_scores) - min(model_scores)
                    if score_range <= 0.1:
                        agreement_level = "High"
                    elif score_range <= 0.3:
                        agreement_level = "Medium"
                    else:
                        agreement_level = "Low"
                else:
                    consistency_score = 1.0
                    agreement_level = "N/A"

                # Count contradictions for this question
                num_contradictions = len(parsed["cross_model"]["contradictions"])

                # Calculate unique facts per model (not per question)
                unique_facts_per_model = {}
                for uf in parsed["cross_model"]["facts_unique_to_model"]:
                    anchor_model = uf.get("anchor_model", "unknown")
                    fact_count = len(uf.get("facts", []))
                    unique_facts_per_model[anchor_model] = unique_facts_per_model.get(anchor_model, 0) + fact_count

                # Store consistency metrics for this question
                question_consistency[parsed["question_id"]] = {
                    "ConsistencyScore": round(consistency_score, 3),
                    "AgreementLevel": agreement_level,
                    "Contradictions": num_contradictions,
                    "UniqueFactsPerModel": unique_facts_per_model
                }

                # per-model summary rows
                for pm in parsed["per_model"]:
                    all_rows.append({
                        "QID": parsed["question_id"],
                        "Question": parsed["question"],
                        "Model": pm["name"],
                        "Verdict": pm["verdict"],
                        "Correctness": pm["correctness_score"],
                        "CriticalErrorCount": len(pm["critical_errors"]),
                        "MissingCount": len(pm["key_points_missing"])
                    })

                    # Detailed per-question data with consistency metrics
                    qid = parsed["question_id"]
                    metrics = question_consistency.get(qid, {})
                    # Get unique facts count for this specific model
                    unique_facts_per_model = metrics.get("UniqueFactsPerModel", {})
                    model_unique_facts = unique_facts_per_model.get(pm["name"], 0)

                    all_details.append({
                        "QID": qid,
                        "Question": parsed["question"],  # Full question, no truncation
                        "Model": pm["name"],
                        "Verdict": pm["verdict"],
                        "Correctness": pm["correctness_score"],
                        "CriticalErrors": "; ".join(pm["critical_errors"]) if pm["critical_errors"] else "",
                        "UnsupportedAssertions": "; ".join(pm["unsupported_assertions"]) if pm["unsupported_assertions"] else "",
                        "KeyPointsCovered": "; ".join(pm["key_points_covered"]) if pm["key_points_covered"] else "",
                        "KeyPointsMissing": "; ".join(pm["key_points_missing"]) if pm["key_points_missing"] else "",
                        "Notes": pm["notes"],
                        "ConsistencyScore": metrics.get("ConsistencyScore", "N/A"),
                        "AgreementLevel": metrics.get("AgreementLevel", "N/A"),
                        "Contradictions": metrics.get("Contradictions", 0),
                        "UniqueFacts": model_unique_facts,
                        "EvalConfidence": parsed.get("final_confidence", "N/A")
                    })

                    # Complete analysis data with model responses
                    all_complete_analysis.append({
                        "QID": qid,
                        "Question": parsed["question"],  # Full question text
                        "Model": pm["name"],
                        "Model Response": response_cache.get(f"{qid}_{pm['name']}", ""),
                        "Verdict": pm["verdict"],
                        "Correctness": pm["correctness_score"],
                        "CriticalErrors": "; ".join(pm["critical_errors"]) if pm["critical_errors"] else "",
                        "UnsupportedAssertions": "; ".join(pm["unsupported_assertions"]) if pm["unsupported_assertions"] else "",
                        "KeyPointsCovered": "; ".join(pm["key_points_covered"]) if pm["key_points_covered"] else "",
                        "KeyPointsMissing": "; ".join(pm["key_points_missing"]) if pm["key_points_missing"] else "",
                        "Notes": pm["notes"],
                        "ConsistencyScore": metrics.get("ConsistencyScore", "N/A"),
                        "AgreementLevel": metrics.get("AgreementLevel", "N/A"),
                        "Contradictions": metrics.get("Contradictions", 0),
                        "UniqueFacts": model_unique_facts,
                        "EvalConfidence": parsed.get("final_confidence", "N/A")
                    })

                # cross-model flags
                for diff in parsed["cross_model"]["facts_unique_to_model"]:
                    if diff.get("facts"):  # Only add if there are actual facts
                        all_diffs.append({
                            "QID": parsed["question_id"],
                            "Question": parsed["question"],  # Add full question
                            "Type": "Unique Facts",
                            "Model": diff["anchor_model"],
                            "MissingFromModels": "; ".join(diff["missing_in_models"]) if diff["missing_in_models"] else "N/A",
                            "Details": " | ".join(diff["facts"][:3]) if len(diff["facts"]) > 3 else " | ".join(diff["facts"])  # Limit to 3 examples
                        })

                for con in parsed["cross_model"]["contradictions"]:
                    if con.get("model_positions"):  # Only add if there are actual model positions
                        # Get the models involved in the contradiction
                        models_involved = []
                        for mp in con["model_positions"]:
                            if mp.get("name"):
                                models_involved.append(mp["name"])

                        if models_involved:  # Only add if we have actual model data
                            all_diffs.append({
                                "QID": parsed["question_id"],
                                "Question": parsed["question"],  # Add full question
                                "Type": "Contradiction",
                                "Model": ", ".join(models_involved[:3]) if models_involved else "Multiple",
                                "MissingFromModels": "N/A",
                                "Details": con["topic"] if con.get("topic") and con["topic"] != "unspecified" else "Disagreement detected"
                            })
                    elif con.get("topic") and con["topic"] not in ["unspecified", ""]:
                        # Even if model_positions is empty, if we have a topic, note it
                        all_diffs.append({
                            "QID": parsed["question_id"],
                            "Question": parsed["question"],  # Add full question
                            "Type": "Contradiction",
                            "Model": "Multiple",
                            "MissingFromModels": "N/A",
                            "Details": con["topic"]
                        })

                processed += 1
                if args.max_rows and processed >= args.max_rows:
                    break

    # Write CSVs (keep for backwards compatibility)
    pd.DataFrame(all_rows).to_csv(sum_csv, index=False)
    pd.DataFrame(all_diffs).to_csv(diff_csv, index=False)
    pd.DataFrame(all_details).to_csv(detail_csv, index=False)

    # Write individual Excel files (keep for backwards compatibility)
    diff_xlsx = os.path.join(args.outdir, "cross_model_flags.xlsx")
    pd.DataFrame(all_diffs).to_excel(diff_xlsx, index=False, sheet_name="Cross Model Flags")

    detail_xlsx = os.path.join(args.outdir, "per_question_details.xlsx")
    pd.DataFrame(all_details).to_excel(detail_xlsx, index=False, sheet_name="Per Question Details")

    # Create consolidated Excel file with multiple sheets
    consolidated_file = os.path.join(args.outdir, "cra_evaluation_consolidated.xlsx")
    with pd.ExcelWriter(consolidated_file, engine='openpyxl') as writer:
        # Tab 1: Complete Analysis (comprehensive view with model responses)
        complete_df = pd.DataFrame(all_complete_analysis)
        if not complete_df.empty:
            complete_df.to_excel(writer, sheet_name='Complete Analysis', index=False)

            # Format Complete Analysis worksheet
            worksheet = writer.sheets['Complete Analysis']
            # Set column widths
            worksheet.column_dimensions['B'].width = 50  # Question
            worksheet.column_dimensions['D'].width = 80  # Model Response
            worksheet.column_dimensions['E'].hidden = True  # Hide Verdict column
            for col in ['F', 'G', 'H', 'I', 'J']:
                worksheet.column_dimensions[col].width = 20

        # Tab 2: Summary Scores
        summary_df = pd.DataFrame(all_rows)
        if not summary_df.empty:
            summary_df.to_excel(writer, sheet_name='Summary Scores', index=False)
            worksheet = writer.sheets['Summary Scores']
            worksheet.column_dimensions['B'].width = 50  # Question

        # Tab 3: Cross Model Flags
        flags_df = pd.DataFrame(all_diffs)
        if not flags_df.empty:
            flags_df.to_excel(writer, sheet_name='Cross Model Flags', index=False)
            worksheet = writer.sheets['Cross Model Flags']
            worksheet.column_dimensions['B'].width = 50  # Question
            worksheet.column_dimensions['F'].width = 60  # Details

        # Tab 4: Per Question Details (for backwards compatibility)
        details_df = pd.DataFrame(all_details)
        if not details_df.empty:
            details_df.to_excel(writer, sheet_name='Per Question Details', index=False)
            worksheet = writer.sheets['Per Question Details']
            worksheet.column_dimensions['B'].width = 50  # Question
            # Find and hide Verdict column
            for idx, col in enumerate(details_df.columns):
                if col == 'Verdict':
                    # Convert index to Excel column letter (A, B, ..., Z, AA, AB, ...)
                    col_letter = ''
                    col_num = idx + 1  # Excel columns are 1-indexed
                    while col_num > 0:
                        col_num, remainder = divmod(col_num - 1, 26)
                        col_letter = chr(65 + remainder) + col_letter
                    worksheet.column_dimensions[col_letter].hidden = True
                    break

    # Generate and print summary statistics
    summary_output = []
    summary_output.append("="*80)
    summary_output.append("EVALUATION SUMMARY")
    summary_output.append("="*80)

    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)

    # Calculate summary statistics
    summary_df = pd.DataFrame(all_rows)
    details_df = pd.DataFrame(all_details)
    flags_df = pd.DataFrame(all_diffs)

    if not summary_df.empty:
        num_questions = summary_df['QID'].nunique()
        num_models = summary_df['Model'].nunique()
        total_evaluations = len(summary_df)

        line = f"\nEvaluation Completed: {num_questions} questions, {num_models} models each = {total_evaluations} evaluations"
        print(line)
        summary_output.append("")
        summary_output.append(line.strip())

        # Verdict distribution
        verdict_counts = summary_df['Verdict'].value_counts()
        print(f"\nVerdict Distribution:")
        summary_output.append("")
        summary_output.append("Verdict Distribution:")
        for verdict, count in verdict_counts.items():
            percentage = (count / total_evaluations) * 100
            line = f"   {verdict:20s}: {count:3d} ({percentage:5.1f}%)"
            print(line)
            summary_output.append(line)

        # Correctness score statistics
        avg_correctness = summary_df['Correctness'].mean()
        min_correctness = summary_df['Correctness'].min()
        max_correctness = summary_df['Correctness'].max()
        print(f"\nCorrectness Scores:")
        summary_output.append("")
        summary_output.append("Correctness Scores:")
        line1 = f"   Average: {avg_correctness:.3f}"
        line2 = f"   Range:   {min_correctness:.3f} - {max_correctness:.3f}"
        print(line1)
        print(line2)
        summary_output.append(line1)
        summary_output.append(line2)

        # Critical errors and missing points
        total_critical_errors = summary_df['CriticalErrorCount'].sum()
        total_missing = summary_df['MissingCount'].sum()
        evals_with_critical = (summary_df['CriticalErrorCount'] > 0).sum()
        print(f"\nIssues Found:")
        summary_output.append("")
        summary_output.append("Issues Found:")
        line1 = f"   Total critical errors: {total_critical_errors} (in {evals_with_critical} evaluations)"
        line2 = f"   Total missing points:  {total_missing}"
        print(line1)
        print(line2)
        summary_output.append(line1)
        summary_output.append(line2)

        # Confidence statistics (if available)
        if not details_df.empty and 'EvalConfidence' in details_df.columns:
            confidence_values = details_df['EvalConfidence'].replace('N/A', None).dropna()
            if len(confidence_values) > 0:
                confidence_values = confidence_values.astype(float)
                avg_confidence = confidence_values.mean()
                low_confidence_count = (confidence_values < 0.5).sum() // num_models  # Divide by models since it's duplicated
                print(f"\nEvaluation Confidence:")
                summary_output.append("")
                summary_output.append("Evaluation Confidence:")
                line = f"   Average confidence: {avg_confidence:.3f}"
                print(line)
                summary_output.append(line)
                if low_confidence_count > 0:
                    line = f"   Questions needing review (confidence < 0.5): {low_confidence_count}"
                    print(line)
                    summary_output.append(line)

        # Agreement levels
        if not details_df.empty and 'AgreementLevel' in details_df.columns:
            agreement_by_question = details_df.drop_duplicates('QID')[['AgreementLevel']]
            agreement_counts = agreement_by_question['AgreementLevel'].value_counts()
            print(f"\nCross-Model Agreement:")
            summary_output.append("")
            summary_output.append("Cross-Model Agreement:")
            for level, count in agreement_counts.items():
                percentage = (count / num_questions) * 100
                line = f"   {level:8s} agreement: {count:2d} questions ({percentage:5.1f}%)"
                print(line)
                summary_output.append(line)

        # Contradictions and unique facts
        if not flags_df.empty:
            contradictions = flags_df[flags_df['Type'] == 'Contradiction']
            unique_facts = flags_df[flags_df['Type'] == 'Unique Facts']
            print(f"\nCross-Model Analysis:")
            summary_output.append("")
            summary_output.append("Cross-Model Analysis:")
            line1 = f"   Contradictions found: {len(contradictions)}"
            line2 = f"   Unique facts identified: {len(unique_facts)}"
            print(line1)
            print(line2)
            summary_output.append(line1)
            summary_output.append(line2)

        # Top performers
        model_avg_scores = summary_df.groupby('Model')['Correctness'].mean().sort_values(ascending=False)
        print(f"\nModel Performance (Average Correctness):")
        summary_output.append("")
        summary_output.append("Model Performance (Average Correctness):")
        for model, score in model_avg_scores.items():
            line = f"   {model:30s}: {score:.3f}"
            print(line)
            summary_output.append(line)

        # Questions with issues
        questions_with_errors = summary_df[summary_df['CriticalErrorCount'] > 0].groupby('QID').size()
        if len(questions_with_errors) > 0:
            print(f"\nQuestions with Critical Errors: {len(questions_with_errors)}")
            summary_output.append("")
            summary_output.append(f"Questions with Critical Errors: {len(questions_with_errors)}")
            for qid in questions_with_errors.index[:3]:  # Show up to 3
                question_text = summary_df[summary_df['QID'] == qid]['Question'].iloc[0]
                models_with_errors = summary_df[(summary_df['QID'] == qid) & (summary_df['CriticalErrorCount'] > 0)]['Model'].tolist()
                line1 = f"   {qid}: {question_text[:60]}..."
                line2 = f"      Models: {', '.join(models_with_errors)}"
                print(line1)
                print(line2)
                summary_output.append(line1)
                summary_output.append(line2)

        # Questions with perfect scores
        perfect_questions = summary_df[summary_df['Correctness'] == 1.0].groupby('QID').size()
        all_perfect = perfect_questions[perfect_questions == num_models]
        if len(all_perfect) > 0:
            print(f"\nQuestions with Perfect Scores (all models 1.0): {len(all_perfect)}")
            summary_output.append("")
            summary_output.append(f"Questions with Perfect Scores (all models 1.0): {len(all_perfect)}")
            for qid in all_perfect.index[:3]:  # Show up to 3
                question_text = summary_df[summary_df['QID'] == qid]['Question'].iloc[0]
                line = f"   {qid}: {question_text[:60]}..."
                print(line)
                summary_output.append(line)

    # Write summary to text file
    summary_file = os.path.join(args.outdir, "eval_summary.txt")
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_output))

    print("\n" + "="*80)
    print("OUTPUT FILES")
    print("="*80)
    print(" -", jsonl_path)
    print(" -", sum_csv)
    print(" -", diff_csv)
    print(" -", diff_xlsx)
    print(" -", detail_csv)
    print(" -", detail_xlsx)
    print(" -", consolidated_file)
    print(" -", summary_file)
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
