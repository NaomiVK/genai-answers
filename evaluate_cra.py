import os, json, argparse, hashlib
import pandas as pd
import numpy as np
from tqdm import tqdm
from openai import OpenAI

# ---------- Config ----------
DEFAULT_MODEL = "anthropic/claude-opus-4.1"  # OpenRouter model: Claude Opus 4.1
REQUIRED_FIELDS = ["Question"]  # will auto-detect model columns if not provided

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

def _load_env_from_dotenv_or_file():
    """Load environment variables from a .env file if present.

    Tries python-dotenv if installed; otherwise performs a minimal parse of
    KEY=VALUE lines in a local .env file. Existing env vars are not overridden.
    """
    # Try python-dotenv first (no hard dependency)
    try:
        from dotenv import load_dotenv  # type: ignore
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
"""

def _call_openrouter_api(client: OpenAI, model: str, prompt: str):
    """Call OpenRouter API using chat completions.

    Returns the parsed JSON object according to EVAL_SCHEMA.
    """
    # OpenRouter uses the chat completions API
    # Note: OpenRouter may not support response_format for all models
    # Instead, we'll ask for JSON in the prompt
    json_instruction = "\n\nIMPORTANT: Return your response as a valid JSON object only, with no additional text or markdown formatting."

    kwargs = {
        'model': model,
        'messages': [
            {'role': 'system', 'content': JUDGE_PREAMBLE + json_instruction},
            {'role': 'user', 'content': prompt}
        ],
        'temperature': 0.1,  # Low temperature for more consistent JSON output
        'max_tokens': 4000  # Reasonable limit for evaluation responses
    }

    try:
        resp = client.chat.completions.create(**kwargs)
    except Exception as e:
        print(f"[ERROR] OpenRouter API call failed: {e}")
        raise

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
    """
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
        for name, val in pm.items():
            if isinstance(val, dict):
                tmp = dict(val)
                tmp.setdefault("name", name)
                items.append(tmp)
    # If still empty, synthesize uncertain entries for provided model columns
    if not items and model_cols:
        for name in model_cols:
            items.append({"name": name, "verdict": "uncertain", "correctness_score": 0.0})

    normalized = []
    for idx, it in enumerate(items):
        name = it.get("name") or it.get("model") or "model"

        # If we got generic "model" but have the actual model names, use them
        if name == "model" and model_cols and idx < len(model_cols):
            name = model_cols[idx]

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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="genai-answers.xlsx", help="Path to Excel workbook")
    ap.add_argument("--sheet", default=None, help="Sheet name (default: all)")
    ap.add_argument("--models", default=None,
                    help="Comma-separated model column names to evaluate (default: auto-detect long text columns)")
    ap.add_argument("--outdir", default="eval_out", help="Output directory")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Judge model (default: anthropic/claude-opus-4.1)")
    ap.add_argument("--max-rows", type=int, default=None, help="Evaluate at most this many rows per sheet")
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

                prompt = make_prompt(qid, question, model_answers)

                parsed = _call_openrouter_api(client, args.model, prompt)
                parsed = _coerce_to_eval_schema(parsed, qid, question, [d["name"] for d in model_answers])

                # write JSONL artifact
                w.write(json.dumps(parsed, ensure_ascii=False) + "\n")

                # Calculate consistency metrics for this question
                correctness_scores = [pm["correctness_score"] for pm in parsed["per_model"]]

                # Calculate consistency score (0-1, where 1 is perfect consistency)
                if len(correctness_scores) > 1:
                    # Use coefficient of variation (CV) inverted and bounded
                    mean_score = np.mean(correctness_scores)
                    std_score = np.std(correctness_scores)

                    if mean_score > 0:
                        cv = std_score / mean_score  # Coefficient of variation
                        consistency_score = max(0, 1 - cv)  # Invert so 1 is consistent
                    else:
                        consistency_score = 1.0 if std_score == 0 else 0.0
                else:
                    consistency_score = 1.0  # Single model is perfectly consistent with itself

                # Count contradictions and unique facts for this question
                num_contradictions = len(parsed["cross_model"].get("contradictions", []))
                num_unique_facts = len(parsed["cross_model"].get("facts_unique_to_model", []))

                # Calculate agreement level
                if std_score < 0.1:
                    agreement_level = "High"
                elif std_score < 0.25:
                    agreement_level = "Medium"
                else:
                    agreement_level = "Low"

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
                    all_details.append({
                        "QID": parsed["question_id"],
                        "Question": parsed["question"][:100] + "..." if len(parsed["question"]) > 100 else parsed["question"],
                        "Model": pm["name"],
                        "Verdict": pm["verdict"],
                        "Correctness": pm["correctness_score"],
                        "CriticalErrors": "; ".join(pm["critical_errors"]) if pm["critical_errors"] else "",
                        "UnsupportedAssertions": "; ".join(pm["unsupported_assertions"]) if pm["unsupported_assertions"] else "",
                        "KeyPointsCovered": "; ".join(pm["key_points_covered"]) if pm["key_points_covered"] else "",
                        "KeyPointsMissing": "; ".join(pm["key_points_missing"]) if pm["key_points_missing"] else "",
                        "Notes": pm["notes"],
                        "ConsistencyScore": round(consistency_score, 2),
                        "AgreementLevel": agreement_level,
                        "Contradictions": num_contradictions,
                        "UniqueFacts": num_unique_facts
                    })

                # cross-model flags
                for diff in parsed["cross_model"]["facts_unique_to_model"]:
                    if diff.get("facts"):  # Only add if there are actual facts
                        all_diffs.append({
                            "QID": parsed["question_id"],
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
                                "Type": "Contradiction",
                                "Model": ", ".join(models_involved[:3]) if models_involved else "Multiple",
                                "MissingFromModels": "N/A",
                                "Details": con["topic"] if con.get("topic") and con["topic"] != "unspecified" else "Disagreement detected"
                            })
                    elif con.get("topic") and con["topic"] not in ["unspecified", ""]:
                        # Even if model_positions is empty, if we have a topic, note it
                        all_diffs.append({
                            "QID": parsed["question_id"],
                            "Type": "Contradiction",
                            "Model": "Multiple",
                            "MissingFromModels": "N/A",
                            "Details": con["topic"]
                        })

                processed += 1
                if args.max_rows and processed >= args.max_rows:
                    break

    # Write CSVs
    pd.DataFrame(all_rows).to_csv(sum_csv, index=False)
    pd.DataFrame(all_diffs).to_csv(diff_csv, index=False)
    pd.DataFrame(all_details).to_csv(detail_csv, index=False)

    # Write cross_model_flags to Excel as well
    diff_xlsx = os.path.join(args.outdir, "cross_model_flags.xlsx")
    pd.DataFrame(all_diffs).to_excel(diff_xlsx, index=False, sheet_name="Cross Model Flags")

    print("Wrote:")
    print(" -", jsonl_path)
    print(" -", sum_csv)
    print(" -", diff_csv)
    print(" -", diff_xlsx)
    print(" -", detail_csv)

if __name__ == "__main__":
    main()
