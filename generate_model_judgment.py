import argparse
import copy
import json
import os
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))
from request_api import request_internal_conv_api


DEFAULT_MODEL = "gpt-5"
DEFAULT_TEMPERATURE = 0.4
DEFAULT_MAX_TOKENS = 16 * 1024
DEFAULT_NUM_THREADS = 64
DEFAULT_OUTER_RETRIES = 3


CUSTOM_SYSTEM_PROMPT = """Act as an impartial judge and decide which of two AI assistants (A vs B) better answers the given USER PROMPT.

You will compare A and B **relatively** without consulting external sources. Do not add new facts beyond what is reasonably common knowledge; do not browse.

## What to evaluate (in order of priority)
1) **Factual accuracy & correctness.** Briefly point out any factual errors (one sentence each). Do not speculate.
2) **Instruction-following & task completion.** Does the answer do what the prompt asked?
3) **Relevance & completeness.** Does it cover all requested parts without going off-topic?
4) **Clarity & concision.** Prefer precise, organized, and succinct answers over verbosity.
5) **Safety & policy alignment.** Prefer safe refusals with a brief rationale over unsafe compliance.
6) **Creativity (only if prompt is open-ended/creative).** Novelty and appropriateness of ideas.

## Special cases
- **Materiality principle.** Count only differences that materially affect usefulness to the user (e.g., change the conclusion, fail a key instruction, or introduce a safety/policy risk) as decisive. Do **not** treat style, minor wording, or secondary elaboration as decisive differences unless the content is otherwise identical.
- **Tie-breaking policy.**
  1) If there is **no material difference** on the top three priorities (accuracy, instruction-following, relevance/completeness), you must use lower-priority dimensions (clarity/length/minor formatting) to determine the winner.
  2) You must choose either A or B. Do not output a tie.
- **Ambiguity:** If the USER PROMPT is genuinely ambiguous, reward the assistant that *briefly* identifies the ambiguity and asks a focused clarifying question or states reasonable assumptions.
- **Language:** Match the USER PROMPT language(s). Penalize unprompted code-switching. Bilingual output is fine only if the user asked or the prompt includes multiple languages.
- **Formatting/length bias:** Do not reward unnecessary length or decorative formatting.

## How to structure your explanation
- **Requirements extracted from the USER PROMPT:** 2–5 bullet points.
- **Assistant A – strengths & weaknesses:** 3–6 bullets.
- **Assistant B – strengths & weaknesses:** 3–6 bullets.
- **Head-to-head comparison:** 2–4 bullets stating the decisive differences.
- **Missing but useful information (if any):** 1–3 bullets.

After your explanation, output the final verdict by wrapping the letter of the better assistant in \\boxed{}. Do not output any other text after the box.

Example:
\\boxed{A}
or
\\boxed{B}
"""


def mock_dialogue_context(conversation_list: List[Dict[str, Any]]) -> str:
    formatted_turns: List[str] = []
    for turn in conversation_list:
        role = turn.get("role", "")
        content = turn.get("content", "")
        if role == "user":
            formatted_turns.append(f"User: {content}")
        elif role == "assistant":
            formatted_turns.append(f"Assistant: {content}")
        else:
            formatted_turns.append(f"{str(role).capitalize()}: {content}")
    return "\n".join(formatted_turns)


def format_judge_answers_custom(
    conversation_list: List[Dict[str, Any]],
    answer_a_text: str,
    answer_b_text: str,
    inverse_order: bool = False,
) -> Tuple[str, str]:
    system_prompt = CUSTOM_SYSTEM_PROMPT

    conv_context = ""
    if len(conversation_list) > 1:
        conv_context = (
            f"<|Dialogue Context|>\n{mock_dialogue_context(conversation_list[:-1])}\n\n"
            f"<|End of Dialogue Context|>\n\n"
        )

    user_last_query = ""
    if len(conversation_list) > 0:
        last_turn = conversation_list[-1]
        user_last_query = f"<|User Prompt|>\n{last_turn.get('content', '')}\n\n"
    else:
        user_last_query = "<|User Prompt|>\n\n"

    if inverse_order:
        section_a = (
            f"<|The Start of Assistant A's Answer with User|>\n\n"
            f"{answer_b_text}\n\n"
            f"<|The End of Assistant A's Answer with User|>\n\n"
        )
        section_b = (
            f"<|The Start of Assistant B's Answer with User|>\n\n"
            f"{answer_a_text}\n\n"
            f"<|The End of Assistant B's Answer with User|>\n\n"
        )
    else:
        section_a = (
            f"<|The Start of Assistant A's Answer with User|>\n\n"
            f"{answer_a_text}\n\n"
            f"<|The End of Assistant A's Answer with User|>\n\n"
        )
        section_b = (
            f"<|The Start of Assistant B's Answer with User|>\n\n"
            f"{answer_b_text}\n\n"
            f"<|The End of Assistant B's Answer with User|>\n\n"
        )

    user_prompt = conv_context + user_last_query + section_a + section_b
    return system_prompt, user_prompt


def process_judgement_custom(judgment: str) -> Optional[str]:
    """Extract final verdict: \\boxed{A} / \\boxed{B} (or boxed{A/B})."""
    if not judgment:
        return None
    m = re.search(r"\\boxed\{([AB])\}", judgment)
    if m:
        return m.group(1)
    m = re.search(r"boxed\{([AB])\}", judgment)
    if m:
        return m.group(1)
    return None


def _sanitize_model_name_for_filename(model: str) -> str:
    s = (model or "").strip()
    if not s:
        return "unknown_model"
    s = s.replace("/", "__").replace("\\", "__")
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9A-Za-z._\-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "unknown_model"


def _derive_output_path(input_path: Path, model: str) -> Path:
    model_part = _sanitize_model_name_for_filename(model)
    result_dir = Path(__file__).resolve().parent / "result"
    return result_dir / f"{input_path.stem}__{model_part}.jsonl"


def _build_api_conversation(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    conversations = item["conversations"]
    answer_a_text = item["response_A"]
    answer_b_text = item["response_B"]
    system_prompt, user_prompt = format_judge_answers_custom(
        conversations, answer_a_text, answer_b_text, inverse_order=False
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _get_one_model_judgment_with_retry(
    item: Dict[str, Any],
    *,
    model: str,
    base_url: str,
    api_key: Optional[str],
    temperature: float,
    max_tokens: int,
    outer_retries: int,
) -> Optional[str]:
    conversation = _build_api_conversation(item)
    last_error: Optional[str] = None
    for attempt in range(outer_retries):
        try:
            response_text = request_internal_conv_api(
                conversation=conversation,
                internal_api_model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                base_url=base_url,
                api_key=api_key,
                return_raw_requst_res=False,
                concat_resoning_content=True,
            )
            if response_text and len(response_text) > 0:
                return response_text
            last_error = "empty_response"
        except Exception as e:
            last_error = f"api_error: {e}"

        if attempt < outer_retries - 1:
            time.sleep(random.uniform(1, 3))

    sample_id = item.get("id", "<no-id>")
    subset = None
    try:
        subset = (item.get("source") or {}).get("subset")
    except Exception:
        subset = None
    print(
        f"[warn] failed to get model_judgment after retries={outer_retries} "
        f"id={sample_id} subset={subset} last_error={last_error}"
    )
    return None


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input_jsonl",
        type=str,
        required=True,
        help="Input JSONL (must include conversations/response_A/response_B/gt_label).",
    )
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--base_url", type=str, default=os.environ.get("OPENAI_BASE_URL", ""))
    p.add_argument("--api_key", type=str, default=os.environ.get("OPENAI_API_KEY", ""))
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    p.add_argument("--num_threads", type=int, default=DEFAULT_NUM_THREADS)
    p.add_argument("--retries", type=int, default=DEFAULT_OUTER_RETRIES)
    p.add_argument("--overwrite", action="store_true", help="Allow overwriting existing output file.")
    p.add_argument("--debug", action="store_true", help="Only run the first 10 samples.")
    return p.parse_args()


def main() -> None:
    args = get_args()
    input_path = Path(args.input_jsonl)
    if not input_path.exists():
        raise FileNotFoundError(f"input_jsonl not found: {input_path}")
    if not args.base_url:
        raise ValueError("base_url is required (pass --base_url or set OPENAI_BASE_URL).")

    output_path = _derive_output_path(input_path, args.model)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"output_jsonl already exists: {output_path}\n"
            "Refusing to overwrite. Use --overwrite if you really want to overwrite."
        )

    print(f"[config] input_jsonl : {input_path}")
    print(f"[config] output_jsonl: {output_path}")
    print(
        f"[config] model={args.model} base_url={args.base_url} "
        f"temp={args.temperature} max_tokens={args.max_tokens}"
    )
    print(f"[config] num_threads={args.num_threads} retries={args.retries}")
    if args.debug:
        print("[config] debug=True (only first 10 samples)")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    items: List[Dict[str, Any]] = []
    total_bad_json = 0
    with open(input_path, "r", encoding="utf-8") as in_f:
        for line in in_f:
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except Exception:
                total_bad_json += 1
                continue

    if args.debug:
        items = items[:10]

    results: List[Optional[str]] = [None for _ in range(len(items))]
    if len(items) > 0:
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            future_to_i: Dict[Any, int] = {}
            for i in range(len(items)):
                fut = executor.submit(
                    _get_one_model_judgment_with_retry,
                    items[i],
                    model=args.model,
                    base_url=args.base_url,
                    api_key=args.api_key or None,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                    outer_retries=args.retries,
                )
                future_to_i[fut] = i

            with tqdm(total=len(items), desc="Processing") as pbar:
                for future in as_completed(future_to_i):
                    i = future_to_i[future]
                    try:
                        results[i] = future.result()
                    except Exception as e:
                        print(f"[warn] future_exception i={i} id={items[i].get('id','<no-id>')} err={e}")
                        results[i] = None
                    pbar.update(1)

    total_written = 0
    total_parse_fail = 0
    total_call_fail = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for item, model_judgment in zip(items, results):
            out_item = copy.deepcopy(item)
            out_item["model_judgment"] = model_judgment
            pred = process_judgement_custom(model_judgment or "")
            if pred is None:
                total_parse_fail += 1
            out_item["model_pred"] = pred

            if model_judgment is None:
                total_call_fail += 1

            if not isinstance(out_item.get("meta"), dict):
                out_item["meta"] = {}
            out_item["meta"]["model_name"] = args.model

            out_f.write(json.dumps(out_item, ensure_ascii=False) + "\n")
            total_written += 1

    print(
        "[done] "
        f"read={len(items)} written={total_written} bad_json={total_bad_json} "
        f"call_fail={total_call_fail} parse_fail={total_parse_fail}"
    )
    print(f"[done] saved to: {output_path}")


if __name__ == "__main__":
    main()

