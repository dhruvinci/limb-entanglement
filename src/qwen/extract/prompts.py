from __future__ import annotations

import json
from dataclasses import dataclass


ACTOR_IDS = ("A1", "A2")
PROMPT_CONDITIONS = ("identity_only", "contact_ownership", "action_relational")
THINKING_MODES = ("off", "xhigh")
SYSTEM_PROMPT = (
    "You are a visual evidence analyst. Use only evidence present in the supplied visual input. "
    "Treat A1 and A2 as temporary identities local to this input. State uncertainty rather than "
    "inventing details. Follow the requested output format."
)
PROHIBITED_NUDGES = (
    "gray", "patterned", "plain black", "red ankle", "armbar", "submitting", "defending", "trapped",
)


@dataclass(frozen=True)
class PromptRunSpec:
    condition: str
    thinking_mode: str

    def __post_init__(self) -> None:
        if self.condition not in PROMPT_CONDITIONS:
            raise ValueError(f"unknown prompt condition: {self.condition}")
        if self.thinking_mode not in THINKING_MODES:
            raise ValueError(f"unknown thinking mode: {self.thinking_mode}")


def _json_contract() -> str:
    return (
        'Return only one compact JSON object with the string fields "A1", "A2", "relation", '
        'and "uncertainty". Use "uncertain" when the visual evidence does not support a claim.'
    )


def build_grounding_conditions() -> list[dict[str, str]]:
    shared = (
        "Using only the visual evidence in this input, identify exactly the two foreground competitors "
        "and assign them the temporary identities A1 and A2. Keep each identity consistent across all "
        "frames. Ignore spectators and officials. "
    )
    return [
        {
            "condition": "identity_only",
            "prompt_id": "identity_only",
            "text": shared + (
                "Briefly state the observable evidence that distinguishes A1 from A2 without interpreting "
                "what they are doing. " + _json_contract()
            ),
        },
        {
            "condition": "contact_ownership",
            "prompt_id": "contact_ownership",
            "text": shared + (
                "At visible contact, crossing, or occlusion regions, describe the visible body-part ownership "
                "for A1 and A2. Mark any unsupported assignment as ambiguous and do not infer hidden pixels. "
                + _json_contract()
            ),
        },
        {
            "condition": "action_relational",
            "prompt_id": "action_relational",
            "text": shared + (
                "Identify the physical action and participant roles only when supported by visible evidence, "
                "and describe the relationship between A1 and A2 at contact regions. Do not infer hidden pixels. "
                + _json_contract()
            ),
        },
    ]


def grounding_condition(condition: str) -> dict[str, str]:
    try:
        return next(item for item in build_grounding_conditions() if item["condition"] == condition)
    except StopIteration as exc:
        raise ValueError(f"unknown prompt condition: {condition}") from exc


def build_actor_query(actor_id: str, marker: str, condition: str = "identity_only") -> dict[str, str]:
    if actor_id not in ACTOR_IDS:
        raise ValueError(f"unknown actor identity: {actor_id}")
    if condition not in PROMPT_CONDITIONS:
        raise ValueError(f"unknown prompt condition: {condition}")
    return {
        "prompt_id": f"{condition}_{actor_id}",
        "condition": condition,
        "actor_id": actor_id,
        "marker": marker,
        "text": (
            f"For {actor_id} as you identified in the preceding JSON, identify every visible region that "
            f"physically belongs to {actor_id}. Maintain physical ownership through contact and occlusion. "
            "Ignore spectators and officials. Do not infer hidden pixels."
        ),
    }


def build_actor_queries(marker: str, condition: str = "identity_only") -> list[dict[str, str]]:
    return [build_actor_query(actor_id, marker, condition=condition) for actor_id in ACTOR_IDS]


def build_primary_prompt_runs() -> list[PromptRunSpec]:
    return [PromptRunSpec(condition, "off") for condition in PROMPT_CONDITIONS]


def build_prompt_runs_for_context(context: str, artifact_prefix: str = "") -> list[PromptRunSpec]:
    runs = build_primary_prompt_runs()
    if context == "4fps" and not artifact_prefix:
        runs.append(PromptRunSpec("action_relational", "xhigh"))
    return runs


def build_format_repair_prompt() -> str:
    return (
        'Reformat your previous answer as exactly one JSON object with the string fields "A1", "A2", '
        '"relation", and "uncertainty". Preserve its meaning, do not add new visual claims, and use '
        '"uncertain" for missing information.'
    )


def parse_grounding_answer(answer: str) -> dict[str, str] | None:
    decoder = json.JSONDecoder()
    for position, character in enumerate(answer):
        if character != "{":
            continue
        try:
            value, _end = decoder.raw_decode(answer[position:])
        except json.JSONDecodeError:
            continue
        if not isinstance(value, dict):
            continue
        required = ("A1", "A2", "relation", "uncertainty")
        if all(isinstance(value.get(key), str) and value[key].strip() for key in required):
            return {key: value[key].strip() for key in required}
    return None


def canonical_grounding_answer(answer: str) -> str | None:
    parsed = parse_grounding_answer(answer)
    if parsed is None:
        return None
    return json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def validate_prompt_contract(
    system_prompt: str,
    conditions: list[dict[str, str]],
    queries: list[dict[str, str]],
) -> None:
    for item in queries:
        if item.get("marker") == "<SEG>" or "<seg>" in item.get("text", "").lower():
            raise ValueError("a new untrained <SEG> token is prohibited")
        if not item.get("marker"):
            raise ValueError("marker must be an existing non-empty vocabulary string")
    if [item.get("condition") for item in conditions] != list(PROMPT_CONDITIONS):
        raise ValueError("grounding conditions must contain the frozen three-condition matrix")
    expected = {(condition, actor) for condition in PROMPT_CONDITIONS for actor in ACTOR_IDS}
    actual = {(item.get("condition"), item.get("actor_id")) for item in queries}
    if actual != expected:
        raise ValueError("actor queries must contain A1 and A2 for every grounding condition")
    combined = " ".join([system_prompt, *(item.get("text", "") for item in conditions), *(item.get("text", "") for item in queries)]).lower()
    for prohibited in PROHIBITED_NUDGES:
        if prohibited in combined:
            raise ValueError(f"human-provided visual/action answer nudge is prohibited: {prohibited}")
    for item in queries:
        if item["actor_id"] not in item["text"] or "as you identified" not in item["text"]:
            raise ValueError("actor query must bind to Qwen's own preceding identity assignment")
