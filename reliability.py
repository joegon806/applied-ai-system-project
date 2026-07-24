# reliability.py
"""
Reliability checks for the Mood Machine.

A "reliability system" is separate from accuracy. Accuracy asks "how often is
the model right?" Reliability asks questions like:
  - Are we even measuring accuracy fairly? (label-space validation)
  - Does the model give the same answer for the same input? (determinism)
  - Do harmless changes to the text leave the label unchanged? (invariance)

This file starts with LABEL-SPACE VALIDATION, which must pass before any
accuracy number is trustworthy. More checks can be added below over time.

Run:
    python reliability.py
"""

from typing import Dict, List, Set

from mood_analyzer import MoodAnalyzer
from dataset import SAMPLE_POSTS, TRUE_LABELS, EVAL_LABELS, LABEL_MAP


# ---------------------------------------------------------------------
# The labels the model is actually capable of producing.
#
# MoodAnalyzer.predict_label() can only ever return one of these four
# strings. Any "true" label outside this set is UNREACHABLE: the model
# can never match it, so those examples are guaranteed wrong no matter
# how good the model is. That makes raw accuracy misleading.
# ---------------------------------------------------------------------
MODEL_LABELS: Set[str] = {"positive", "negative", "neutral", "mixed"}


class ReliabilityResult:
    """Small container so each check can report pass/fail plus details."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.passed = True
        self.messages: List[str] = []

    def fail(self, message: str) -> None:
        self.passed = False
        self.messages.append(message)

    def note(self, message: str) -> None:
        self.messages.append(message)


def check_label_space(
    labels: List[str],
    model_labels: Set[str] = MODEL_LABELS,
) -> ReliabilityResult:
    """
    Validate that every human label in TRUE_LABELS is a label the model
    can actually output.

    Why this matters:
      MoodAnalyzer can only return {positive, negative, neutral, mixed}.
      If TRUE_LABELS contains something else (e.g. "sarcastic"), that
      example can NEVER be predicted correctly. Accuracy is then capped
      below 1.0 for reasons that have nothing to do with model quality,
      and the accuracy number quietly lies to you.

    This check reports:
      - which labels are unreachable (in the data but not producible),
      - how many posts carry an unreachable label,
      - the best accuracy still theoretically achievable.
    """
    result = ReliabilityResult("Label-space validation")

    used_labels = set(labels)
    unreachable = used_labels - model_labels

    total = len(labels)
    unreachable_count = sum(1 for label in labels if label in unreachable)

    if unreachable:
        result.fail(
            f"{len(unreachable)} label(s) in TRUE_LABELS cannot be produced "
            f"by the model: {sorted(unreachable)}"
        )
        result.note(
            f"{unreachable_count} of {total} posts carry an unreachable label "
            f"and are guaranteed to be scored wrong."
        )
        if total:
            reachable = total - unreachable_count
            result.note(
                f"Best achievable accuracy is capped at "
                f"{reachable}/{total} = {reachable / total:.2f}, "
                f"regardless of model quality."
            )
        result.note(
            "Fix options: (a) map these labels onto the model's four labels, "
            "(b) extend the model to produce them, or "
            "(c) evaluate only on posts whose labels are reachable."
        )
        # Show exactly which posts are affected so they are easy to find.
        for post, label in zip(SAMPLE_POSTS, labels):
            if label in unreachable:
                result.note(f'  unreachable: "{post}" -> labeled "{label}"')
    else:
        result.note(
            f"All {len(used_labels)} label(s) used in TRUE_LABELS are "
            f"producible by the model. Accuracy is measured fairly."
        )

    return result


def check_label_map_coverage(
    raw_labels: List[str],
    label_map: Dict[str, str],
    model_labels: Set[str] = MODEL_LABELS,
) -> ReliabilityResult:
    """
    Validate the LABEL_MAP that resolves the mismatch:
      1. Every human label in TRUE_LABELS has an entry in LABEL_MAP
         (otherwise EVAL_LABELS silently passes it through unmapped).
      2. Every value the map produces is a real model label.

    This guards the *fix* itself: if someone adds a new human label to
    TRUE_LABELS but forgets to map it, this check fails loudly instead of
    letting an unreachable label sneak back into evaluation.
    """
    result = ReliabilityResult("Label-map coverage")

    unmapped = sorted(set(raw_labels) - set(label_map))
    if unmapped:
        result.fail(
            f"{len(unmapped)} human label(s) in TRUE_LABELS are missing from "
            f"LABEL_MAP: {unmapped}. Add an entry mapping each to a model label."
        )

    bad_targets = sorted(
        {target for target in label_map.values() if target not in model_labels}
    )
    if bad_targets:
        result.fail(
            f"LABEL_MAP maps to value(s) the model cannot produce: {bad_targets}. "
            f"Targets must be one of {sorted(model_labels)}."
        )

    if result.passed:
        result.note(
            f"All {len(set(raw_labels))} human label(s) map onto valid model labels."
        )
    return result


def check_lengths_aligned(
    posts: List[str],
    labels: List[str],
) -> ReliabilityResult:
    """
    SAMPLE_POSTS and TRUE_LABELS must be the same length, or zip() will
    silently drop the tail of the longer list and every downstream metric
    is computed on mismatched pairs.
    """
    result = ReliabilityResult("Dataset length alignment")
    if len(posts) != len(labels):
        result.fail(
            f"SAMPLE_POSTS has {len(posts)} entries but TRUE_LABELS has "
            f"{len(labels)}. They must match exactly."
        )
    else:
        result.note(f"Aligned: {len(posts)} posts and {len(labels)} labels.")
    return result


def print_result(result: ReliabilityResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    print(f"[{status}] {result.name}")
    for message in result.messages:
        print(f"    {message}")
    print()


def main() -> int:
    print("=== Mood Machine Reliability Report ===\n")

    results = [
        check_lengths_aligned(SAMPLE_POSTS, TRUE_LABELS),
        check_label_map_coverage(TRUE_LABELS, LABEL_MAP),
        # Validate the NORMALIZED labels the rule-based model is scored on.
        check_label_space(EVAL_LABELS),
    ]

    for result in results:
        print_result(result)

    failures = [r for r in results if not r.passed]
    if failures:
        print(f"{len(failures)} check(s) FAILED: " +
              ", ".join(r.name for r in failures))
        return 1

    print("All reliability checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
