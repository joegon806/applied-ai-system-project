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

from typing import Dict, List, Optional, Set

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


# ---------------------------------------------------------------------
# Single-post primitives
#
# Each returns a list of failure messages for ONE post (empty == passed).
# They take an analyzer so a caller (e.g. main.py's interactive loop) can
# reuse one instance, and they need no ground-truth label — so they work
# on text a user just typed in.
# ---------------------------------------------------------------------

def determinism_failures(
    analyzer: MoodAnalyzer,
    post: str,
    runs: int = 5,
) -> List[str]:
    """Same input must always produce the same label."""
    labels = {analyzer.predict_label(post) for _ in range(runs)}
    if len(labels) > 1:
        return [
            f'"{post}" produced {len(labels)} different labels across '
            f'{runs} runs: {sorted(labels)}'
        ]
    return []


def invariance_failures(analyzer: MoodAnalyzer, post: str) -> List[str]:
    """
    Metamorphic checks: transformations that should NOT change the label.
    Asserts a relationship (transformed == original), so no label needed.

      - case:        "I LOVE it" should equal "i love it"
      - whitespace:  extra/leading/trailing spaces should not matter
      - punctuation: a trailing "!" should not flip the label
    """
    transforms = [
        ("case", lambda s: s.upper()),
        ("whitespace", lambda s: "   " + s.replace(" ", "   ") + "   "),
        ("trailing-punctuation", lambda s: s + "!"),
    ]
    base = analyzer.predict_label(post)
    failures = []
    for name, transform in transforms:
        variant = analyzer.predict_label(transform(post))
        if base != variant:
            failures.append(
                f'[{name}] "{post}" -> {base}, but transformed -> {variant}'
            )
    return failures


def directional_failures(analyzer: MoodAnalyzer, post: str) -> List[str]:
    """
    Directional metamorphic checks on the raw score:
      - adding a strong positive word must NOT lower the score,
      - adding a strong negative word must NOT raise the score.

    The added word follows a ". " so a punctuation token resets any pending
    negation first. That makes the property hold for ANY input, even text
    that ends in a negation word like "not".
    """
    base = analyzer.score_text(post)
    with_pos = analyzer.score_text(post + ". love")
    with_neg = analyzer.score_text(post + ". hate")

    failures = []
    if with_pos < base:
        failures.append(
            f'adding "love" LOWERED the score for "{post}": {base} -> {with_pos}'
        )
    if with_neg > base:
        failures.append(
            f'adding "hate" RAISED the score for "{post}": {base} -> {with_neg}'
        )
    return failures


def check_single_post(post: str, analyzer: Optional[MoodAnalyzer] = None) -> List[str]:
    """
    Run every ground-truth-free reliability check on ONE post and return a
    combined list of warnings (empty == all clear).

    Designed for main.py's interactive loop: pass the text the user typed
    and surface any reliability warnings next to the prediction.
    """
    analyzer = analyzer if analyzer is not None else MoodAnalyzer()
    warnings: List[str] = []
    warnings += determinism_failures(analyzer, post)
    warnings += invariance_failures(analyzer, post)
    warnings += directional_failures(analyzer, post)
    return warnings


# ---------------------------------------------------------------------
# Reliability-gated prediction (the INTEGRATION layer)
#
# This is where reliability stops being a report and starts driving the
# app. Instead of trusting a single predict_label() call, predict_reliable
# probes the prediction's stability and turns that into a confidence score.
# When confidence is too low the system ABSTAINS -- it emits "uncertain"
# instead of a label it cannot stand behind. The label a user sees is
# therefore *produced by* the reliability analysis, not annotated after it.
# ---------------------------------------------------------------------

# Below this confidence the model declines to commit to a label.
UNCERTAINTY_THRESHOLD = 0.40

UNCERTAIN_LABEL = "uncertain"


class ReliablePrediction:
    """The outcome of a reliability-gated prediction."""

    def __init__(
        self,
        label: str,
        confidence: float,
        base_label: str,
        signals: Dict[str, float],
    ) -> None:
        self.label = label            # what the system commits to (may be "uncertain")
        self.confidence = confidence  # 0.0 - 1.0
        self.base_label = base_label  # the naive predict_label() result
        self.signals = signals        # the individual reliability signals
        self.abstained = label == UNCERTAIN_LABEL

    def format(self) -> str:
        """Human-readable label + confidence for display in the app."""
        pct = f"{self.confidence * 100:.0f}%"
        if self.abstained:
            # Still reveal which way it leaned, but do not commit to it.
            return f"{UNCERTAIN_LABEL} ({pct} confidence, leaning {self.base_label})"
        return f"{self.label} ({pct} confidence)"


def _leave_one_out_stability(analyzer: MoodAnalyzer, tokens: List[str], base_label: str) -> float:
    """
    Fraction of single-token removals that leave the label unchanged.

    If the label survives dropping any one token, the prediction rests on
    broad evidence (stable). If removing one word flips it, the prediction
    hangs by a thread (fragile). A single-token input is treated as maximally
    fragile: there is nothing holding the label up but that one word.
    """
    if not tokens:
        return 1.0  # empty -> always neutral, trivially stable
    if len(tokens) == 1:
        return 0.0

    survived = 0
    for i in range(len(tokens)):
        reduced = " ".join(tokens[:i] + tokens[i + 1:])
        if analyzer.predict_label(reduced) == base_label:
            survived += 1
    return survived / len(tokens)


def predict_reliable(
    text: str,
    analyzer: Optional[MoodAnalyzer] = None,
    threshold: float = UNCERTAINTY_THRESHOLD,
) -> ReliablePrediction:
    """
    Predict a mood label AND decide whether the model should trust it.

    Confidence is built from three reliability signals, all reusing the
    same ideas as the standalone checks:

      consistency : do metamorphic variants (case / whitespace / trailing
                    punctuation) agree with the base label? A disagreement
                    is a genuine invariance failure and should gut trust,
                    so it multiplies the score (a hard gate).
      stability   : leave-one-out robustness -- does the label survive
                    dropping any single token?
      strength    : how decisive the signal is. A large score margin is
                    strong; a score of 0 caused by positive and negative
                    words CANCELING (a conflict) is the weakest of all.

    If the resulting confidence is below `threshold`, the system abstains
    and emits "uncertain" rather than committing to `base_label`.
    """
    analyzer = analyzer if analyzer is not None else MoodAnalyzer()

    base_label = analyzer.predict_label(text)
    score, positive_count, negative_count = analyzer._score_details(text)
    tokens = analyzer.preprocess(text)
    margin = abs(score)

    # 1. Metamorphic consistency -- these transforms SHOULD NOT change the
    #    label, so a disagreement is a real reliability failure (hard gate).
    variants = [
        text.upper(),
        "   " + text.replace(" ", "   ") + "   ",
        text + "!",
    ]
    agree = sum(1 for v in variants if analyzer.predict_label(v) == base_label)
    consistency = agree / len(variants) if variants else 1.0

    # 2. Leave-one-out stability.
    stability = _leave_one_out_stability(analyzer, tokens, base_label)

    # 3. Signal strength.
    conflict = positive_count > 0 and negative_count > 0
    if score != 0:
        strength = min(margin, 3) / 3.0          # 0.33 / 0.67 / 1.0
    elif conflict:
        strength = 0.05                          # positives and negatives cancel out
    else:
        strength = 0.50                          # genuine, quiet neutral

    confidence = (0.45 * stability + 0.55 * strength) * consistency

    label = base_label if confidence >= threshold else UNCERTAIN_LABEL

    signals = {
        "consistency": consistency,
        "stability": stability,
        "strength": strength,
    }
    return ReliablePrediction(label, confidence, base_label, signals)


# ---------------------------------------------------------------------
# Dataset-level checks (aggregate the single-post primitives)
# ---------------------------------------------------------------------

def check_determinism(posts: List[str], runs: int = 5) -> ReliabilityResult:
    """
    Every post must give a stable label. A rule-based model has no
    randomness, so this should be rock solid; a failure means something
    stateful or nondeterministic crept in.
    """
    result = ReliabilityResult("Determinism (same input -> same label)")
    analyzer = MoodAnalyzer()
    for post in posts:
        for message in determinism_failures(analyzer, post, runs):
            result.fail(message)
    if result.passed:
        result.note(f"All {len(posts)} posts gave a stable label across {runs} runs.")
    return result


def check_invariance(posts: List[str]) -> ReliabilityResult:
    """Harmless edits (case, whitespace, trailing punctuation) keep the label."""
    result = ReliabilityResult("Invariance (harmless edits keep the label)")
    analyzer = MoodAnalyzer()
    for post in posts:
        for message in invariance_failures(analyzer, post):
            result.fail(message)
    if result.passed:
        result.note(
            f"All {len(posts)} posts survived case, whitespace, and "
            f"trailing-punctuation edits unchanged."
        )
    return result


def check_directional(posts: List[str]) -> ReliabilityResult:
    """Adding a positive word can't lower the score; a negative word can't raise it."""
    result = ReliabilityResult("Directional monotonicity (score moves the right way)")
    analyzer = MoodAnalyzer()
    for post in posts:
        for message in directional_failures(analyzer, post):
            result.fail(message)
    if result.passed:
        result.note(f"Score moved in the expected direction for all {len(posts)} posts.")
    return result


def print_result(result: ReliabilityResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    print(f"[{status}] {result.name}")
    for message in result.messages:
        print(f"    {message}")
    print()

def evaluate_rule_based(
    posts: List[str],
    labels: List[str],
    original_labels: List[str] = None,
) -> float:
    """
    Evaluate the rule based MoodAnalyzer on a labeled dataset.

    `labels` are the labels the model is scored against — pass the
    NORMALIZED EVAL_LABELS so the four-label model is judged fairly (see
    LABEL_MAP in dataset.py and reliability.py for why). `original_labels`
    is the optional richer human label shown alongside for transparency.

    Predictions come from predict_reliable(), so the label scored here is
    the reliability-gated one: on low-confidence inputs the model emits
    "uncertain" instead of guessing, which counts as incorrect. Alongside
    the overall accuracy we also report accuracy on the CONFIDENT subset
    (the answers the model actually stood behind).
    """
    analyzer = MoodAnalyzer()
    correct = 0
    total = len(posts)

    committed = 0          # predictions where the model did NOT abstain
    committed_correct = 0
    abstained = 0

    if original_labels is None:
        original_labels = labels

    print("=== Rule Based Evaluation on SAMPLE_POSTS ===")
    for text, true_label, human_label in zip(posts, labels, original_labels):
        prediction = predict_reliable(text, analyzer)
        predicted_label = prediction.label
        is_correct = predicted_label == true_label
        if is_correct:
            correct += 1

        if prediction.abstained:
            abstained += 1
        else:
            committed += 1
            if is_correct:
                committed_correct += 1

        # Show the reliability-gated label + confidence, plus the original
        # human label when it differs (e.g. "sarcastic" -> "negative").
        human_note = f" (human: {human_label})" if human_label != true_label else ""
        print(f'"{text}" -> {prediction.format()}, '
              f'true={true_label}{human_note}')

    if total == 0:
        print("\nNo labeled examples to evaluate.")
        return 0.0

    accuracy = correct / total
    print(f"\nRule based accuracy on SAMPLE_POSTS: {accuracy:.2f} "
          f"({correct}/{total})")
    print(f"Abstained (uncertain): {abstained}/{total}")
    if committed:
        print(f"Accuracy on confident answers only: "
              f"{committed_correct / committed:.2f} "
              f"({committed_correct}/{committed})")
    print("")
    return accuracy

def main() -> int:

    print("=== Mood Machine Reliability Report ===\n")
    results = [
        check_lengths_aligned(SAMPLE_POSTS, TRUE_LABELS),
        check_label_map_coverage(TRUE_LABELS, LABEL_MAP),
        # Validate the NORMALIZED labels the rule-based model is scored on.
        check_label_space(EVAL_LABELS),
        # Consistency checks (no ground-truth labels needed).
        check_determinism(SAMPLE_POSTS),
        check_invariance(SAMPLE_POSTS),
        check_directional(SAMPLE_POSTS),
    ]

    for result in results:
        print_result(result)

    failures = [r for r in results if not r.passed]
    if failures:
        print(f"{len(failures)} check(s) FAILED: " +
              ", ".join(r.name for r in failures))
        return 1

    print("All reliability checks passed.")
    
    # Score against normalized labels; show the original human labels too.
    evaluate_rule_based(SAMPLE_POSTS, EVAL_LABELS, TRUE_LABELS)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
