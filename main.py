"""
Entry point for the Mood Machine rule based mood analyzer.
"""

from typing import List

from mood_analyzer import MoodAnalyzer
from dataset import SAMPLE_POSTS, TRUE_LABELS, EVAL_LABELS
from reliability import predict_reliable


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
    return accuracy


def run_batch_demo() -> None:
    """
    Run the MoodAnalyzer on the sample posts and print predictions only.

    This is a quick way to see how your rules behave without comparing
    to the true labels.
    """
    analyzer = MoodAnalyzer()
    print("\n=== Batch Demo on SAMPLE_POSTS (rule based) ===")
    for text in SAMPLE_POSTS:
        prediction = predict_reliable(text, analyzer)
        print(f'"{text}" -> {prediction.format()}')


def run_interactive_loop() -> None:
    """
    Let the user type their own sentences and see the predicted mood.

    Type 'quit' or press Enter on an empty line to exit.
    """
    analyzer = MoodAnalyzer()
    print("\n=== Interactive Mood Machine (rule based) ===")
    print("Type a sentence to analyze its mood.")
    print("Type 'quit' or press Enter on an empty line to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input == "" or user_input.lower() == "quit":
            print("Goodbye from the Mood Machine.")
            break

        # The reliability layer decides the answer: it commits to a label
        # only when its consistency/stability/strength signals are strong
        # enough, otherwise it abstains with "uncertain".
        prediction = predict_reliable(user_input, analyzer)
        print(f"Model: {prediction.format()}")
        if prediction.abstained:
            print("  (low confidence — the signals for this text are weak "
                  "or conflicting, so the model will not commit to a mood.)")


if __name__ == "__main__":
    # Score against normalized labels; show the original human labels too.
    evaluate_rule_based(SAMPLE_POSTS, EVAL_LABELS, TRUE_LABELS)

    run_batch_demo()

    run_interactive_loop()

    print("\nTip: After you explore the rule based model here,")
    print("run `python ml_experiments.py` to try a simple ML based model")
    print("trained on the same SAMPLE_POSTS and TRUE_LABELS.")
