"""
Entry point for the Mood Machine rule based mood analyzer.
"""

from typing import List

from mood_analyzer import MoodAnalyzer
from dataset import SAMPLE_POSTS, TRUE_LABELS, EVAL_LABELS
from reliability import predict_reliable

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
    run_interactive_loop()

    print("\nTip: After you explore the rule based model here,")
    print("run `python ml_experiments.py` to try a simple ML based model")
    print("trained on the same SAMPLE_POSTS and TRUE_LABELS.")
