# mood_analyzer.py
"""
Rule based mood analyzer for short text snippets.

This class starts with very simple logic:
  - Preprocess the text
  - Look for positive and negative words
  - Compute a numeric score
  - Convert that score into a mood label
"""

import re
from typing import List, Dict, Tuple, Optional

from dataset import POSITIVE_WORDS, NEGATIVE_WORDS, NEGATION_WORDS


class MoodAnalyzer:
    """
    A very simple, rule based mood classifier.
    """

    def __init__(
        self,
        positive_words: Optional[List[str]] = None,
        negative_words: Optional[List[str]] = None,
    ) -> None:
        # Use the default lists from dataset.py if none are provided.
        positive_words = positive_words if positive_words is not None else POSITIVE_WORDS
        negative_words = negative_words if negative_words is not None else NEGATIVE_WORDS

        # Store as sets for faster lookup.
        self.positive_words = set(w.lower() for w in positive_words)
        self.negative_words = set(w.lower() for w in negative_words)
        self.negation_words = set(w.lower() for w in NEGATION_WORDS)

    # ---------------------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------------------

    def preprocess(self, text: str) -> List[str]:
        """
        Convert raw text into a list of tokens the model can work with.

        TODO: Improve this method.

        Right now, it does the minimum:
          - Strips leading and trailing whitespace
          - Converts everything to lowercase
          - Splits on spaces

        Ideas to improve:
          - Remove punctuation
          - Handle simple emojis separately (":)", ":-(", "🥲", "😂")
          - Normalize repeated characters ("soooo" -> "soo")
        """
        cleaned = text.strip().lower()
        # Pad runs of punctuation with spaces so each run becomes its own token.
        cleaned = re.sub(r"([^\w\s]+)", r" \1 ", cleaned)
        tokens = cleaned.split()

        return tokens

    # ---------------------------------------------------------------------
    # Scoring logic
    # ---------------------------------------------------------------------

    def score_text(self, text: str) -> int:
        """
        Compute a numeric "mood score" for the given text.

        Positive words increase the score.
        Negative words decrease the score.

        TODO: You must choose AT LEAST ONE modeling improvement to implement.
        For example:
          - Handle simple negation such as "not happy" or "not bad"
          - Count how many times each word appears instead of just presence
          - Give some words higher weights than others (for example "hate" < "annoyed")
          - Treat emojis or slang (":)", "lol", "💀") as strong signals
        """
        return self._score_details(text)[0]

    def _score_details(self, text: str) -> Tuple[int, int, int]:
        """
        Score the text and also report how many sentiment words contributed
        positively vs negatively (after negation is applied).

        Returns (score, positive_count, negative_count). This lets callers such
        as predict_label tell a zero score caused by *no* sentiment words apart
        from one caused by positives and negatives canceling out.
        """
        tokens = self.preprocess(text)

        score = 0
        positive_count = 0
        negative_count = 0
        negate = False  # flips the polarity of the next sentiment word
        for token in tokens:
            if token in self.negation_words:
                negate = True
                continue

            # A punctuation token (no word characters) ends the clause,
            # so negation does not carry across it.
            if not re.search(r"\w", token):
                negate = False
                continue

            sign = -1 if negate else 1
            if token in self.positive_words:
                contribution = 1 * sign
            elif token in self.negative_words:
                contribution = -1 * sign
            else:
                continue  # not a sentiment word; leave negate armed

            score += contribution
            if contribution > 0:
                positive_count += 1
            else:
                negative_count += 1
            negate = False  # consumed by the first sentiment word after it

        return score, positive_count, negative_count

    # ---------------------------------------------------------------------
    # Label prediction
    # ---------------------------------------------------------------------

    def predict_label(self, text: str) -> str:
        """
        Turn the numeric score for a piece of text into a mood label.

        The default mapping is:
          - score > 0  -> "positive"
          - score < 0  -> "negative"
          - score == 0 -> "neutral"

        TODO: You can adjust this mapping if it makes sense for your model.
        For example:
          - Use different thresholds (for example score >= 2 to be "positive")
          - Add a "mixed" label for scores close to zero
        Just remember that whatever labels you return should match the labels
        you use in TRUE_LABELS in dataset.py if you care about accuracy.
        """
        score, positive_count, negative_count = self._score_details(text)

        if score > 0:
            return "positive"
        if score < 0:
            return "negative"

        # score == 0: was it silence, or positives and negatives canceling out?
        if positive_count > 0 and negative_count > 0:
            return "mixed"
        return "neutral"

    # ---------------------------------------------------------------------
    # Explanations (optional but recommended)
    # ---------------------------------------------------------------------

    def explain(self, text: str) -> str:
        """
        Return a short string explaining WHY the model chose its label.

        TODO:
          - Look at the tokens and identify which ones counted as positive
            and which ones counted as negative.
          - Show the final score.
          - Return a short human readable explanation.

        Example explanation (your exact wording can be different):
          'Score = 2 (positive words: ["love", "great"]; negative words: [])'

        The current implementation is a placeholder so the code runs even
        before you implement it.
        """
        tokens = self.preprocess(text)

        positive_hits: List[str] = []
        negative_hits: List[str] = []
        score = 0

        for token in tokens:
            if token in self.positive_words:
                positive_hits.append(token)
                score += 1
            if token in self.negative_words:
                negative_hits.append(token)
                score -= 1

        return (
            f"Score = {score} "
            f"(positive: {positive_hits or '[]'}, "
            f"negative: {negative_hits or '[]'})"
        )


if __name__ == "__main__":
    analyzer = MoodAnalyzer()
    print(analyzer.preprocess("I absolutely love... getting stuck in traffic."))
