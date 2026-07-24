"""
Shared data for the Mood Machine lab.

This file defines:
  - POSITIVE_WORDS: starter list of positive words
  - NEGATIVE_WORDS: starter list of negative words
  - SAMPLE_POSTS: short example posts for evaluation and training
  - TRUE_LABELS: human labels for each post in SAMPLE_POSTS
"""

# ---------------------------------------------------------------------
# Starter word lists
# ---------------------------------------------------------------------

POSITIVE_WORDS = [
    "happy",
    "great",
    "good",
    "love",
    "excited",
    "awesome",
    "fun",
    "chill",
    "relaxed",
    "amazing",
    "proud",
]

NEGATIVE_WORDS = [
    "sad",
    "bad",
    "terrible",
    "awful",
    "angry",
    "upset",
    "tired",
    "stressed",
    "hate",
    "boring",
    "exhausted",
]

# Words that flip the polarity of the sentiment word that follows them.
# Note: preprocess() strips apostrophes, so "don't" -> "dont", "isn't" -> "isnt".
NEGATION_WORDS = [
    "not",
    "no",
    "never",
    "cant",
    "cannot",
    "dont",
    "isnt",
    "wasnt",
    "wont",
]

# ---------------------------------------------------------------------
# Starter labeled dataset
# ---------------------------------------------------------------------

# Short example posts written as if they were social media updates or messages.
SAMPLE_POSTS = [
    "I love this class so much",
    "Today was a terrible day",
    "Feeling tired but kind of hopeful",
    "This is fine",
    "So excited for the weekend",
    "I am not happy about this",
    "im dead 💀", #colloquially, means laughing really hard, or shocked
    "I absolutely love getting stuck in traffic",
    "I feel very strongly about this.",
    "What do you mean",
    "It is what it is",
    "I told you so",
    "Oh great, another Monday"
]

# Human labels for each post above.
# Allowed labels in the starter:
#   - "positive"
#   - "negative"
#   - "neutral"
#   - "mixed"
TRUE_LABELS = [  # the "right answers"
    "positive",  # "I love this class so much"
    "negative",  # "Today was a terrible day"
    "mixed",     # "Feeling tired but kind of hopeful"
    "neutral",   # "This is fine"
    "positive",  # "So excited for the weekend"
    "negative",  # "I am not happy about this"
    
    "slang",     # "im dead 💀"
    "sarcastic", # "I absolutely love getting stuck in traffic"
    "ambiguous",  # "I feel very strongly about this."
    "questioning", # "What do you mean"
    "ambiguous",  # "It is what it is"
    "passive_aggressive",  # "I told you so"
    "sarcastic"  # "Oh great, another Monday"
]

# ---------------------------------------------------------------------
# Label mapping (resolving the label-space mismatch)
# ---------------------------------------------------------------------
#
# TRUE_LABELS above uses a rich human vocabulary (slang, sarcastic, etc.)
# that captures HOW a post is worded. But MoodAnalyzer can only ever
# output four labels: positive, negative, neutral, mixed. Comparing the
# rich labels directly against the model is unfair: those posts can never
# match, so accuracy is capped for reasons unrelated to model quality.
#
# LABEL_MAP collapses every human label onto the model's four-label space
# so that rule-based evaluation is measured fairly. The four model labels
# map to themselves; the richer labels are judgment calls you can revise.
#
# EVAL_LABELS is the normalized list to score the RULE-BASED model against.
# (The ML model in ml_experiments.py can instead learn TRUE_LABELS directly,
# since it is not limited to the four-label space.)
LABEL_MAP = {
    # The model's own labels map to themselves.
    "positive": "positive",
    "negative": "negative",
    "neutral": "neutral",
    "mixed": "mixed",
    # Richer human labels collapsed onto the four-label space.
    # These are interpretations — adjust them if you disagree.
    "slang": "positive",             # "im dead 💀" = laughing hard
    "sarcastic": "negative",         # positive words, negative meaning
    "passive_aggressive": "negative",
    "ambiguous": "neutral",          # no clear polarity
    "questioning": "neutral",        # a question, not a sentiment
}

# Normalized labels for fair rule-based evaluation. Uses .get(label, label)
# so any label missing from LABEL_MAP passes through unchanged and is caught
# (rather than crashing) by the reliability report's label-space check.
EVAL_LABELS = [LABEL_MAP.get(label, label) for label in TRUE_LABELS]

# TODO: Add 5-10 more posts and labels.
#
# Requirements:
#   - For every new post you add to SAMPLE_POSTS, you must add one
#     matching label to TRUE_LABELS.
#   - SAMPLE_POSTS and TRUE_LABELS must always have the same length.
#   - Include a variety of language styles, such as:
#       * Slang ("lowkey", "highkey", "no cap")
#       * Emojis (":)", ":(", "🥲", "😂", "💀")
#       * Sarcasm ("I absolutely love getting stuck in traffic")
#       * Ambiguous or mixed feelings
#
# Tips:
#   - Try to create some examples that are hard to label even for you.
#   - Make a note of any examples that you and a friend might disagree on.
#     Those "edge cases" are interesting to inspect for both the rule based
#     and ML models.
#
# Example of how you might extend the lists:
#
# SAMPLE_POSTS.append("Lowkey stressed but kind of proud of myself")
# TRUE_LABELS.append("mixed")
#
# Remember to keep them aligned:
#   len(SAMPLE_POSTS) == len(TRUE_LABELS)
