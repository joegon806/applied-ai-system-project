# The Mood Machine

The Mood Machine is a text classifier that classifies the mood of a text as positive, negative, neutral, or mixed. The classification can be determined based on either a determinisitic rule-based model or a machine learning model.

The final project extends The Mood Machine's Rule-based model to have checks for reliabilty and confidence scores based on those checks. A test suite for the implemented functionality is also added.

# Architecture Overview

In the Rule-based Model, the model first runs evaluate_rule_based, which takes the SAMPLE_POSTS from dataset.py and applies MoodAnalyzer to them to predict their moods, and then applies reliability.py to them to generate their confidence scores. This function also compares the sample posts' given labels and compares them to the predicted labels to generate an accuracy report. Then the model lets the user enter their own phrases, and applies MoodAnalyzer to predict their moods and applies reliability.py to generate their confidence scores.

In the Machine Learning-based Model, the LogisticRegression model is trained on the vectorized SAMPLE_POSTS and their TRUE_LABELS, and uses this data to predict the label of the user's entered posts.

The test suite performs automated tests on the functionality of reliability.py.

# Setup Instructions

To run the Rule-based model, run "python main.py" in the terminal. When prompted, type the post you want classified and press enter.
To run the Machine Learning-based model, run "python ml_experiments.py" in the terminal. When prompted, type the post you want classified and press enter.
To run the automated reliability tests, run "pytest test_predict_reliable.py".

# Sample Interactions

Rule-based model:

```
You: This is a good day
Model: positive (54% confidence)
You: I hate fast food
Model: negative (52% confidence)
You: I love and hate you so much
Model: uncertain (35% confidence, leaning mixed)
  (low confidence — the signals for this text are weak or conflicting, so the model will not commit to a mood.)
```

ML-based model:
```
You: This is a good day
ML model: negative
You: I hate fast food
ML model: negative
You: I love and hate you so much
ML model: positive
```

# Design Decisions

The function predict_reliable() in reliability.py computes a confidence from three reliability signals: 
- consistency (case / whitespace / trailing-punctuation don't flip the label),
- stability (adding a positive word doesn't lower the mood score, and adding a negative word doesn't increase it),
- and strength (how much positive and negative words cancel each other out),
and if the confidence lands below a set threshold of uncertainty, the function abstains from giving the label confidently, labeling the mood as "uncertain". Confidence is calculated as (0.45 * stability + 0.55 * strength) * consistency: stablity and strength are added together because they form the base for confidence, but are not too significant, and both are about equal in weight. Consistency, however, is multiplied into the formula, giving it significant weight. This is because consistency, which involves changing insignificant parts of the text, should not have any effect on the label. If changing something insignificant does affect the label, then it's the result a significant reliability failure, and the model should not be confident in that label.

One trade-off I made when developing this project's expansion is that I neglected to add many more labels, sample posts, positive words, or negative words to the dataset. I made this choice because I believed adding words would be a sisyphean task, in which I wouldn't know how much data I would need to add before I could be confident enough in my model, especially considering edge cases like slang or word conjugation. So instead, I opted to create a confidence generator for the model that I already have.

# Testing Summary

What worked with this model is that obviously positive or negative posts (e.g., no words of the opposite mood) get labeled correctly. One thing that didn't work with this model is that I found that mixed posts (e.g. posts with both positive and negative posts) tended to be flagged as uncertain with low confidence. I also found that the model still produces wrong answers if their computed uncertainty is lower than the uncertainty threshold, which taught me that such a mood detection algorithm is not a cut-and-dry technique but instead requires careful human evaluation and adjustment, such as adjusting the threshold of uncertainty in this model.

# Reflection: A brief note on what this project taught you about AI and problem-solving. Your graded responsible-AI reflection — how you collaborated with AI, one helpful and one flawed AI suggestion, and your system's limitations — goes in model_card.md (see Step 5), not here. Reflection content placed only in the README does not earn the reflection points.

This project taught me that using and creating an AI must involve a healthy amount of human collaboration, including supervising the AI's suggestions and adjusting details as needed.