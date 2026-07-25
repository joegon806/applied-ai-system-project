# test_predict_reliable.py
# written by ChatGPT (I ran out of credits in Claude Code)

'''
Covers:
✓ Every sample post in SAMPLE_POSTS is processed.
✓ predict_reliable() always returns a ReliablePrediction.
✓ Confidence is always between 0 and 1.
✓ All three reliability signals (consistency, stability, strength) are present and valid.
✓ The abstained flag is consistent with the "uncertain" label.
✓ Predictions above the uncertainty threshold keep the original base_label.
✓ The function is deterministic across repeated calls.
✓ The formatted output is always a non-empty string.
'''

import pytest

from dataset import SAMPLE_POSTS
from mood_analyzer import MoodAnalyzer
from reliability import (
    predict_reliable,
    ReliablePrediction,
    UNCERTAIN_LABEL,
)


@pytest.fixture
def analyzer():
    return MoodAnalyzer()


def test_predict_reliable_returns_prediction_for_all_posts(analyzer):
    """Every sample post should produce a ReliablePrediction."""
    for post in SAMPLE_POSTS:
        prediction = predict_reliable(post, analyzer)

        assert isinstance(prediction, ReliablePrediction)


def test_prediction_fields_are_valid(analyzer):
    """Returned prediction should have valid fields."""
    valid_labels = {
        "positive",
        "negative",
        "neutral",
        "mixed",
        UNCERTAIN_LABEL,
    }

    base_labels = {
        "positive",
        "negative",
        "neutral",
        "mixed",
    }

    for post in SAMPLE_POSTS:
        prediction = predict_reliable(post, analyzer)

        assert prediction.label in valid_labels
        assert prediction.base_label in base_labels
        assert 0.0 <= prediction.confidence <= 1.0
        assert isinstance(prediction.signals, dict)


def test_reliability_signals_exist(analyzer):
    """Every prediction should include all reliability signals."""
    for post in SAMPLE_POSTS:
        prediction = predict_reliable(post, analyzer)

        assert set(prediction.signals.keys()) == {
            "consistency",
            "stability",
            "strength",
        }

        for value in prediction.signals.values():
            assert 0.0 <= value <= 1.0


def test_abstained_matches_label(analyzer):
    """The abstained flag should agree with the output label."""
    for post in SAMPLE_POSTS:
        prediction = predict_reliable(post, analyzer)

        assert prediction.abstained == (
            prediction.label == UNCERTAIN_LABEL
        )


def test_confident_predictions_keep_base_label(analyzer):
    """If confidence is above the threshold, the committed label should equal the base label."""
    for post in SAMPLE_POSTS:
        prediction = predict_reliable(post, analyzer)

        if not prediction.abstained:
            assert prediction.label == prediction.base_label


def test_predict_reliable_is_deterministic(analyzer):
    """Repeated calls on the same post should produce identical results."""
    for post in SAMPLE_POSTS:
        first = predict_reliable(post, analyzer)
        second = predict_reliable(post, analyzer)

        assert first.label == second.label
        assert first.base_label == second.base_label
        assert first.confidence == second.confidence
        assert first.signals == second.signals


def test_format_returns_string(analyzer):
    """format() should always return a human-readable string."""
    for post in SAMPLE_POSTS:
        prediction = predict_reliable(post, analyzer)

        formatted = prediction.format()

        assert isinstance(formatted, str)
        assert len(formatted) > 0