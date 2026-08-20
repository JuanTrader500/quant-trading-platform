import pytest

from features.ohlc_validation import InvalidOHLCError, validate_ohlc


def test_valid_ohlc_does_not_raise():
    validate_ohlc(open_=100, high=105, low=98, close=102)


def test_low_greater_than_high_is_rejected():
    with pytest.raises(InvalidOHLCError):
        validate_ohlc(open_=100, high=95, low=98, close=96)


def test_open_outside_low_high_is_rejected():
    with pytest.raises(InvalidOHLCError):
        validate_ohlc(open_=110, high=105, low=98, close=100)


def test_close_outside_low_high_is_rejected():
    with pytest.raises(InvalidOHLCError):
        validate_ohlc(open_=100, high=105, low=98, close=120)


def test_non_positive_values_are_rejected():
    with pytest.raises(InvalidOHLCError):
        validate_ohlc(open_=0, high=105, low=98, close=100)
