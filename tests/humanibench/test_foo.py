"""Test for foo module."""

from humanibench.foo import foo as foofn


def test_foofn(my_test_number: int) -> None:
    """Test foofn function."""
    assert foofn(f"bar-{my_test_number}") == "foobar-42"
