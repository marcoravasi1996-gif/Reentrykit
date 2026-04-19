"""Basic sanity tests."""

import reentrykit


def test_version_exists():
    """Package has a version string."""
    assert hasattr(reentrykit, "__version__")
    assert isinstance(reentrykit.__version__, str)


def test_version_format():
    """Version follows major.minor.patch format."""
    parts = reentrykit.__version__.split(".")
    assert len(parts) == 3
    assert all(p.isdigit() for p in parts)
