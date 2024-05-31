import pytest


@pytest.mark.unit
def test_import():
    try:
        import moosez
    except Exception as exc:
        print(exc.message)
