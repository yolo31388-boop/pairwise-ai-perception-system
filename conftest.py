"""
The shipped test classes use ``self.assertIsNotNone`` without inheriting
from ``unittest.TestCase``. Test files must not be modified, so provide
the missing unittest-style assertion methods on collected plain test
classes here.
"""


def pytest_collection_finish(session):
    for item in session.items:
        cls = getattr(item, "cls", None)
        if cls is None or hasattr(cls, "assertIsNotNone"):
            continue

        def _assertIsNotNone(self_, value, msg=None):
            if value is None:
                raise AssertionError(msg or f"{value!r} is None")

        cls.assertIsNotNone = _assertIsNotNone
