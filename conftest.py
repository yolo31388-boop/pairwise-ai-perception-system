"""Provide unittest-style assertion helpers to plain Test* classes.

The test classes in this project use ``self.assertIsNotNone`` etc. without
inheriting from ``unittest.TestCase``. Inject the standard assertion methods
at collection time so those tests can run unmodified.
"""

import unittest


def pytest_pycollect_makeitem(collector, name, obj):
    if isinstance(obj, type) and name.startswith("Test") and not issubclass(obj, unittest.TestCase):
        for attr in dir(unittest.TestCase):
            if attr.startswith("assert") and not hasattr(obj, attr):
                setattr(obj, attr, getattr(unittest.TestCase, attr))
