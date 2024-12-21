"""
Perform automated tests.

Corresponds to src/preprocessing.py
"""

import os
import unittest


class TestPreprocessing(unittest.TestCase):
    """
    Perform automated tests.

    Args:
        unittest (_type_): testcase
    """

    def test_entrypoint(self) -> None:
        """
        Test if script can be called.

        Call with --help optional to prevent execution.
        """
        exit_status = os.system(
            "python src/preprocessing.py input_path output_path --help"
        )
        assert exit_status == 0
