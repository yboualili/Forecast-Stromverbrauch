"""
Perform automated tests.

Corresponds to src/data/bda.py
"""

import os
import shutil
import unittest

from data.bda import BDA


class TestSetup(unittest.TestCase):
    """
    Perform automated tests.

    Args:
        unittest (_type_): testcase
    """

    def setUp(self) -> None:
        """Init variables.

        Set test data paths.
        """
        self._test_path = "test_data"
        self._test_input_path = os.path.join(self._test_path, "test_input_data")
        self._test_output_path = os.path.join(self._test_path, "test_output_data")

    def test_mk_dir_not_existing(self) -> None:
        """
        Test if directories are created.

        Assure that none of the dirs are existing.
        """
        if os.path.exists(self._test_input_path):
            os.removedirs(self._test_input_path)

        if os.path.exists(self._test_output_path):
            os.removedirs(self._test_output_path)

        self._bda = BDA(self._test_input_path, self._test_output_path)

        assert os.path.exists(self._test_input_path) and os.path.exists(
            self._test_output_path
        )

    def tearDown(self) -> None:
        """
        Clean up generated files.

        Will remove directory including all dummy directories
        """
        shutil.rmtree(self._test_path)
