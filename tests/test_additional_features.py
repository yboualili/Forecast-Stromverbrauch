"""
Perform automated tests.

Tests calling the additional features script.
"""

import os
import unittest


class TestAdditionalFeatures(unittest.TestCase):
    """
    Perform automated tests.

    Args:
        unittest (_type_): testcase
    """

    def test_entrypoint(self) -> None:
        """
        Test if script can be called.

        Call with --help optional.
        """
        exit_status = os.system(
            "python src/additional_features.py\
            input_path output_path --help"
        )
        assert exit_status == 0

    def test_impute(self) -> None:
        """
        Test if script can be called with activated imputer.

        Call with --help optional.
        """
        exit_status = os.system(
            "python src/additional_features.py\
            input_path output_path --impute --help"
        )
        assert exit_status == 0

    def test_no_impute(self) -> None:
        """
        Test if script can be called with disabled imputer.

        Call with --help optional.
        """
        exit_status = os.system(
            "python src/additional_features.py\
             input_path output_path --no-impute --help"
        )
        assert exit_status == 0

    def test_remove_outlier(self) -> None:
        """
        Test if script can be called with activated outlier removal.

        Call with --help optional.
        """
        exit_status = os.system(
            "python src/additional_features.py\
            input_path output_path --compensate-outliers --help"
        )
        assert exit_status == 0

    def test_remove_no_outlier(self) -> None:
        """
        Test if script can be called with disabled outlier removal.

        Call with --help optional.
        """
        exit_status = os.system(
            "python src/additional_features.py\
             input_path output_path --no-compensate-outliers --help"
        )
        assert exit_status == 0
