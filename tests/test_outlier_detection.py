"""
Perform automated tests.

Outlier removal is part of the feature engineering.
"""

import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from data.bda import BDA
from data.featureprocessor import FeatureProcessor


class TestOutlierDetection(unittest.TestCase):
    """
    Perform automated tests regarding the outlier detection and compensation.

    Args:
        unittest (_type_): testcase
    """

    def setUp(self) -> None:
        """Init variables.

        DataFrame has similar index and columns to the original DataFrame.
        """
        # determine some indices for anomalies
        self.indexes_SFH10_S_1 = [(6, 0), (18, 0)]
        self.indexes_SFH10_S_2 = [(1, 1), (22, 1)]
        self.indexes_SFH10_S_3 = [(12, 2)]

        # generate 3 years of data 8760 * 3 = 26280
        data = np.zeros((26280, 3))

        self.index = pd.date_range("2018-01-01", "2020-12-31", freq="h")[:-1]

        self._df = pd.DataFrame(
            data=data,
            index=self.index,
            columns=["SFH10_S_1", "SFH10_S_2", "SFH10_S_3"],
        )

        bda = BDA("foo.pkl", "bar.pkl")
        self._feature_preprocessor = FeatureProcessor(bda)
        self._feature_preprocessor._df = self._df

    def test_remove_outliers_with_outliers(self) -> None:
        """Test, if outliers are compensated correctly."""
        self.setUp()

        # insert outliers only on the first day
        for idx in self.indexes_SFH10_S_1:
            self._df.iloc[idx] = 9
        for idx in self.indexes_SFH10_S_2:
            self._df.iloc[idx] = 24
        for idx in self.indexes_SFH10_S_3:
            self._df.iloc[idx] = 60

        self._feature_preprocessor._remove_outlier()
        output = self._feature_preprocessor._df

        expected_output_data = np.zeros((26280, 3))

        # outliers get compensated with the load profile (average of all points
        # in time over all data available), here we have 3 years all zero, so the
        # outlier is compensated with 1/3 of its value
        for idx in self.indexes_SFH10_S_1:
            expected_output_data[idx] = 3
        for idx in self.indexes_SFH10_S_2:
            expected_output_data[idx] = 8
        for idx in self.indexes_SFH10_S_3:
            expected_output_data[idx] = 20

        expected_output = pd.DataFrame(
            data=expected_output_data,
            index=self.index,
            columns=["SFH10_S_1", "SFH10_S_2", "SFH10_S_3"],
        )

        assert_frame_equal(output, expected_output)

    def test_remove_outliers_under_threshold(self) -> None:
        """Test, if outliers lower than +- std * threshold are compensated correctly."""
        self.setUp()
        # insert outliers only in the first day for SFH10_S_1 the outliers should stay
        # below the limit for detection (which is 4.9)
        self._df.iloc[::2, 0] += 1

        for idx in self.indexes_SFH10_S_1:
            self._df.iloc[idx] = 4
        for idx in self.indexes_SFH10_S_2:
            self._df.iloc[idx] = 24
        for idx in self.indexes_SFH10_S_3:
            self._df.iloc[idx] = 60

        self._feature_preprocessor._remove_outlier()
        output = self._feature_preprocessor._df

        expected_output_data = np.zeros((26280, 3))

        # outliers get compensated with the load profile (average of all points
        # in time over all data available), here we have 3 years all zero, so the
        # outlier is compensated with 1/3 of its value except for SFH10_S_1
        expected_output_data[::2, 0] += 1
        for idx in self.indexes_SFH10_S_1:
            expected_output_data[idx] = 4
        for idx in self.indexes_SFH10_S_2:
            expected_output_data[idx] = 8
        for idx in self.indexes_SFH10_S_3:
            expected_output_data[idx] = 20

        expected_output = pd.DataFrame(
            data=expected_output_data,
            index=self.index,
            columns=["SFH10_S_1", "SFH10_S_2", "SFH10_S_3"],
        )

        assert_frame_equal(output, expected_output)

    def test_remove_outliers_no_outliers(self) -> None:
        """Test, if data is unchanged if no outliers are present."""
        self.setUp()
        # do not insert any outliers
        self._feature_preprocessor._remove_outlier()
        output = self._feature_preprocessor._df
        expected_output_data = np.zeros((26280, 3))
        expected_output = pd.DataFrame(
            data=expected_output_data,
            index=self.index,
            columns=["SFH10_S_1", "SFH10_S_2", "SFH10_S_3"],
        )

        assert_frame_equal(output, expected_output)
