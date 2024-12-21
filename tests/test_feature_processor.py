"""
Perform automated tests.

Tests all major functions from the feature processor.
"""

import os
import shutil
import unittest
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal, assert_series_equal

from data.bda import BDA
from data.featureprocessor import FeatureProcessor


class TestFeatureProcessor(unittest.TestCase):
    """
    Perform automated tests.

    Args:
        unittest (_type_): testcase
    """

    def setUp(self) -> None:
        """Init variables.

        DataFrame has similar index and columns to original DataFrame.
        """
        self._df = pd.DataFrame(
            data=[[1, 2, np.NaN], [np.NaN, np.NaN, np.NaN], [3, 4, np.NaN]],
            index=[
                "2018-01-01 00:00:00+00:00",
                "2018-01-01 00:15:00+00:00",
                "2018-06-09 00:15:00+00:00",
            ],
            columns=["SFH10_S_1", "SFH10_S_2", "SFH10_S_3"],
        )
        self._df = self._df.set_index(pd.to_datetime(self._df.index))

        self._test_path = "test_data"
        self._test_path_input = "test_data/input/"
        self._test_path_output = "test_data/output/"

        if not os.path.exists(self._test_path_input):
            os.makedirs(self._test_path_input)
        if not os.path.exists(self._test_path_output):
            os.makedirs(self._test_path_output)

        bda = BDA(self._test_path_input, self._test_path_output)
        self._feature_preprocessor = FeatureProcessor(bda)
        self._feature_preprocessor._df = self._df

    def test_date_is_holiday(self) -> None:
        """
        Test, if holidays in lower saxony are correctly identified.

        E. g., New Year is a holiday. Thus 1 = True.
        """
        self.setUp()
        self._feature_preprocessor._make_time()
        output = self._feature_preprocessor._df
        expected_output = pd.Series(data=[1, 1, 0], index=self._df.index).astype(int)
        assert_series_equal(
            output["date_is_holiday"], expected_output, check_names=False
        )

    def test_date_is_weekday(self) -> None:
        """
        Test, if weekday is correctly identified.

        I. S., Monday to Fridays are weekdays.
        """
        self.setUp()
        self._feature_preprocessor._make_time()
        output = self._feature_preprocessor._df
        expected_output = pd.Series(data=[1, 1, 0], index=self._df.index).astype(int)
        assert_series_equal(
            output["date_is_weekday"], expected_output, check_names=False
        )

    def test_date_is_weekend(self) -> None:
        """
        Test, if weekend is correctly identified.

        I. S., Saturdays and Sundays is the weekend.
        """
        self.setUp()
        self._feature_preprocessor._make_time()
        output = self._feature_preprocessor._df
        expected_output = pd.Series(data=[0, 0, 1], index=self._df.index).astype(int)
        assert_series_equal(
            output["date_is_weekend"], expected_output, check_names=False
        )

    def test_date_not_weekend_and_weekday(self) -> None:
        """
        Test, if day is not weekend and weekday at the same time.

        E. g., a Monday should be detected as weekday, but not as weekend.
        """
        self.setUp()
        self._feature_preprocessor._make_time()
        output = self._feature_preprocessor._df
        assert (output["date_is_weekend"] != output["date_is_weekday"]).all()

    def test_date_season(self) -> None:
        """
        Test, if season is correctly identified.

        Season is mapped to [0, 1] range with 0 being winter, 1 being summer,
        as there are only two seasons in the dataframe.
        """
        self.setUp()
        self._feature_preprocessor._make_time()
        output = self._feature_preprocessor._df
        expected_output = pd.Series(data=[0.0, 0.0, 1.0], index=self._df.index)
        assert output["date_season"].equals(expected_output)

    def test_date_encoding_sin(self) -> None:
        """
        Test if date is correctly mapped to sin() on unit circle.

        See visualization: http://blog.davidkaleko.com/images/unit_circle.png
        """
        self.setUp()
        self._feature_preprocessor._make_time()
        output = self._feature_preprocessor._df
        expected_output = pd.Series(
            data=[0.49999999999999994, 0.49999999999999994, 1.2246467991473532e-16],
            index=self._df.index,
        )
        assert_series_equal(
            output["date_month_sin"], expected_output, check_names=False
        )

    def test_date_encoding_cos(self) -> None:
        """
        Test if date is correctly mapped to cos() on unit circle.

        See visualization: http://blog.davidkaleko.com/images/unit_circle.png
        """
        self.setUp()
        self._feature_preprocessor._make_time()
        output = self._feature_preprocessor._df
        expected_output = pd.Series(
            data=[0.8660254037844387, 0.8660254037844387, -1.0], index=self._df.index
        )
        assert_series_equal(
            output["date_month_cos"],
            expected_output,
            check_names=False,
            check_dtype=False,
        )

    def test_fill_na_remainder(self) -> None:
        """
        Test if dataframe is filled with zeros, if previous step is unsucessful.

        DataFame may not have any remaining pd.NA values.
        """
        self.setUp()
        self._df.rename(
            columns={
                "SFH10_S_1": "SFH3_S_1",
                "SFH10_S_2": "SFH9_S_1",
                "SFH10_S_3": "SFH23_S_1",
            },
            inplace=True,
        )
        self._feature_preprocessor._fill_na()
        output = self._feature_preprocessor._df
        expected_output = pd.DataFrame(
            data=[[1.0, 2.0, 1.20], [3.0, 4.0, 2.80]],
            index=[
                "2018-01-01 00:00:00+00:00",
                "2018-06-09 00:15:00+00:00",
            ],
            columns=["SFH3_S_1", "SFH9_S_1", "SFH23_S_1"],
        )
        expected_output = expected_output.set_index(
            pd.to_datetime(expected_output.index)
        )

        assert_frame_equal(output, expected_output, rtol=0.01)

    def test_make_spq_tot(self) -> None:
        """
        Test generation of spq tot features in no pv dataset.

        Expects non-empty dataframe.
        """
        self.setUp()
        self._df = pd.DataFrame(
            data=[[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]],
            index=[
                "2018-01-01 00:00:00+00:00",
                "2018-01-01 00:15:00+00:00",
                "2018-06-09 00:15:00+00:00",
            ],
            columns=[
                "SFH10_P_TOT",
                "SFH10_S_TOT",
                "SFH10_Q_TOT",
                "SFH12_P_TOT",
                "SFH12_S_TOT",
                "SFH12_Q_TOT",
            ],
        )
        self._df = self._df.set_index(pd.to_datetime(self._df.index))
        self._feature_preprocessor._df = self._df
        pv = False
        self._feature_preprocessor._make_spq_total(pv)
        output = self._feature_preprocessor._df
        exp_output = self._df
        exp_output["S_TOT"] = np.repeat(2, 3)
        exp_output["P_TOT"] = np.repeat(2, 3)
        exp_output["Q_TOT"] = np.repeat(2, 3)

        assert_frame_equal(output, exp_output)

    def test_make_spq_tot_solar(self) -> None:
        """
        Test generation of spq tot features in only PV dataset.

        Expects non-empty dataframe.
        """
        self.setUp()
        self._df = pd.DataFrame(
            data=[
                [1, 2, 1, 1, 1, 2, 1, 1],
                [1, 2, 1, 1, 1, 2, 1, 1],
                [1, 2, 1, 1, 1, 2, 1, 1],
            ],
            index=[
                "2018-01-01 00:00:00+00:00",
                "2018-01-01 00:15:00+00:00",
                "2018-06-09 00:15:00+00:00",
            ],
            columns=[
                "SFH10_P_TOT",
                "SFH10_P_TOT_WITH_PV",
                "SFH10_S_TOT",
                "SFH10_Q_TOT",
                "SFH12_P_TOT",
                "SFH12_P_TOT_WITH_PV",
                "SFH12_S_TOT",
                "SFH12_Q_TOT",
            ],
        )
        self._df = self._df.set_index(pd.to_datetime(self._df.index))
        self._feature_preprocessor._df = self._df
        pv = True
        self._feature_preprocessor._make_spq_total(pv)
        output = self._feature_preprocessor._df
        exp_output = self._df
        exp_output["S_TOT"] = np.repeat(2, 3)
        exp_output["P_TOT"] = np.repeat(4, 3)
        exp_output["Q_TOT"] = np.repeat(2, 3)

        assert_frame_equal(output, exp_output)

    def test_to_pickle(self) -> None:
        """Test if to_pickle creates a new pkl file."""
        self.setUp
        self._feature_preprocessor._to_pickle(filename="test.pkl")
        assert os.path.isfile("./test_data/output/test.pkl")

    def test_read_non_existing_pkl(self) -> None:
        """
        Test, transformation of non-existing pickle file.

        Expects FileNotFoundError.
        """
        with self.assertRaises(
            FileNotFoundError, msg=r"'\w' not found. Generate first."
        ):
            self._feature_preprocessor._read_pickle("test.pkl")

    def test_read_existing_pkl(self) -> None:
        """
        Test, transformation of existing pickle file.

        Expects non-empty dataframe.
        """
        self.setUp()

        test_file = "test.pkl"
        self._df.to_pickle(os.path.join(self._test_path_input, test_file))
        self._feature_preprocessor._read_pickle(test_file)
        assert not self._df.empty

    def test_remove_na(self) -> None:
        """Test if NA values are correctly removed."""
        self._df = pd.DataFrame(
            data=[
                [None, 2, 1, 1, 1, 2, 1, 1],
                [None, 2, 1, 1, 1, 2, 1, 1],
                [1, 2, 1, 1, 1, 2, 1, 1],
            ],
            index=[
                "2018-01-01 00:00:00+00:00",
                "2018-01-01 00:15:00+00:00",
                "2018-06-09 00:15:00+00:00",
            ],
            columns=[
                "SFH10_P_TOT",
                "SFH10_P_TOT_WITH_PV",
                "SFH10_S_TOT",
                "SFH10_Q_TOT",
                "SFH12_P_TOT",
                "SFH12_P_TOT_WITH_PV",
                "SFH12_S_TOT",
                "SFH12_Q_TOT",
            ],
        )
        self._feature_preprocessor._df = self._df.set_index(
            pd.to_datetime(self._df.index)
        )
        self._feature_preprocessor._remove_na(threshold=0.6)
        assert self._feature_preprocessor._df.shape == (3, 7)

    @patch("data.featureprocessor.FeatureProcessor._make_time")
    @patch("data.featureprocessor.FeatureProcessor._fill_na")
    @patch("data.featureprocessor.FeatureProcessor._remove_na")
    @patch("data.featureprocessor.FeatureProcessor._read_pickle")
    @patch("data.featureprocessor.FeatureProcessor._make_spq_total")
    def test_transform(
        self,
        make_spq_total: Any,
        read_pickle: Any,
        remove_na: Any,
        fill_na: Any,
        make_time: Any,
    ) -> None:
        """Test if transform calls pipeline function correctly."""
        self._feature_preprocessor.transform()
        assert read_pickle.call_count == 2
        assert make_spq_total.call_count == 2
        assert remove_na.call_count == 2
        assert fill_na.call_count == 2
        assert make_time.call_count == 2

    def tearDown(self) -> None:
        """
        Clean up generated files.

        Will remove directory including all dummy files,
        """
        shutil.rmtree(self._test_path)
