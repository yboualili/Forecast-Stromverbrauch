"""
Perform automated tests.

Includes: tests for paths, reading hdf5 and pickeling.
"""

import os
import shutil
import unittest
from typing import Any
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd

from data.bda import BDA
from data.dataprocessor import DataProcessor


class TestDataProcessor(unittest.TestCase):
    """
    Perform automated tests.

    Args:
        unittest (_type_): testcase
    """

    def setUp(self) -> None:
        """Init variables.

        DataFrame has similar index and columns to original DataFrame.
        """
        self._test_path = "test_data"

        if not os.path.exists(self._test_path):
            os.makedirs(self._test_path)

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

        bda = BDA(self._test_path, self._test_path)
        self._data_preprocessor = DataProcessor(bda)
        self._data_preprocessor._df = self._df

    def test_paths(self) -> None:
        """
        Test, if paths are assigned correctly.

        Paths must equal the one specified in the BDA object.
        """
        assert (
            self._data_preprocessor._path_output == self._test_path
            and self._data_preprocessor._path_input == self._test_path
        )

    def test_to_pickle(self) -> None:
        """
        Test, if data frame can be written pickle file.

        File will be written to output_path/test.pkl
        """
        filename = "test.pkl"
        self._data_preprocessor._to_pickle(filename)

        assert os.path.isfile(
            os.path.join(self._data_preprocessor._path_input, filename)
        )

    def test_transform_non_exisiting_hdf(self) -> None:
        """
        Test, transformation of non-existing hdf5 file.

        Expects FileNotFoundError.
        """
        with self.assertRaises(
            FileNotFoundError, msg=r"'\w' not found. Download first."
        ):
            self._data_preprocessor.transform(pv=False)

    def test_read_hdf(self) -> None:
        """
        Test, if loading a wrong hdf5 file causes an error.

        Expects KeyError.
        """
        d1 = np.random.random(size=(100, 33))
        # initialize empty hdf5 file
        hf = h5py.File(os.path.join(self._test_path, "data.hdf5"), "w")
        hf.create_dataset("dataset_1", data=d1)

        hf.close()

        with self.assertRaises(
            KeyError, msg=r"Unable to open object (object 'NO_PV' doesn't exist)"
        ):
            self._data_preprocessor._read_hdf(
                os.path.join(self._test_path, "data.hdf5")
            )

    @patch("pandas.concat")
    @patch("data.dataprocessor.DataProcessor._to_pickle")
    @patch("data.dataprocessor.DataProcessor._read_hdf")
    @patch("os.path.exists")
    def test_transform(
        self, path_exists: Any, read_hdf: Any, to_pickle: Any, concat: Any
    ) -> None:
        """Test if two df are pickled."""
        path_exists.return_value = True

        testdata = pd.DataFrame(
            data=[[1, 2, 1], [1, 2, 1], [1, 2, 1]],
            index=[
                "2018-01-01 00:00:00+00:00",
                "2018-01-01 00:15:00+00:00",
                "2018-06-09 00:15:00+00:00",
            ],
            columns=["SFH10_S_1", "SFH10_S_2", "SFH10_S_3"],
        )
        testdata = testdata.set_index(pd.to_datetime(testdata.index))

        read_hdf.return_value = pd.DataFrame([])
        concat.return_value = testdata
        self._data_preprocessor.transform(False)
        assert to_pickle.call_count == 2

    def tearDown(self) -> None:
        """
        Clean up generated files.

        Will remove directory including all dummy files, such as raw hdf files
        and pre-processed pkl file.
        """
        shutil.rmtree(self._test_path)
