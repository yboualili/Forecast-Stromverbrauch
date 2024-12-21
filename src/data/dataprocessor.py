"""
Hold information related to the raw / pre-processed dataframes.

This includes:
1. load data or
2. heatpump data and
3. context
"""

import logging
from os import path
from typing import Any, List, Union

import h5py
import pandas as pd

from data import bda, const


class DataProcessor:
    """
    OO representation of raw / pre-processed data.

    May include load or heatpump data.
    """

    def __init__(self, bda: bda.BDA) -> None:
        """
        Construct a datapreprocessor object.

        May include the data itself, but also information on the context.

        Args:
            bda (data.bda.BDA): bda object holding information about file paths.
        """
        self._df = pd.DataFrame()
        self._bda = bda
        self._path_output = bda.output_path
        self._path_input = bda.input_path

    def transform(self, pv: bool) -> None:
        """
        Turn raw households file into DataFrames.

        One DataFrame contains heatpump data, the other load data.

        Args:
            pv (bool, optional): determines whether only households with/without
            PV system are exported.

        Raises:
            FileNotFoundError: Input file can not be found
        """
        logger = logging.getLogger(__name__)
        # 1. create load data set
        logger.info("processing load data... 💡")
        dfs = []

        for filename in const.FILENAME_HOUSEHOLDS_HDF5:
            if not path.exists(path.join(self._path_input, filename)):
                raise FileNotFoundError(f"'{filename}' not found. Download first.")
            else:
                if pv:
                    df = self._read_hdf(
                        path.join(self._path_input, filename),
                        levels=["WITH_PV", None, "HOUSEHOLD"],
                    )
                    dfs.append(df)
                else:
                    df = self._read_hdf(
                        path.join(self._path_input, filename),
                        levels=["NO_PV", None, "HOUSEHOLD"],
                    )
                    dfs.append(df)

        # concat and resample to 15 min freq
        self._df = pd.concat(dfs, axis=0).asfreq("15min")

        if pv:
            self._to_pickle(const.FILENAME_LOAD_SOLAR_PKL)
        else:
            self._to_pickle(const.FILENAME_LOAD_PKL)

        # 2. create heatpump data set
        logger.info("processing heatpump data... 🥵")
        dfs = []
        for filename in const.FILENAME_HOUSEHOLDS_HDF5:
            if pv:
                df = self._read_hdf(
                    path.join(self._path_input, filename),
                    levels=["WITH_PV", None, "HEATPUMP"],
                )
                dfs.append(df)
            else:
                df = self._read_hdf(
                    path.join(self._path_input, filename),
                    levels=["NO_PV", None, "HEATPUMP"],
                )
                dfs.append(df)

        # concat and resample to 15 min freq
        self._df = pd.concat(dfs, axis=0).asfreq("15min")
        if pv:
            self._to_pickle(const.FILENAME_HEATPUMP_SOLAR_PKL)
        else:
            self._to_pickle(const.FILENAME_HEATPUMP_PKL)

    def _to_pickle(self, filename: str) -> None:
        """
        Write DataFrame to Pickle File.

        Args:
            filename (str): filename.
        """
        logger = logging.getLogger(__name__)
        logger.info(f"writing '{path.join(self._path_output, filename)}'.")

        self._df.to_pickle(path.join(self._path_output, filename))

    def _read_hdf(
        self,
        path: str = "data/raw/2019_data_15min.hdf5",
        levels: List[Union[str, Union[List[Any], Any]]] = [
            "NO_PV",
            None,
            "HOUSEHOLD",
        ],
    ) -> pd.DataFrame:
        """Convert hdf5 file to dataframe and perform preliminary processing.

        Args:
            path (str, optional): path to file.
            Defaults to "data/raw/2019_data_15min.hdf5".
            levels (List[Union[str, Union[List[str],str]]], optional): levels in tree.
            Defaults to [ "NO_PV", None, "HOUSEHOLD", ].

        Returns:
            pd.DataFrame: DataFrame in row/ column format.
        """
        logger = logging.getLogger(__name__)
        logger.info(f"loading '{path}'.")

        households = []
        df_all = []

        f = h5py.File(path, "r")

        if levels[1] is None:
            households = list(f[levels[0]].keys())
        elif isinstance(levels[1], list):
            households = levels[1]
        elif isinstance(levels[1], str):
            households = list(levels[1])

        for household in households:
            df_household = pd.DataFrame(f[levels[0]][household][levels[2]]["table"][:])
            df_household["index"] = pd.to_datetime(
                df_household["index"], unit="s", utc=True
            )
            df_household.set_index("index", inplace=True)
            df_household = df_household.add_prefix(f"{household}_")
            df_household.index.names = ["date"]
            df_all.append(df_household)
        return pd.concat(df_all, axis=1)
