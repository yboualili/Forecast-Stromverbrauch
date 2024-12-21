"""
Hold information related to pre-processed / final dataframe.

This includes:
1. load data or
2. heatpump data and
3. context
"""
import logging
import re
from os import path

import holidays
import numpy as np
import pandas as pd
from sklearn.preprocessing import minmax_scale

import data.bda
from data import const


class FeatureProcessor:
    """
    OO representation of pre-processed / final data.

    May include load or heatpump data.
    """

    def __init__(self, bda: data.bda.BDA) -> None:
        """
        Construct a featureprocessor object.

        May include the data itself, but also information on the context.

        Args:
            bda (data.bda.BDA): bda object holding information about file paths.
        """
        self._df = pd.DataFrame()
        self._bda = bda
        self._path_output = bda.output_path
        self._path_input = bda.input_path

    def transform(
        self, impute: bool = True, compensate_outliers: bool = False, pv: bool = False
    ) -> None:
        """
        Enrich preliminary DataFrame with more Features to become the final DataFrame.

        Caution: Order matters as make_target() depends on the outlier removal.
        Args:
            impute (bool, optional): Impute missing data. Defaults to True.
            compensate_outliers (bool, optional): Compensate outliers with statistical
                                                  approach
        Raises:
            FileNotFoundError: Input file can not be found
        """
        logger = logging.getLogger(__name__)
        # 1. process load data set
        logger.info("adding features to load data... 💡")

        if pv:
            self._read_pickle(const.FILENAME_LOAD_SOLAR_PKL)
        else:
            self._read_pickle(const.FILENAME_LOAD_PKL)

        if impute:
            self._remove_na()
            self._fill_na()

        if compensate_outliers:
            self._remove_outlier()

        self._make_spq_total(pv)
        self._make_time()

        if pv:
            self._to_pickle(const.FILENAME_LOAD_SOLAR_PKL)
        else:
            self._to_pickle(const.FILENAME_LOAD_PKL)

        # 2. process heatpump data set
        logger.info("adding features to heatpump data... 🥵")
        if pv:
            self._read_pickle(const.FILENAME_HEATPUMP_SOLAR_PKL)
        else:
            self._read_pickle(const.FILENAME_HEATPUMP_PKL)

        if impute:
            self._remove_na()
            self._fill_na()

        if compensate_outliers:
            self._remove_outlier()

        self._make_spq_total(pv)
        self._make_time()

        if pv:
            self._to_pickle(const.FILENAME_HEATPUMP_SOLAR_PKL)
        else:
            self._to_pickle(const.FILENAME_HEATPUMP_PKL)

    def _read_pickle(self, filename: str) -> None:
        """
        Read DataFrame from Pickle File.

        Args:
            filename (str): filename.
        """
        logger = logging.getLogger(__name__)
        logger.info(f"reading '{path.join(self._path_input, filename)}'.")

        if not path.exists(path.join(self._path_input, filename)):
            raise FileNotFoundError(f"'{filename}' not found. Generate first.")
        else:
            self._df = pd.read_pickle(path.join(self._path_input, filename))

    def _to_pickle(self, filename: str) -> None:
        """
        Write DataFrame to Pickle File.

        Args:
            filename (str): filename to write to.

        """
        logger = logging.getLogger(__name__)
        logger.info(f"writing '{path.join(self._path_output, filename)}'.")

        self._df.to_pickle(path.join(self._path_output, filename))

    def _fill_na(self) -> None:
        """
        Fill missing values using greedy approach.

        Algorithm:
        1. For each missing value, calculate the average load per household
        2. Scale the load to the building area in square meters
        2. Fill remaining values with zero.

        """
        features = self._df.columns.to_list()

        feature_profiles = pd.DataFrame(index=self._df.index)
        for feature in features:
            # since all features start with 'SFH'+one/two digits we use str.find('_')
            feature_group = feature[feature.find("_") :]
            if feature_group not in feature_profiles.columns:
                same_feature_households = list(
                    filter(lambda x: x.endswith(feature_group), features)
                )
                feature_profiles[feature_group] = self._df[
                    same_feature_households
                ].mean(axis=1)

        # For area per household see:
        # @article{schlemmingerDatasetElectricalSinglefamily2022,
        # title = {Dataset on Electrical Single-Family House and Heat
        # Pump Load Profiles in {{Germany}}},
        # author = {Schlemminger, Marlon and Ohrdes, Tobias and Schneider,
        # Elisabeth and Knoop, Michael},
        # date = {2022-12},
        # journaltitle = {Scientific Data},
        # shortjournal = {Sci Data},
        # volume = {9},
        # number = {1},
        # pages = {56},
        # doi = {10.1038/s41597-022-01156-1},
        # }
        # For household 30 the area is unknown so we use the mean (141.48648648648648)
        area_per_household = pd.Series(
            {
                "3": 140,
                "4": 160,
                "5": 160,
                "6": 140,
                "7": 150,
                "8": 160,
                "9": 195,
                "10": 135,
                "11": 230,
                "12": 112,
                "13": 130,
                "14": 150,
                "15": 120,
                "16": 136,
                "17": 200,
                "18": 87,
                "19": 203,
                "20": 220,
                "21": 110,
                "22": 117,
                "23": 113,
                "24": 120,
                "25": 100,
                "26": 120,
                "27": 110,
                "28": 145,
                "29": 104,
                "30": 141,
                "31": 135,
                "32": 160,
                "33": 111,
                "34": 110,
                "35": 100,
                "36": 108,
                "37": 199,
                "38": 190,
                "39": 135,
                "40": 120,
            }
        )

        self._df["search_key"] = self._df.index

        for feature in features:
            household_number = re.findall("[0-9]+", feature.lstrip("SFH"))[0]
            feature_group = feature[feature.find("_") :]
            scale_factor = (
                area_per_household[household_number] / area_per_household.mean()
            )
            self._df[feature] = self._df[feature].fillna(
                self._df.search_key.map(feature_profiles[feature_group]) * scale_factor
            )

        self._df.drop(columns=["search_key"], inplace=True)
        # First row of load data available after dropping the missing values is
        # 2018-05-02 14:30:00+00:00 and results in 93542 measurements
        # First row of heatpump data available after dropping the missing values is
        # 2018-05-06 08:30:00+00:00 and results in 93182 measurements
        self._df.dropna(inplace=True)

        logger = logging.getLogger(__name__)
        logger.info("finished replacing pd.NA.")

    def _remove_na(self, threshold: float = 0.8) -> None:
        """Remove columns where more than X % of values are missing.

        Example:
        input = pd.DataFrame([[np.NaN, 1],[np.NaN, 2]])
        ...
        output = pd.DataFrame([[1], [2]])

        Args:
            threshold (float, optional): threshold. Defaults to 0.8.
        """
        if threshold <= 0 or threshold >= 1:
            raise ValueError("threshold must be between (0,1).")

        self._df = self._df[self._df.columns[self._df.isnull().mean() < threshold]]
        logger = logging.getLogger(__name__)
        logger.info(
            f"finished removing columns with >= {threshold * 100} % pd.NA values."
        )

    def _remove_outlier(self) -> None:
        """
        Remove outliers based on mean and standard deviation of each time series.

        For this option, a threshold, that regulates the maximal distance a point
        can have to the mean, can be chosen. The outliers are than removed and
        substituted with the respective value in the load profile.

        Example:
        input = pd.DataFrame([[0,0,0,0,0,10,0,0,0,0,0],[5,5,5,5,5,5,5,5,5,5,5]])
        load_profile = pd.DataFrame([[0,0,0,0,0,4,0,0,0,0,0],[5,5,5,5,5,5,5,5,5,5,5]])
        ...
        output = pd.DataFrame([[0,0,0,0,0,4,0,0,0,0,0],[5,5,5,5,5,5,5,5,5,5,5]]
        """
        threshold = 10

        # create search key e. g., (1,datetime.time(0,0)) (day, timestep)
        day_of_year = self._df.index.dayofyear
        time = self._df.index.time
        self._df["search_key"] = list(zip(day_of_year, time))

        # generate load profile for time slices
        load_profiles_from_data = self._df.groupby("search_key").mean()
        features = self._df.columns.to_list()
        features.pop(-1)

        for feature in features:
            x = self._df[feature]
            mean = x.mean(skipna=True)
            std = x.std(skipna=True)
            # determine anomalies based on mean, std, threshold
            anomalies = x > mean + threshold * std
            anomalies |= x < mean - threshold * std
            self._df.loc[anomalies, feature] = self._df["search_key"].map(
                load_profiles_from_data[feature]
            )

        self._df.drop(columns=["search_key"], inplace=True)

        logger = logging.getLogger(__name__)
        logger.info("finished removal of outliers.")

    def _make_time(self) -> None:
        """Generate date features.

        These include:
        1. date month (cyclicially encoded using sin / cos)
        2. date year (min-max scaled to interval [0,1])
        3. date season (min-max scaled to interval [0,1])
        4. indicator if date is holiday / weekday / weekend.
        """
        # split date into cyclically encoded month (2x) year
        self._df["date_month"] = self._df.index.month
        self._df["date_month_sin"] = np.sin(2 * np.pi * self._df["date_month"] / 12)
        self._df["date_month_cos"] = np.cos(2 * np.pi * self._df["date_month"] / 12)

        # add year
        self._df["date_year"] = self._df.index.year

        # use German holiday calendar lower saxony / Niedersachsen (NI)
        ger_holidays = holidays.country_holidays(
            "DE",
            subdiv="NI",
            years=range(2018, 2021),
        )
        ger_holidays = list(ger_holidays.keys())

        # create feature if holiday
        self._df["date_is_holiday"] = np.isin(self._df.index.date, ger_holidays)

        # create workday feature
        self._df["date_is_weekday"] = self._df.index.weekday <= 4
        self._df["date_is_weekend"] = self._df.index.weekday > 4

        self._df[["date_is_holiday", "date_is_weekday", "date_is_weekend"]] = self._df[
            ["date_is_holiday", "date_is_weekday", "date_is_weekend"]
        ].astype(int)

        # create season feature, as seen here: https://en.wikipedia.org/wiki/Season
        self._df["date_season"] = self._df["date_month"]
        map = {
            1: 0,
            2: 0,
            3: 1,
            4: 1,
            5: 1,
            6: 2,
            7: 2,
            8: 2,
            9: 3,
            10: 3,
            11: 3,
            12: 0,
        }
        self._df["date_season"].replace(map, inplace=True)

        # scale year and season to range (0, 1)
        self._df[["date_year", "date_season"]] = minmax_scale(
            self._df[["date_year", "date_season"]]
        )

        self._df.drop(columns=["date_month"], inplace=True)

    def _make_spq_total(self, pv: bool = False) -> None:
        """
        Create row-wise sums of S, P and Q over all households.

        Sum e. g., S_TOT might be different from all phases of S.
        Args:
            pv (bool, optional):  photovoltaic households. Defaults to False.
        """
        s_tot_households = list(
            filter(lambda x: x.endswith("_S_TOT"), self._df.columns)
        )
        if pv:
            p_tot_households = list(
                filter(lambda x: x.endswith("_P_TOT_WITH_PV"), self._df.columns)
            )
        else:
            p_tot_households = list(
                filter(lambda x: x.endswith("_P_TOT"), self._df.columns)
            )
        q_tot_households = list(
            filter(lambda x: x.endswith("_Q_TOT"), self._df.columns)
        )

        self._df["S_TOT"] = self._df[s_tot_households].sum(axis=1)
        self._df["P_TOT"] = self._df[p_tot_households].sum(axis=1)
        self._df["Q_TOT"] = self._df[q_tot_households].sum(axis=1)

        logger = logging.getLogger(__name__)
        logger.info("finished making 'S_TOT', 'P_TOT', 'Q_TOT'.")
