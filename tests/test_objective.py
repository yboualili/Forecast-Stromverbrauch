"""
Tests for Objectives.

Includes LSTM, CNN and AE.
"""
import datetime as dt
import os
import unittest
from unittest.mock import patch

import optuna
import pandas as pd

from model_selection.objective import (
    AEObjective,
    CNNObjective,
    LSTMObjective,
    TransformerObjective,
    split_train_test,
)


class TestObjectives(unittest.TestCase):
    """
    Perform automated tests for objectives.

    Args:
        metaclass (_type_, optional): parent. Defaults to abc.ABCMeta.
    """

    def setUp(self) -> None:
        """
        Set up basic data.

        Construct feature matrix and target with length of raw dataset.
        """
        self._old_cwd = os.getcwd()
        start = dt.datetime(2018, 1, 1).replace(tzinfo=dt.timezone.utc)
        end = dt.datetime(2020, 12, 31).replace(tzinfo=dt.timezone.utc)
        index = pd.date_range(start=start, end=end, freq="15min")

        self._X = pd.DataFrame(index=index)
        self._X = self._X.assign(new=1)
        self._y = pd.Series(index=index, data=self._X["new"])

    def test_cnn_objective(self) -> None:
        """
        Test if CNN objective returns a valid value.

        Value obtained is the MAPE. Should lie in [0,inf).
        Value may not be NaN.
        """
        os.chdir("./src/")
        params = {
            "batch_size": 512,
            "dropout": 0,
            "filters": 8,
            "hidden_units": 8,
            "lr": 1e-1,
            "pool_size": 2,
        }

        study = optuna.create_study(direction="minimize")
        objective = CNNObjective(self._X, self._y)

        # patch epochs with one to reduce runtime
        with patch.object(objective, "epochs", 1):
            study.enqueue_trial(params)
            study.optimize(objective, n_trials=1)

            # set to old cwd
            os.chdir(self._old_cwd)

            # check if MAPE is >= 0. See: https://bit.ly/3olxgUa
            self.assertGreaterEqual(study.best_value, 0.0)

    @unittest.skip(
        reason="Skip. Transformer is compute-intensive."
        "More than our (small) digital ocean droplet can handle."
    )
    def test_transformer_objective(self) -> None:
        """
        Test if Transformer objective returns a valid value.

        Value obtained is the MAPE. Should lie in [0,inf).
        Value may not be NaN.
        """
        os.chdir("./src/")
        params = {
            "batch_size": 512,
            "hidden_units": 8,
            "lr": 1e-1,
            "num_layers": 2,
        }

        study = optuna.create_study(direction="minimize")
        objective = TransformerObjective(self._X, self._y)

        # patch epochs with one to reduce runtime
        with patch.object(objective, "epochs", 1):
            study.enqueue_trial(params)
            study.optimize(objective, n_trials=1)

            # set to old cwd
            os.chdir(self._old_cwd)

            # check if MAPE is >= 0. See: https://bit.ly/3olxgUa
            self.assertGreaterEqual(study.best_value, 0.0)

    def test_lstm_objective(self) -> None:
        """
        Test if LSTM objective returns a valid value.

        Value obtained is the MAPE. Should lie in [0,inf).
        Value may not be NaN.
        """
        os.chdir("./src/")
        params = {
            "batch_size": 512,
            "hidden_units": 8,
            "lr": 1e-1,
            "num_layers": 2,
        }

        study = optuna.create_study(direction="minimize")
        objective = LSTMObjective(self._X, self._y)

        # patch epochs with one to reduce runtime
        with patch.object(objective, "epochs", 1):
            study.enqueue_trial(params)
            study.optimize(objective, n_trials=1)

            # set to old cwd
            os.chdir(self._old_cwd)

            # check if MAPE is >= 0. See: https://bit.ly/3olxgUa
            self.assertGreaterEqual(study.best_value, 0.0)

    def test_ae_objective(self) -> None:
        """
        Test if auto encoder objective returns a valid value.

        Value obtained is the MSE. Check if it is non negative.
        """
        os.chdir("./src/")
        params = {
            "activation": "ReLU",
            "batch_size": 512,
            "bottleneck_capacity": 8,
            "dropout": 0,
            "lr": 1e-1,
            "num_layers": 2,
        }

        study = optuna.create_study(direction="minimize")
        objective = AEObjective(self._X, self._y)

        # patch epochs with one to reduce runtime
        with patch.object(objective, "epochs", 1):

            study.enqueue_trial(params)
            study.optimize(objective, n_trials=1)

            # set to old cwd
            os.chdir(self._old_cwd)

            # check if MAPE is >= 0. See: https://bit.ly/3olxgUa
            self.assertGreaterEqual(study.best_value, 0.0)

    def test_split_train_test(self) -> None:
        """
        Test if train test split returns sequence works as expected.

        Twelve months are used for training, remainder of second year
        for validation and third year for oos testing.
        """
        # set to old cwd
        os.chdir(self._old_cwd)

        X_train, y_train, X_val, y_val, X_test, y_test = split_train_test(
            self._X, self._y
        )
        for date in X_train.index:
            self.assertEqual(date.year, 2018)
        for date in y_train.index:
            self.assertEqual(date.year, 2018)
        for date in X_val.index:
            self.assertEqual(date.year, 2019)
        for date in y_val.index:
            self.assertEqual(date.year, 2019)
        for date in X_test.index:
            self.assertEqual(date.year, 2020)
        for date in y_test.index:
            self.assertEqual(date.year, 2020)
