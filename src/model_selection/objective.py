"""
Provides objective for optimization.

Adds support for Autoencoder, CNNs, LSTMs, and Transformer.
"""

from __future__ import annotations

import os

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pandas.tseries.offsets import DateOffset
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from src.data.dataset import SequenceDataset
from src.models.autoencoder import Autoencoder
from src.models.cnn import CNN
from src.models.lstm import LSTM
from src.models.transformer import Transformer
from src.optim.early_stopping import EarlyStopping


def split_train_test(
    X: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Perform a train, validation and test split.

    Note: X_test and y_test are larger due to 2020 being a leap year.

        X (pd.DataFrame): explanatory variables
        y (pd.Series): target

    Returns:
        tuple[pd.DataFrame,pd.Series, pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    # cut of first few timestamps so the data starts at 00:00:00
    train_start = X.at_time("0:00").index[0]

    # use 1 year for training, start validation at next day
    train_end = train_start + DateOffset(years=1) - pd.Timedelta(minutes=15)
    val_start = train_end + pd.Timedelta(minutes=15)
    val_end = "2019-12-31 23:45:00+00:00"
    test_start = "2020-01-01 00:00:00+00:00"
    test_end = "2020-12-31 23:45:00+00:00"

    X_train, X_val, X_test = (
        X.loc[train_start:train_end, :].copy(),  # type: ignore
        X.loc[val_start:val_end].copy(),  # type: ignore
        X.loc[test_start:test_end].copy(),  # type: ignore
    )
    y_train, y_val, y_test = (
        y.loc[train_start:train_end],  # type: ignore
        y.loc[val_start:val_end],  # type: ignore
        y.loc[test_start:test_end],  # type: ignore
    )

    features = X.columns.to_list()
    date_features = list(filter(lambda x: x.startswith("date"), X.columns))
    features = [feature for feature in features if feature not in date_features]

    feature_scaler = StandardScaler()

    X_train[features] = feature_scaler.fit_transform(X_train[features])
    X_val[features] = feature_scaler.transform(X_val[features])
    X_test[features] = feature_scaler.transform(X_test[features])

    print(f"X.shape: {X.shape}")
    print(f"X_train.shape: {X_train.shape}, y_train.shape: {y_train.shape}")
    print(f"X_val.shape: {X_val.shape}, y_val.shape: {y_val.shape}")
    print(f"X_test.shape: {X_test.shape}, y_test.shape: {y_test.shape}")

    return X_train, y_train, X_val, y_val, X_test, y_test


class LSTMObjective(object):
    """
    Implements an optuna optimization objective.

    See here: https://optuna.readthedocs.io/en/stable/
    Args:
        object (object): object
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series, name: str = "default"):
        """
        Initialize objective.

        Args:
            X (pd.DataFrame): feature matrix
            y (pd.Series): targets
            name (str, optional): Name of Objective. Used in filename.
            Defaults to "default".
        """
        self._X_train, self._y_train, self._X_val, self._y_val, _, _ = split_train_test(
            X, y
        )
        self.name = name
        self.epochs = 4096

    def __call__(self, trial: optuna.Trial) -> float:
        """
        Perform a new search trial in Bayesian search.

        Hyperarameters are suggested, unless they are fixed.
        LSTM is trained on training set.

        Args:
            trial (optuna.Trial): current trial.

        Returns:
            float: MAPE of trial on test set.
        """
        # static params
        weight_decay = 0.001
        dropout = 0
        seq_length_input = 96 * 3
        seq_length_output = 96
        input_shape = self._X_train.shape[1]

        # searchable params
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
        hidden_units = trial.suggest_int("hidden_units", 4, 128, log=False)
        num_layers = trial.suggest_int("num_layers", 1, 2, log=False)
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

        # create training and test set
        training_data = SequenceDataset(
            X=self._X_train,
            y=self._y_train,
            seq_length_input=seq_length_input,
            seq_length_prediction=seq_length_output,
        )

        test_data = SequenceDataset(
            X=self._X_val,
            y=self._y_val,
            seq_length_input=seq_length_input,
            seq_length_prediction=seq_length_output,
        )

        train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        #  use gpu if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load model and move to device
        model = LSTM(
            num_features=input_shape,
            hidden_units=hidden_units,
            num_layers=num_layers,
            batch_size=batch_size,
            seq_length_input=seq_length_input,
            seq_length_output=seq_length_output,
            dropout=dropout,
        ).to(device)

        # Generate the optimizers
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # mean squared error
        criterion = nn.MSELoss()

        # keep track of val loss and do early stopping
        early_stopping = EarlyStopping(patience=5)

        train_history, test_history = [], []
        writer = SummaryWriter(
            os.path.join(
                "models",
                "run",
                "summary_tb",
                f"LSTM_{self.name}",
                f"LSTM_obj_trial{trial.number}_bs{batch_size}"
                f"_lr{lr}_nl{num_layers}_hu{hidden_units}",
            )
        )

        for epoch in range(self.epochs):

            # perform training
            loss_in_epoch_train = 0

            model.train()

            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                # reset the gradients back to zero
                # PyTorch accumulates gradients on subsequent backward passes
                optimizer.zero_grad()

                outputs = model(inputs)

                train_loss = criterion(outputs, targets)

                # compute accumulated gradients
                train_loss.backward()

                # perform parameter update based on current gradients
                optimizer.step()

                # add the mini-batch training loss to epoch loss
                loss_in_epoch_train += train_loss.item()

            # evaluate model on test set.
            # Thus, disable gradient calculation and go into evaluation mode.
            model.eval()

            loss_in_epoch_test = 0

            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    outputs = model(inputs)

                    test_loss = criterion(outputs, targets)
                    loss_in_epoch_test += test_loss.item()

            train_loss = loss_in_epoch_train / len(train_loader)
            test_loss = loss_in_epoch_test / len(test_loader)

            writer.add_scalar("Training loss", train_loss, global_step=epoch)
            writer.add_scalar("Test loss", test_loss, global_step=epoch)

            train_history.append(train_loss)
            test_history.append(test_loss)

            print(f"epoch : {epoch + 1}/{self.epochs},", end=" ")
            print(f"loss (train) = {train_loss:.8f}, loss (test) = {test_loss:.8f}")

            # return early if val loss doesn't decrease for several iterations
            early_stopping(test_loss)
            if early_stopping.early_stop:
                break

        writer.add_graph(model, inputs)
        writer.close()

        # save model for each trial
        torch.save(
            model.state_dict(),
            f"../models/{model.__class__.__name__}"
            f"_{self.name}_trial_{trial.number}.pth",
        )

        trial.report(test_loss, epoch)

        # make predictions with final model
        predictions_all, targets_all = [], []

        model.eval()

        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            prediction = model(inputs)
            predictions_all.append(prediction.detach().cpu().numpy().flatten())
            targets_all.append(targets.numpy().flatten())

        # saveguard NaN MAPE if prediction contains NaN
        # replace with 1e-3 for large MAPE.
        predictions_all = np.concatenate(predictions_all)
        targets_all = np.concatenate(targets_all)
        predictions_all[np.isnan(predictions_all)] = 1e-3

        return mean_absolute_percentage_error(predictions_all, targets_all)


class TransformerObjective(object):
    """
    Implements an optuna optimization objective.

    See here: https://optuna.readthedocs.io/en/stable/
    Args:
        object (object): object
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series, name: str = "default"):
        """
        Initialize objective.

        Args:
            X (pd.DataFrame): feature matrix
            y (pd.Series): targets
            name (str, optional): Name of Objective. Used in filename.
            Defaults to "default".
        """
        self._X_train, self._y_train, self._X_val, self._y_val, _, _ = split_train_test(
            X, y
        )
        self.name = name
        self.epochs = 128

    def __call__(self, trial: optuna.Trial) -> float:
        """
        Perform a new search trial in Bayesian search.

        Hyperarameters are suggested, unless they are fixed.
        LSTM is trained on training set.

        Args:
            trial (optuna.Trial): current trial.

        Returns:
            float: MAPE of trial on test set.
        """
        # static params
        weight_decay = 0.001
        dropout = 0.2
        seq_length_input = 96 * 3
        seq_length_output = 96
        input_shape = self._X_train.shape[1]
        n_heads = 8

        # searchable params
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
        # dim must be evenly divisable by attention heads (normally 8)
        hidden_units = trial.suggest_categorical(
            "hidden_units", [1 * n_heads, 2 * n_heads, 3 * n_heads, 4 * n_heads]
        )
        num_layers = trial.suggest_int("num_layers", 1, 5, log=False)
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)

        # create training and test set
        training_data = SequenceDataset(
            X=self._X_train,
            y=self._y_train,
            seq_length_input=seq_length_input,
            seq_length_prediction=seq_length_output,
        )

        test_data = SequenceDataset(
            X=self._X_val,
            y=self._y_val,
            seq_length_input=seq_length_input,
            seq_length_prediction=seq_length_output,
        )

        train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        #  use gpu if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load model to device
        model = Transformer(
            num_features=input_shape,
            dec_seq_len=seq_length_input,
            max_seq_len=seq_length_input,
            out_seq_len=seq_length_output,
            dim_val=hidden_units,
            n_encoder_layers=num_layers,
            n_decoder_layers=num_layers,
            n_heads=n_heads,
            dropout_encoder=dropout,
            dropout_decoder=dropout,
            dropout_pos_enc=dropout,
            dim_feedforward_encoder=hidden_units,
            dim_feedforward_decoder=hidden_units,
        ).to(device)

        # Generate the optimizers
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # mean squared error
        criterion = nn.MSELoss()

        # keep track of val loss and do early stopping
        early_stopping = EarlyStopping(patience=5)

        train_history, test_history = [], []

        writer = SummaryWriter(
            os.path.join(
                "models",
                "run",
                "summary_tb",
                f"Transformer_{self.name}",
                f"Transformer_obj_trial{trial.number}_bs{batch_size}"
                f"_lr{lr}_nl{num_layers}_hu{hidden_units}",
            )
        )

        for epoch in range(self.epochs):

            # perform training
            loss_in_epoch_train = 0

            model.train()

            for inputs, targets in train_loader:

                inputs = inputs.to(device)
                targets = targets.to(device)

                # reset the gradients back to zero
                optimizer.zero_grad()

                outputs = model(inputs, targets)

                train_loss = criterion(outputs, targets)

                # compute accumulated gradients
                train_loss.backward()

                # perform parameter update based on current gradients
                optimizer.step()

                # add the mini-batch training loss to epoch loss
                loss_in_epoch_train += train_loss.item()

                # Decay Learning Rate
                # scheduler.step()

            model.eval()

            loss_in_epoch_test = 0.0

            with torch.no_grad():
                for inputs, targets in test_loader:

                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    outputs = model(inputs, targets)

                    test_loss = criterion(outputs, targets)
                    loss_in_epoch_test += test_loss.item()

            train_loss = loss_in_epoch_train / len(train_loader)
            test_loss = loss_in_epoch_test / len(test_loader)

            writer.add_scalar("Training loss", train_loss, global_step=epoch)
            writer.add_scalar("Test loss", test_loss, global_step=epoch)

            train_history.append(train_loss)
            test_history.append(test_loss)

            print(f"epoch : {epoch + 1}/{self.epochs},", end=" ")
            print(f"loss (train) = {train_loss:.8f}, loss (test) = {test_loss:.8f}")

            # return early if val loss doesn't decrease for several iterations
            early_stopping(test_loss)
            if early_stopping.early_stop:
                break

        writer.close()

        # save model for each trial
        torch.save(
            model.state_dict(),
            f"../models/{model.__class__.__name__}"
            f"_{self.name}_trial_{trial.number}.pth",
        )

        trial.report(test_loss, epoch)

        # make predictions with final model
        predictions_all, targets_all = [], []

        model.eval()

        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            prediction = model(inputs, targets)
            predictions_all.append(prediction.detach().cpu().numpy().flatten())
            targets_all.append(targets.detach().cpu().numpy().flatten())

        # saveguard NaN MAPE if prediction contains NaN
        # replace with 1e-3 for large MAPE.
        predictions_all = np.concatenate(predictions_all)
        targets_all = np.concatenate(targets_all)
        predictions_all[np.isnan(predictions_all)] = 1e-3

        return mean_absolute_percentage_error(predictions_all, targets_all)


class CNNObjective(object):
    """
    Implements an optuna optimization objective.

    See here: https://optuna.readthedocs.io/en/stable/
    Args:
        object (object): object
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series, name: str = "default"):
        """
        Initialize objective.

        Args:
            X (pd.DataFrame): feature matrix
            y (pd.Series): targets
            name (str, optional): Name of Objective. Used in filename.
            Defaults to "default".
        """
        self._X_train, self._y_train, self._X_val, self._y_val, _, _ = split_train_test(
            X, y
        )
        self.name = name
        self.epochs = 4096

    def __call__(self, trial: optuna.Trail) -> float:
        """
        Perform a new search trial in Bayesian search.

        Hyperarameters are suggested, unless they are fixed.
        CNN is trained on training set.

        Args:
            trial (optuna.Trial): current trial.

        Returns:
            float: MAPE of trial on test set.
        """
        # static params
        weight_decay = 0.001
        seq_length_input = 96 * 3
        seq_length_output = 96
        input_shape = self._X_train.shape[1]

        # searchable params
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
        dropout = trial.suggest_float("dropout", 0, 0.5, log=False)
        filters = trial.suggest_int("filters", 4, 128, log=False)
        hidden_units = trial.suggest_int("hidden_units", 4, 128, log=False)
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        pool_size = trial.suggest_int("pool_size", 1, 16, log=False)

        training_data = SequenceDataset(
            X=self._X_train,
            y=self._y_train,
            seq_length_input=seq_length_input,
            seq_length_prediction=seq_length_output,
        )

        test_data = SequenceDataset(
            X=self._X_val,
            y=self._y_val,
            seq_length_input=seq_length_input,
            seq_length_prediction=seq_length_output,
        )

        train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        #  use gpu if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load model to device
        model = CNN(
            num_features=input_shape,
            hidden_units=hidden_units,
            batch_size=batch_size,
            seq_length_input=seq_length_input,
            seq_length_output=seq_length_output,
            filters=filters,
            pool_size=pool_size,
            dropout=dropout,
        ).to(device)

        # Generate the optimizers
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        # mean squared error
        criterion = nn.MSELoss()

        # keep track of val loss and do early stopping
        early_stopping = EarlyStopping(patience=5)

        train_history, test_history = [], []
        writer = SummaryWriter(
            os.path.join(
                "models",
                "run",
                "summary_tb",
                f"CNN_{self.name}",
                f"CNN_obj_trial{trial.number}_bs{batch_size}"
                f"_lr{lr}_fil{filters}_hu{hidden_units}"
                f"_ps{pool_size}_drop{dropout}",
            )
        )

        for epoch in range(self.epochs):

            # perform training
            loss_in_epoch_train = 0

            model.train()

            for inputs, targets in train_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)

                # reset the gradients back to zero.
                # PyTorch accumulates gradients on subsequent backward passes.
                optimizer.zero_grad()

                outputs = model(inputs)

                train_loss = criterion(outputs, targets)

                # compute accumulated gradients
                train_loss.backward()

                # perform parameter update based on current gradients
                optimizer.step()

                # add the mini-batch training loss to epoch loss
                loss_in_epoch_train += train_loss.item()

            model.eval()

            loss_in_epoch_test = 0

            with torch.no_grad():
                for inputs, targets in test_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)

                    outputs = model(inputs)

                    test_loss = criterion(outputs, targets)
                    loss_in_epoch_test += test_loss.item()

            train_loss = loss_in_epoch_train / len(train_loader)
            test_loss = loss_in_epoch_test / len(test_loader)

            writer.add_scalar("Training loss", train_loss, global_step=epoch)
            writer.add_scalar("Test loss", test_loss, global_step=epoch)

            train_history.append(train_loss)
            test_history.append(test_loss)

            print(f"epoch : {epoch + 1}/{self.epochs},", end=" ")
            print(f"loss (train) = {train_loss:.8f}, loss (test) = {test_loss:.8f}")

            # return early if val loss doesn't decrease for several iterations
            early_stopping(test_loss)
            if early_stopping.early_stop:
                break

        writer.add_graph(model, inputs)
        writer.close()

        # save model for each trial
        torch.save(
            model.state_dict(),
            f"../models/{model.__class__.__name__}"
            f"_{self.name}_trial_{trial.number}.pth",
        )

        trial.report(test_loss, epoch)

        # make predictions with final model
        predictions_all, targets_all = [], []

        model.eval()

        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            prediction = model(inputs)
            predictions_all.append(prediction.detach().cpu().numpy().flatten())
            targets_all.append(targets.numpy().flatten())

        # saveguard NaN MAPE if prediction contains NaN
        # replace with 1e-3 for large MAPE.
        predictions_all = np.concatenate(predictions_all)
        targets_all = np.concatenate(targets_all)
        predictions_all[np.isnan(predictions_all)] = 1e-3

        return mean_absolute_percentage_error(predictions_all, targets_all)


class AEObjective(object):
    """
    Implements an optuna optimization objective.

    See here: https://optuna.readthedocs.io/en/stable/
    Args:
        object (object): object
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series, name: str = "default"):
        """
        Initialize objective.

        Args:
            X (pd.DataFrame): feature matrix
            y (pd.Series): targets
            name (str, optional): Name of Objective. Used in filename.
            Defaults to "default".
        """
        self._X_train, _, self._X_val, _, _, _ = split_train_test(X, y)
        self.name = name
        self.epochs = 4096

    def __call__(self, trial: optuna.Trial) -> None:
        """
        Perform a new search trial in Bayesian search.

        Hyperarameters are suggested, unless they are fixed.
        Autoencoder is trained on training set.

        Implementation adapted from: https://bit.ly/3xTcP54.

        Args:
            trial (optuna.Trial): current trial.

        Returns:
            float: MSE of trial on test set.
        """
        # static params
        num_features = self._X_train.shape[1]

        # searchable params
        activation = trial.suggest_categorical(
            "activation", ["ReLU", "Sigmoid", "SiLU"]
        )
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
        bottleneck_capacity = trial.suggest_int("bottleneck_capacity", 8, 64, log=False)
        dropout = trial.suggest_float("dropout", 0, 0.5, log=False)
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        num_layers = trial.suggest_int("num_layers", 2, 5, log=False)

        dataset_train = TensorDataset(Tensor(self._X_train.astype(np.float32).values))
        dataset_test = TensorDataset(Tensor(self._X_val.astype(np.float32).values))

        train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

        #  use gpu if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # load model to device
        model = Autoencoder(
            num_features=num_features,
            bottleneck_capacity=bottleneck_capacity,
            num_layers=num_layers,
            dropout=dropout,
            activation=activation,
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)

        loss = nn.MSELoss()

        writer = SummaryWriter(
            os.path.join(
                "models",
                "run",
                "summary_tb",
                f"AE_{self.name}",
                f"AE_obj_trial{trial.number}_bs{batch_size}"
                f"_lr{lr}_num_layers{num_layers}_act{activation}"
                f"_bottleneck_cap{bottleneck_capacity}_drop{dropout}",
            )
        )

        # keep track of val loss and do early stopping
        early_stopping = EarlyStopping(patience=5)
        for epoch in range(self.epochs):

            loss_in_epoch_train = 0

            # perform training
            model.train()
            for batch_features in train_loader:

                # reshape mini-batch data to [N, [X.shape[1]] matrix
                batch_features = batch_features[0].to(device)

                optimizer.zero_grad()

                outputs, _ = model(batch_features)
                train_loss = loss(outputs, batch_features)

                # compute accumulated gradients
                train_loss.backward()

                # perform parameter update based on current gradients
                optimizer.step()

                # add the mini-batch training loss to epoch loss
                loss_in_epoch_train += train_loss.item()

            # Validation of the model e. g., disable Dropout when testing
            model.eval()

            loss_in_epoch_test = 0

            with torch.no_grad():
                for batch_features in test_loader:
                    # reshape mini-batch data to [N, [X.shape[1]] matrix
                    batch_features = batch_features[0].to(device)
                    outputs, _ = model(batch_features)

                    test_loss = loss(outputs, batch_features)
                    loss_in_epoch_test += test_loss.item()

            train_loss = loss_in_epoch_train / len(train_loader)
            test_loss = loss_in_epoch_test / len(test_loader)

            writer.add_scalar("Training loss", train_loss, global_step=epoch)
            writer.add_scalar("Test loss", test_loss, global_step=epoch)

            # return early if test loss doesn't decrease for several iterations
            early_stopping(test_loss)
            if early_stopping.early_stop:
                break

            print(f"epoch : {epoch + 1}/{self.epochs},", end=" ")
            print(f"loss (train) = {train_loss:.8f}, loss (test) = {test_loss:.8f}")

            trial.report(test_loss, epoch)

        writer.add_graph(model, batch_features)
        writer.close()

        # Save model for each trial
        torch.save(
            model.state_dict(),
            f"../models/{model.__class__.__name__}"
            f"_{self.name}_trial_{trial.number}.pth",
        )

        return test_loss
