"""
Perform automated tests.

Includes tests for correct slicing, padding and len().
"""

import unittest

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from data.dataset import SequenceDataset


class TestDataLoader(unittest.TestCase):
    """
    Perform automated tests.

    Args:
        unittest (_type_): testcase
    """

    def test_len_partial(self) -> None:
        """
        Test, if length returned by data set is correct, if not evenly dividable.

        Example:
        Sequence has 10 elements, length of input is 6, length of prediction is 3.

        First 6 elements are used for training, predicting the next 3 elements.
        Next, window can be moved forward once predicting the 10th element
        and padding.

        Thus, total length is 2.
        """
        X = pd.DataFrame(np.arange(30).reshape(10, 3))
        y = pd.Series(np.arange(11))

        training_data = SequenceDataset(
            X=X, y=y, seq_length_input=6, seq_length_prediction=3
        )
        assert len(training_data) == 2

    def test_len_full(self) -> None:
        """
        Test, if length returned by data set is correct, if data can be evenly divided.

        Example:
        Sequence has 12 elements, length of input is 6, length of prediction is 3.

        First 6 elements are used for training, predicting the next 3 elements.
        Next, window can be moved forward once predicting the 10th - 12th element.

        Thus, total length is 2.
        """
        X = pd.DataFrame(np.arange(36).reshape(12, 3))
        y = pd.Series(np.arange(13))

        training_data = SequenceDataset(
            X=X, y=y, seq_length_input=6, seq_length_prediction=3
        )
        assert len(training_data) == 2

    def test_shape_features_target(self) -> None:
        """
        Test correct shapes of features and target with following shapes.

        features.shape = [batch_size, sequence_length_inputs, number_of_features]
        target.shape = [batch_size, sequence_length_prediction]

        A data loader is applied to generate batches of data.
        """
        batch_size, seq_length_input, seq_length_prediction, num_features = 2, 6, 3, 3
        X = pd.DataFrame(np.arange(30).reshape(10, 3))
        y = pd.Series(np.arange(11))

        training_data = SequenceDataset(
            X=X,
            y=y,
            seq_length_input=seq_length_input,
            seq_length_prediction=seq_length_prediction,
        )
        train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=False)
        inputs, target = next(iter(train_loader))
        input_shape, target_shape = list(inputs.shape), list(target.shape)

        self.assertListEqual(input_shape, [batch_size, seq_length_input, num_features])
        self.assertListEqual(target_shape, [batch_size, seq_length_prediction])

    def test_padding(self) -> None:
        """
        Test, if padding is added to underful target sequence.

        Example for input length 4, prediction length 3:
        [0, 1, 2, 3, 4, 5, 6, 7]

        Input: [0, 1, 2, 3], target: [4, 5, 6] and [7, padding, padding]

        """
        X = pd.DataFrame(np.arange(24).reshape(8, 3))
        y = pd.Series(np.arange(8))

        seq_length_prediction = 3
        training_data = SequenceDataset(
            X=X, y=y, seq_length_input=4, seq_length_prediction=seq_length_prediction
        )
        train_loader = DataLoader(training_data, batch_size=2, shuffle=False)

        _, target = next(iter(train_loader))
        self.assertListEqual(target.tolist()[-1], [7.0] * seq_length_prediction)

    def test_no_padding(self) -> None:
        """
        Test, that no padding is applied, if data can be evenly divided into sequences.

        Example for input length 4, prediction length 4:
        [0, 1, 2, 3, 4, 5, 6, 7]

        Input: [0, 1, 2, 3], target: [4, 5, 6, 7]
        """
        X = pd.DataFrame(np.arange(24).reshape(8, 3))
        y = pd.Series(np.arange(8))

        training_data = SequenceDataset(
            X=X, y=y, seq_length_input=4, seq_length_prediction=4
        )
        train_loader = DataLoader(training_data, batch_size=2, shuffle=False)

        _, target = next(iter(train_loader))
        self.assertListEqual(target.tolist()[-1], [4.0, 5.0, 6.0, 7.0])

    def test_split_mutually_exclusive_target(self) -> None:
        """
        Test, if target is mutually exclusive.

        That is, no target sequence contains elements of another target sequence.
        """
        X = pd.DataFrame(np.arange(24).reshape(8, 3))
        y = pd.Series(np.arange(8))

        training_data = SequenceDataset(
            X=X, y=y, seq_length_input=4, seq_length_prediction=3
        )
        train_loader = DataLoader(training_data, batch_size=2, shuffle=False)

        inputs, target = next(iter(train_loader))

        batch = target.tolist()
        slice_1, slice_2 = batch[0], batch[1]
        assert not any(i in slice_1 for i in slice_2)

    def test_split_feature_target(self) -> None:
        """
        Test, if slices of features and target contain correct values.

        Slice of features start at beginning with seq. length input and
        is moved forward by seq. length prediction.

        Slice of target starts after features.
        """
        X = pd.DataFrame(np.arange(24).reshape(8, 3))
        y = pd.Series(np.arange(9))

        training_data = SequenceDataset(
            X=X, y=y, seq_length_input=4, seq_length_prediction=3
        )
        train_loader = DataLoader(training_data, batch_size=2, shuffle=False)

        inputs, target = next(iter(train_loader))

        batch_inputs = inputs.tolist()
        batch_target = target.tolist()
        slice_1_inputs, slice_2_inputs = batch_inputs[0], batch_inputs[1]
        slice_1_target, slice_2_target = batch_target[0], batch_target[1]

        self.assertListEqual(
            slice_1_inputs,
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0], [9.0, 10.0, 11.0]],
        )
        self.assertListEqual(
            slice_2_inputs,
            [
                [9.0, 10.0, 11.0],
                [12.0, 13.0, 14.0],
                [15.0, 16.0, 17.0],
                [18.0, 19.0, 20.0],
            ],
        )
        self.assertListEqual(slice_1_target, [4.0, 5.0, 6.0])
        self.assertListEqual(slice_2_target, [7.0, 8.0, 8.0])
