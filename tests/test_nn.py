"""
Tests for Neural networks.

See:
https://thenerdstation.medium.com/how-to-unit-test-machine-learning-code-57cf6fd81765
http://karpathy.github.io/2019/04/25/recipe/
https://krokotsch.eu/posts/deep-learning-unit-tests/
"""
import unittest

import torch
import torch.nn as nn
import torch.optim as optim

from models.autoencoder import Autoencoder
from models.cnn import CNN
from models.lstm import LSTM
from models.transformer import Transformer


class TestNN(unittest.TestCase):
    """
    Perform automated tests for neural networks.

    Args:
        metaclass (_type_, optional): parent. Defaults to abc.ABCMeta.
    """

    def get_outputs(self) -> torch.Tensor:
        """
        Return relevant output of model.

        Returns:
            torch.Tensor: outputs
        """
        outputs = self.net(self.inputs)
        return outputs

    def make_deterministic(self, seed: int = 42) -> None:
        """
        Set up a fixed  seed for the random number generator in torch.

        Args:
            seed (int, optional): seed of random number generator. Defaults to 42.
        """
        # PyTorch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def setUp(self) -> None:
        """
        Set up basic network and data.

        Prepares inputs and expected outputs for testing.
        """
        self.num_features = 5
        self.batch_size = 1
        self.hidden_units = 64
        self.seq_length_input = 3 * 96
        self.seq_length_output = 96
        self.epochs = 256
        self.threshold = 5e-2

        self.make_deterministic()

        # lstm moves to device autoamtically, if available. see lstm.py
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.inputs = torch.randn(
            self.batch_size, self.seq_length_input, self.num_features
        ).to(device)
        self.expected_outputs = torch.randn(self.batch_size, self.seq_length_output).to(
            device
        )

        self.net = LSTM(
            num_features=self.num_features,
            hidden_units=self.hidden_units,
            num_layers=1,
            batch_size=self.batch_size,
            seq_length_input=self.seq_length_input,
            seq_length_output=self.seq_length_output,
            dropout=0,
        ).to(device)

    @torch.no_grad()
    def test_shapes(self) -> None:
        """
        Test, if shapes of the network equal the targets.

        Loss might be calculated due to broadcasting, but might be wrong.

        Adapted from: # https://krokotsch.eu/posts/deep-learning-unit-tests/
        """
        outputs = self.get_outputs()
        self.assertEqual(self.expected_outputs.shape, outputs.shape)

    def test_convergence(self) -> None:
        """
        Tests, whether loss approaches zero for single batch.

        Training on a single batch leads to serious overfitting.
        If loss does not become, this indicates a possible error.

        See: http://karpathy.github.io/2019/04/25/recipe/
        """
        optimizer = optim.SGD(self.net.parameters(), lr=0.25)
        criterion = nn.MSELoss()

        for _ in range(self.epochs):

            self.net.train()
            optimizer.zero_grad()

            outputs = self.get_outputs()
            loss = criterion(outputs, self.expected_outputs)

            loss.backward()
            optimizer.step()

        self.assertLessEqual(loss.detach().cpu().numpy(), self.threshold)

    @unittest.skip(
        reason="Skip. Some nets reside on gpu automatically, if gpu is found."
    )
    def test_device_moving(self) -> None:
        """
        Test, if all tensors reside on the gpu / cpu.

        Adapted from: https://krokotsch.eu/posts/deep-learning-unit-tests/
        """
        net_on_gpu = self.net.to("cuda:0")
        net_back_on_cpu = net_on_gpu.cpu()

        torch.manual_seed(42)
        outputs_cpu = self.net(self.inputs)
        torch.manual_seed(42)
        outputs_gpu = net_on_gpu(self.inputs.to("cuda:0"))
        torch.manual_seed(42)
        outputs_back_on_cpu = net_back_on_cpu(self.inputs)

        self.assertAlmostEqual(0.0, torch.sum(outputs_cpu - outputs_gpu.cpu()))
        self.assertAlmostEqual(0.0, torch.sum(outputs_cpu - outputs_back_on_cpu))

    def test_batch_independence(self) -> None:
        """
        Checks sample independence by performing of inputs.

        Required as SGD-based algorithms like ADAM work on mini-batches.
        Batching training samples assumes that your model can process each
        sample as if they were fed individually. In other words, the samples in
        your batch do not influence each other when processed. This assumption
        is a brittle one and can break with one misplaced reshape or aggregation
        over a wrong tensor dimension.

        Adapted from: https://krokotsch.eu/posts/deep-learning-unit-tests/
        """
        self.inputs.requires_grad = True
        # Compute forward pass in eval mode to deactivate batch norm
        # self.net.eval()
        outputs = self.get_outputs()

        self.net.train()

        # Mask loss for certain samples in batch
        mask_idx = torch.randint(0, self.batch_size, ())
        mask = torch.ones_like(outputs)
        mask[mask_idx] = 0
        outputs = outputs * mask

        # Compute backward pass
        loss = outputs.mean()
        loss.backward()

        # Check if gradient exists and is zero for masked samples
        for i, grad in enumerate(self.inputs.grad):
            if i == mask_idx:
                self.assertTrue(torch.all(grad == 0).item())
            else:
                self.assertTrue(not torch.all(grad == 0))

    def test_all_parameters_updated(self) -> None:
        """
        Test, if all parameters are updated.

        If parameters are not updated this could indicate dead ends.
        Adapted from: https://krokotsch.eu/posts/deep-learning-unit-tests/
        """
        optim = torch.optim.SGD(self.net.parameters(), lr=0.01)

        outputs = self.get_outputs()
        loss = outputs.mean()
        loss.backward()
        optim.step()

        for param_name, param in self.net.named_parameters():
            if param.requires_grad:
                with self.subTest(name=param_name):
                    self.assertIsNotNone(param.grad)
                    sum = torch.sum(param.grad**2)
                    # special case if sum is tensor not scalar
                    self.assertNotEqual(0.0, sum)


class TestLSTM(TestNN):
    """
    Perform automated tests for LSTMs.

    Args:
        unittest (_type_): testcase
    """

    def setUp(self) -> None:
        """
        Set up basic LSTM and data.

        Prepares inputs and expected outputs for testing.
        """
        self.num_features = 5
        self.batch_size = 1
        self.hidden_units = 64
        self.seq_length_input = 3 * 96
        self.seq_length_output = 96
        self.epochs = 128
        self.threshold = 5e-2

        # lstm moves to device autoamtically, if available. see lstm.py
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.inputs = torch.randn(
            self.batch_size, self.seq_length_input, self.num_features
        ).to(device)
        self.expected_outputs = torch.randn(self.batch_size, self.seq_length_output).to(
            device
        )

        self.net = LSTM(
            num_features=self.num_features,
            hidden_units=self.hidden_units,
            num_layers=1,
            batch_size=self.batch_size,
            seq_length_input=self.seq_length_input,
            seq_length_output=self.seq_length_output,
            dropout=0,
        ).to(device)


class TestCNN(TestNN):
    """
    Perform automated tests for CNNs.

    Args:
        unittest (_type_): testcase
    """

    def setUp(self) -> None:
        """
        Set up basic CNN and data.

        Prepares inputs and expected outputs for testing.
        """
        self.num_features = 5
        self.batch_size = 1
        self.hidden_units = 64
        self.epochs = 512
        self.threshold = 5e-2
        self.seq_length_input = 3 * 96
        self.seq_length_output = 96
        self.pool_size = 2
        self.filters = 64
        self.dropout = 0

        # cnn moves to device autoamtically, if available. see lstm.py
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.inputs = torch.randn(
            self.batch_size, self.seq_length_input, self.num_features
        ).to(device)
        self.expected_outputs = torch.randn(self.batch_size, self.seq_length_output).to(
            device
        )

        self.net = CNN(
            num_features=self.num_features,
            hidden_units=self.hidden_units,
            batch_size=self.batch_size,
            seq_length_input=self.seq_length_input,
            seq_length_output=self.seq_length_output,
            filters=self.filters,
            pool_size=self.pool_size,
            dropout=self.dropout,
        ).to(device)


class TestAE(TestNN):
    """
    Perform automated tests for AutoEncoder.

    Args:
        unittest (_type_): testcase
    """

    def setUp(self) -> None:
        """
        Set up basic AutoEncoder and data.

        Prepares inputs and expected outputs for testing.
        """
        self.num_features = 3
        self.batch_size = 1
        self.num_layers = 2
        self.bottleneck_capacity = 2
        self.dropout = 0.1
        self.activation = "ReLU"
        self.epochs = 16384
        self.threshold = 5e-2

        self.inputs = torch.randn(self.batch_size, self.num_features)
        self.expected_outputs = self.inputs.clone()

        self.net = Autoencoder(
            num_features=self.num_features,
            num_layers=self.num_layers,
            bottleneck_capacity=self.bottleneck_capacity,
            activation=self.activation,
        )

    def get_outputs(self) -> torch.Tensor:
        """
        Overwrite parent.

        Returns:
            torch.Tensor: outputs
        """
        outputs, _ = self.net(self.inputs)
        return outputs

    def test_all_parameters_updated(self) -> None:
        """
        Overwrite parent.

        Sets up a larger AE with more inputs.

        """
        self.num_features = 10

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.net = Autoencoder(
            num_features=self.num_features,
            bottleneck_capacity=3,
            num_layers=1,
            dropout=0.5,
            activation="ReLU",
        ).to(device)

        self.inputs = torch.randn(16384, self.num_features).to(device)

        super().test_all_parameters_updated()


class TestTransformer(TestNN):
    """
    Perform automated tests for Transformer.

    Args:
        unittest (_type_): testcase
    """

    def setUp(self) -> None:
        """
        Set up basic Transformer and data.

        Prepares inputs and expected outputs for testing.
        """
        self.make_deterministic()

        self.num_features = 1
        self.batch_size = 1
        self.hidden_units = 8
        self.seq_length_input = 3 * 96
        self.seq_length_output = 96
        self.epochs = 256
        self.threshold = 5e-2

        # cnn moves to device autoamtically, if available. see lstm.py
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.inputs = torch.randn(
            self.batch_size, self.seq_length_input, self.num_features
        ).to(device)
        self.expected_outputs = torch.randn(self.batch_size, self.seq_length_output).to(
            device
        )

        self.net = Transformer(
            num_features=self.num_features,
            dec_seq_len=self.seq_length_input,
            max_seq_len=self.seq_length_input,
            out_seq_len=self.seq_length_output,
            dim_val=128,
            n_encoder_layers=4,
            n_decoder_layers=4,
            n_heads=8,
            dropout_encoder=0.2,
            dropout_decoder=0.2,
            dropout_pos_enc=0.2,
            dim_feedforward_encoder=64,
            dim_feedforward_decoder=64,
        ).to(device)

        print(self.net.named_parameters())

    def get_outputs(self) -> torch.Tensor:
        """
        Return relevant output of model.

        super().test_all_parameters_updated()
        Returns:
            torch.Tensor: outputs
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        outputs = self.net(self.inputs, self.expected_outputs).to(device)
        return outputs

    def test_convergence(self) -> None:
        """
        Tests, whether loss approaches zero for single batch.

        Training on a single batch leads to serious overfitting.
        If loss does not become, this indicates a possible error.

        See: http://karpathy.github.io/2019/04/25/recipe/
        """
        optimizer = optim.SGD(self.net.parameters(), lr=3e-4)
        criterion = nn.MSELoss()

        for _ in range(self.epochs):

            self.net.train()
            optimizer.zero_grad()

            outputs = self.get_outputs()
            loss = criterion(outputs, self.expected_outputs)

            loss.backward()
            optimizer.step()
            print(loss)

        self.assertLessEqual(loss.detach().cpu().numpy(), self.threshold)
