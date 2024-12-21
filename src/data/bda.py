"""
Provides an OO-representation of the context.

Can be shared between pre-processing steps.
"""


import os
from os import path


class BDA:
    """
    OO representation of BDA files.

    Can be passed between other objects.
    """

    def __init__(self, input_path: str, output_path: str) -> None:
        """
        Construct a context object holding the input_path and output_path.

        Args:
            input_path (str): Path from where local data can be loaded
            output_path (str): Path to which data can be saved
        """
        self.input_path = input_path
        self.output_path = output_path

        self._mk_dir()

    def _mk_dir(self) -> None:
        """
        Make directories for input and output data.

        If directory already exists, no directories will be created.
        """
        if not path.exists(self.input_path):
            os.makedirs(self.input_path)
        if not path.exists(self.output_path):
            os.makedirs(self.output_path)
