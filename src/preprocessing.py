"""
Preprocesses raw data into final DataFrame ready to be analyzed.

Created from notebook preprocessing.ipynb, outlier_detection.ipynb,
and exploration.ipynb.
"""


import logging
from pathlib import Path

import click

from data import const
from data.bda import BDA
from data.dataprocessor import DataProcessor


@click.command()
@click.argument("input_filepath", type=click.Path())
@click.argument("output_filepath", type=click.Path())
@click.option(
    "--pv/--no_pv",
    default=False,
    help="Export only households with or without a PV system.",
)
def main(
    input_filepath: click.Path, output_filepath: click.Path, pv: bool
) -> None:  # pragma: no cover
    """
    Proprocesses raw data into preliminary DataFrame.

    Args:
        input_filepath (click.Path): input folder
        output_filepath (click.Path): output folder
        pv (bool): determines whether households with/without PV
        system are considered
    """
    logger = logging.getLogger(__name__)
    logger.info(const.WELCOME)
    logger.info("starting pre-processing...")

    bda = BDA(input_filepath, output_filepath)

    dataprocessor = DataProcessor(bda)
    dataprocessor.transform(pv)
    logger.info("All done! ✨ 🍰 ✨")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
