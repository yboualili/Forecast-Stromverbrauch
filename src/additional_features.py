"""
Enriches preliminary dataframe with additional features.

See respective notebooks to learn about preliminary tests.
"""


import logging
from pathlib import Path

import click

from data import const
from data.bda import BDA
from data.featureprocessor import FeatureProcessor


@click.command()
@click.argument("input_filepath", type=click.Path())
@click.argument("output_filepath", type=click.Path())
@click.option(
    "--impute/--no-impute",
    default=True,
    help="apply greedy imputation of missing data.",
)
@click.option(
    "--compensate-outliers/--no-compensate-outliers",
    default=False,
    help="apply statistical outlier compensation.",
)
@click.option(
    "--pv/--no_pv",
    default=False,
    help="Apply feature processing on PV or NO_PV data.",
)
def main(
    input_filepath: click.Path,
    output_filepath: click.Path,
    impute: bool,
    compensate_outliers: bool,
    pv: bool,
) -> None:  # pragma: no cover
    """
    Add additional features to final DataFrame ready to be analyzed.

    Args:
        input_filepath (click.Path): input folder
        output_filepath (click.Path): output folder
        impute (bool): Impute missing data.
        compensate_outliers (bool): compensate outliers (statistical)
        pv (bool): pv or no pv data
    """
    logger = logging.getLogger(__name__)
    logger.info(const.WELCOME)
    logger.info("starting adding features...")

    bda = BDA(input_filepath, output_filepath)

    featureprocessor = FeatureProcessor(bda)
    featureprocessor.transform(
        impute=impute, compensate_outliers=compensate_outliers, pv=pv
    )
    logger.info("All done! ✨ 🍰 ✨")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
