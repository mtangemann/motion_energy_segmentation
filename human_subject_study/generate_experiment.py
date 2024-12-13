#!/usr/bin/env python

import logging
from itertools import chain, islice
from pathlib import Path
from typing import Iterable

import click
import coloredlogs
from benedict import benedict

from dataset import (
    RandomDotShapeIdentificationDataset,
    RandomDotShapeIdentificationSample,
)
from experiment import generate_experiment
from stream import (
    EasyStream,
    IndependentUniformStream,
    UniformNumberOfInformativeDotsStream,
)

LOGGER = logging.getLogger(__name__)
coloredlogs.install(fmt="%(asctime)s %(name)s %(levelname)s %(message)s")


@click.command()
@click.option(
    "--config", "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="config.yaml",
)
@click.option("--output-path", type=click.Path(path_type=Path), default="output")
def main(config_path: Path, output_path: Path) -> None:
    if Path(output_path / "done").exists():
        LOGGER.info("Job is done already.")
        return

    LOGGER.info("Creating dataset ...")
    config = benedict(config_path)
    defaults = config.get("dataset.defaults", dict())
    train_samples = _create_dataset(config["dataset.train"], defaults)
    test_samples = _create_dataset(config["dataset.test"], defaults)
    if "dataset.catch" in config:
        catch_samples = _create_dataset(config["dataset.catch"], defaults)
    else:
        catch_samples = []

    generate_experiment(output_path, train_samples, test_samples, catch_samples)

    with open(output_path / "done", "w") as done_file:
        done_file.write("completed")
    LOGGER.info("Done.")


def _create_dataset(
    config: dict, defaults: dict
) -> Iterable[RandomDotShapeIdentificationSample]:
    if isinstance(config, list):
        datasets = [
            _create_dataset(subset_config, defaults) for subset_config in config
        ]
        return chain(*datasets)

    config = {**defaults, **config}

    dataset_type = config.pop("type", "RandomDotShapeIdentificationDataset")

    if dataset_type == "RandomDotShapeIdentificationDataset":
        return RandomDotShapeIdentificationDataset(**config)

    if dataset_type == "IndependentUniformStream":
        number_of_samples = config.pop("number_of_samples", None)
        stream = IndependentUniformStream(**config)
        if number_of_samples is not None:
            return islice(stream, number_of_samples)
        else:
            return stream

    if dataset_type == "UniformNumberOfInformativeDotsStream":
        proposal_stream = _create_dataset(config.pop("proposal_stream"), defaults)
        number_of_samples = config.pop("number_of_samples", None)
        stream = UniformNumberOfInformativeDotsStream(proposal_stream, **config)
        if number_of_samples is not None:
            return islice(stream, number_of_samples)
        else:
            return stream
    
    if dataset_type == "EasyStream":
        proposal_stream = _create_dataset(config.pop("proposal_stream"), defaults)
        number_of_samples = config.pop("number_of_samples", None)
        stream = EasyStream(proposal_stream, **config)
        if number_of_samples is not None:
            return islice(stream, number_of_samples)
        else:
            return stream
    
    raise ValueError(f"Unknown dataset type: {dataset_type}")


if __name__ == "__main__":
    main()
