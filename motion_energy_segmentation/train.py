"""Training script."""

import logging
import os
import shutil
import textwrap
from copy import deepcopy
from pathlib import Path
from typing import Literal

import click
import coloredlogs
import matplotlib.gridspec
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.utils.data
import torch.utils.tensorboard
from benedict import benedict
from torchvision.utils import flow_to_image

from . import io_utils
from .data_loader import ClipLoader, build_data_loader
from .evaluators import BinarySegmentationEvaluator
from .model import MotionSegmentationModel
from .transforms import object_kinematogram

LOGGER = logging.getLogger(__name__)
coloredlogs.install(fmt="%(asctime)s %(name)s %(levelname)s %(message)s")

SCRATCH = Path("/scratch")


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

    os.makedirs(output_path, exist_ok=True)

    config = benedict.from_yaml(config_path)

    torch.manual_seed(config["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    LOGGER.info("device = %s", device)

    LOGGER.info("Building model ...")
    model = MotionSegmentationModel.from_config(config["model"]).to(device)
    LOGGER.info(model)

    num_params = sum(p.numel() for p in model.parameters())
    num_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    parameters = "\n".join(
        f"{name}: shape={tuple(p.shape)}, trainable={p.requires_grad}"
        for name, p in model.named_parameters()
    )
    LOGGER.info(
        "Parameters: %d (trainable: %d)\n%s",
        num_params, num_params_trainable, textwrap.indent(parameters, "  "),
    )
    benedict({
        "num_params": num_params,
        "num_params_trainable": num_params_trainable,
    }).to_json(filepath=str(output_path / "model_info.json"))

    LOGGER.info("Building model finished")

    if config.get("copy_datasets_to_scratch", False):
        LOGGER.info("Copying datasets to scratch directory...")
        for subset in config["data"].keys():
            key = f"data.{subset}.dataset.path"
            config[key] = copy_dataset_to_scratch(config[key])
        LOGGER.info("Copying datasets to scratch directory finished")

    LOGGER.info("Building data loaders ...")
    data_train = build_data_loader(config["data.train"])
    LOGGER.info("train examples: %d", data_train.num_examples)
    data_val_clean = build_data_loader(config["data.val"])
    LOGGER.info("val examples: %d", data_val_clean.num_examples)
    LOGGER.info("Building data loaders finished")

    data_val_object_kinematogram = create_object_kinematograms(config["data.val"])
    data_val = {
        "clean": data_val_clean,
        "object_kinematogram": data_val_object_kinematogram,
    }

    optimizer = torch.optim.Adam(model.parameters(), lr=config["optimizer.lr"])

    max_steps = config["max_steps"]
    global_step = 0

    log_every = config["log_every"]
    log_at = config.get("log_at", [])
    evaluate_every = config["evaluate_every"]
    evaluate_at = config.get("evaluate_at", [])
    visualize_every = config["visualize_every"]
    visualize_at = config.get("visualize_at", [])
    checkpoint_every = config["checkpoint_every"]
    checkpoint_at = config.get("checkpoint_at", [])

    checkpoint_path = output_path / "checkpoint.pth"
    if checkpoint_path.exists():
        LOGGER.info("Restoring checkpoint: %s", checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        global_step = checkpoint["global_step"]
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        LOGGER.info("global_step = %d", global_step)

    writer = torch.utils.tensorboard.SummaryWriter(output_path)

    if global_step == 0:
        for condition in ["clean", "object_kinematogram"]:
            evaluate(
                model,
                data_val[condition],
                condition,
                device,
                output_path / "00000",
                writer,
                global_step,
            )

    LOGGER.info("Training ...")
    for batch in data_train:
        if model.uses_motion_energy:
            input = batch["rgb"].to(device, dtype=torch.float32) / 255.0  # B C T H W
            input = torch.sum(
                input * torch.tensor([0.07, 0.72, 0.21]).to(device).view(1, 3, 1, 1, 1),
                dim=1, keepdim=True,
            )
            target = batch["segmentation"][:, :, 4].to(device, dtype=torch.float32)
        else:
            # All preprocessing and transfer to device is done by mmflow
            input = batch["rgb"]
            target = batch["segmentation"][:, :, 0].to(device, dtype=torch.float32)

        logits, flow = model.forward(input)

        loss = F.binary_cross_entropy_with_logits(logits, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        global_step += 1

        if global_step % log_every == 0 or global_step in log_at:
            writer.add_scalar("loss", loss, global_step=global_step)

            writer.add_scalar(
                "parameters/learning_rate",
                optimizer.param_groups[-1]["lr"],
                global_step,
            )

            LOGGER.info(
                "Training ... (%d/%d, loss=%.3f)",
                global_step, max_steps, loss.item(),
            )


        if global_step % visualize_every == 0 or global_step in visualize_at:
            with torch.no_grad():
                prediction = logits.sigmoid()

            visualize(
                batch["rgb"],
                target,
                prediction,
                flow,
                output_path / f"{global_step:05d}" / "predictions_train.png",
                writer,
                global_step,
                "train",
            )

        if global_step % evaluate_every == 0 or global_step in evaluate_at:
            for condition in ["clean", "object_kinematogram"]:
                evaluate(
                    model,
                    data_val[condition],
                    condition,
                    device,
                    output_path / f"{global_step:05d}",
                    writer,
                    global_step,
                )

        if global_step % checkpoint_every == 0 or global_step in checkpoint_at:
            LOGGER.info("Saving checkpoint ...")
            checkpoint = {
                "global_step": global_step,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(checkpoint, checkpoint_path)
            LOGGER.info("Saving checkpoint finished")

        if global_step >= max_steps:
            break

    LOGGER.info("Done")
    with open(output_path / "done", "w") as done_marker:
        done_marker.write("completed\n")


def evaluate(
    model: MotionSegmentationModel,
    data: ClipLoader,
    condition: Literal["clean", "object_kinematogram"],
    device: str,
    output_path: Path,
    writer: torch.utils.tensorboard.SummaryWriter,
    global_step: int,
) -> None:
    LOGGER.info(f"Evaluating {condition} ...")
    os.makedirs(output_path, exist_ok=True)

    model.segmenter.eval()

    evaluator = BinarySegmentationEvaluator()

    for batch_index, batch in enumerate(data):
        metadata = None #[{ "id": id } for id in batch["id"]]

        input_rgb = batch["rgb"]

        if model.uses_motion_energy:
            input = batch["rgb"].to(device, dtype=torch.float32) / 255.0  # B C T H W
            input = torch.sum(
                input * torch.tensor([0.07, 0.72, 0.21]).to(device).view(1, 3, 1, 1, 1),
                dim=1, keepdim=True,
            )
            target = batch["segmentation"][:, :, 4].to(device, dtype=torch.float32)
        else:
            # All preprocessing and transfer to device is done by mmflow
            input = batch["rgb"]
            target = batch["segmentation"][:, :, 0].to(device, dtype=torch.float32)

        with torch.no_grad():
            logits, flow = model.forward(input)
            prediction = logits.sigmoid()

        evaluator.append(prediction, target, metadata)

        if batch_index == 0:
            visualize(
                input_rgb,
                target,
                prediction,
                flow,
                output_path / f"predictions_val_{condition}.png",
                writer,
                global_step,
                f"val/{condition}",
            )

        if (batch_index + 1) % 100 == 0:
            LOGGER.info(
                f"Evaluating {condition} ... ({batch_index+1}/{data.num_batches})"
            )

    evaluator.results().to_csv(output_path / f"results_{condition}.csv", index=False)
    summary = evaluator.summary()
    summary.to_yaml(filepath=str(output_path / f"summary_{condition}.yaml"))
 
    for metric, value in summary.items():
        writer.add_scalar(f"val/{condition}/{metric}", value, global_step=global_step)

    LOGGER.info(f"Evaluating {condition} finished")
    LOGGER.info(summary)

    model.segmenter.train()


def visualize(
    inputs, targets, prediction, flow, output_path, writer, global_step, tag,
) -> None:
    num_rows = inputs.shape[0]
    num_cols = inputs.shape[2] + 3 + (1 if flow is not None else 0)

    height, width = inputs.shape[-2:]
    cell_height = 1
    cell_width = cell_height * width / height

    figure = plt.figure(figsize=(num_cols * cell_width, num_rows * cell_height))
    padding = 0.1
    top = 1.0 - 0.25 / (num_rows * (1 + padding))
    grid = matplotlib.gridspec.GridSpec(
        num_rows, num_cols, figure,
        left=0.005, right=0.995, bottom=0.005, top=top,
        hspace=0.5 * padding, wspace=padding,
    )

    fontdict = { "size": 12 }

    inputs = inputs.cpu()
    targets = targets.squeeze(1).cpu()

    for row in range(num_rows):
        for frame_index, frame in enumerate(inputs[row].movedim(0, -1)):
            ax = figure.add_subplot(grid[row, frame_index])
            ax.imshow(frame)
            ax.set_axis_off()
            if row == 0:
                ax.set_title(f"rgb[{frame_index}]", fontdict=fontdict)

        if flow is not None:
            flow_rgb = flow_to_image(flow[row]).cpu().movedim(0, -1)
            ax = figure.add_subplot(grid[row, -4])
            ax.imshow(flow_rgb)
            ax.set_axis_off()
            if row == 0:
                ax.set_title("flow", fontdict=fontdict)

        ax = figure.add_subplot(grid[row, -3])
        ax.matshow(targets[row], interpolation="nearest", vmin=0, vmax=1)
        ax.set_axis_off()
        if row == 0:
            ax.set_title("target", fontdict=fontdict)

        ax = figure.add_subplot(grid[row, -2])
        prediction_cpu = prediction[row].squeeze(0).cpu()
        ax.matshow(prediction_cpu, interpolation="nearest", vmin=0, vmax=1)
        ax.set_axis_off()
        if row == 0:
            ax.set_title("prediction", fontdict=fontdict)

        ax = figure.add_subplot(grid[row, -1])
        ax.matshow(prediction_cpu > 0.5, interpolation="nearest", vmin=0, vmax=1)
        ax.set_axis_off()
        if row == 0:
            ax.set_title("prediction (bool)", fontdict=fontdict)

    os.makedirs(output_path.parent, exist_ok=True)
    plt.savefig(output_path)

    writer.add_figure(f"{tag}/predictions", figure, global_step)


def copy_dataset_to_scratch(path: str) -> str:
    destination = SCRATCH / path
    if destination.exists():
        LOGGER.info("Skipping: %s (exists)", path)
    else:
        shutil.copytree(path, destination)
    return str(destination)


def create_object_kinematograms(config: dict) -> ClipLoader:
    LOGGER.info("Creating object kinematograms ...")

    input_path = Path(config["dataset.path"])
    output_path = SCRATCH / f"{input_path.name}_object_kinematograms"
    subset = config["dataset.subset"]

    if output_path.exists():
        LOGGER.info("Skipping: %s (exists)", output_path)

    else:
        example_paths = sorted((input_path / subset).iterdir())

        for example_index, example_path in enumerate(example_paths):
            forward_flow = io_utils.load_flow(
                example_path / "forward_flow.npz",
                format="torch",
            )
            rgb = object_kinematogram(
                forward_flow.movedim(0, 1), num_dots=500, lifetime=8
            ).movedim(1, 0)
            rgb = rgb.expand(3, -1, -1, -1)

            example_output_path = output_path / subset / example_path.name
            os.makedirs(example_output_path, exist_ok=True)

            io_utils.save_video(rgb, example_output_path / "rgb.png.zip")
            os.symlink(
                example_path / "segmentation.json",
                example_output_path / "segmentation.json",
            )

            if (example_index + 1) % 20 == 0:
                LOGGER.info(
                    "Creating object kinematograms ... (%d/%d)",
                    example_index+1, len(example_paths)
                )

    config = deepcopy(config)
    config["dataset.path"] = output_path
    data_loader = build_data_loader(config)

    LOGGER.info("Creating object kinematograms finished")

    return data_loader


if __name__ == "__main__":
    main()
