import logging
import os
import textwrap
from pathlib import Path

import click
import coloredlogs
import matplotlib.gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import torch.utils.tensorboard
from benedict import benedict

from .model import MotionSegmentationModel  # noqa: E402

LOGGER = logging.getLogger(__name__)
coloredlogs.install(fmt="%(asctime)s %(name)s %(levelname)s %(message)s")

SCRATCH = Path("/scratch")


@click.command()
@click.option(
    "--config", "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
)
@click.option("--checkpoint", "checkpoint_path", type=click.Path(path_type=Path))
@click.option("--data-path", type=click.Path(path_type=Path))
@click.option("--output-path", type=click.Path(path_type=Path), default="output")
def main(
    config_path: Path, 
    checkpoint_path: Path,
    data_path: Path,
    output_path: Path,
) -> None:
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

    if checkpoint_path.exists():
        LOGGER.info("Restoring checkpoint: %s", checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        global_step = checkpoint["global_step"]
        model.load_state_dict(checkpoint["model"])
        LOGGER.info("global_step = %d", global_step)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    LOGGER.info("Building model finished")

    dataset = shape_identification_dataset(data_path / "test")

    results = []

    LOGGER.info("Evaluating ...")
    for batch_index, batch in enumerate(dataset):
        if model.uses_motion_energy:
            video = batch["stimulus"].to(device, dtype=torch.float32) / 255.0
            B, C, T, H, W = video.shape
            assert B == 1
            input = torch.stack([
                video[0, :, t:t+9] for t in range(T - 8)
            ])
        else:
            video = batch["stimulus"]
            B, C, T, H, W = video.shape
            assert B == 1
            input = torch.stack([
                video[0, :, t:t+2] for t in range(4, T - 4)
            ])
            input = input.expand(-1, 3, -1, -1, -1)

        with torch.no_grad():
            logits, _ = model.forward(input)
            mask = logits > 0
        
        target_mask = batch["target_mask"][0, 4:-4].to(device)
        distractor_mask = batch["distractor_mask"][0, 4:-4].to(device)

        target_iou = iou(mask, target_mask)
        distractor_iou = iou(mask, distractor_mask)

        correct = target_iou > distractor_iou

        row = 0
        result = {
            "index": batch["index"][row].item(),
        }
        result["target_iou"] = target_iou.item()
        result["distractor_iou"] = distractor_iou.item()
        result["correct"] = correct.item()
        result["number_of_informative_dots"] = batch["number_of_informative_dots"][row]
        results.append(result)

        if batch_index == 0:
            visualize(mask, target_mask, distractor_mask, batch_index, output_path)

        if (batch_index + 1) % 50 == 0:
            LOGGER.info("Evaluating (%d done)", (batch_index + 1) * config["batch_size"])  # noqa: E501
            visualize(mask, target_mask, distractor_mask, batch_index, output_path)

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path / "results.csv", index=False)

    LOGGER.info("Evaluating finished")
    accuracy = results_df["correct"].mean()
    target_iou = results_df["target_iou"].mean()
    LOGGER.info("Accuracy: %.3f", accuracy)
    LOGGER.info("IoU: %.3f", target_iou)

    summary = benedict({
        "accuracy": accuracy,
        "iou": target_iou,
    })
    summary.to_yaml(filepath=str(output_path / "summary.yaml"))

    with open(output_path / "done", "w") as file:
        file.write("completed")
    LOGGER.info("Done.")


def visualize(mask, target_mask, distractor_mask, batch_index, output_path):
    B, _, H, W = mask.shape

    num_rows = 3
    num_cols = B

    height, width = H, W
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

    for row in range(num_rows):
        for col in range(num_cols):
            ax = figure.add_subplot(grid[row, col])
            ax.axis("off")

            if row == 0:
                ax.imshow(mask[col, 0].cpu().numpy(), cmap="gray")
                ax.set_title("Mask", fontdict=fontdict)
            elif row == 1:
                ax.imshow(target_mask[col].cpu().numpy(), cmap="gray")
                ax.set_title("Target", fontdict=fontdict)
            elif row == 2:
                ax.imshow(distractor_mask[col].cpu().numpy(), cmap="gray")
                ax.set_title("Distractor", fontdict=fontdict)

    plt.savefig(output_path / f"{batch_index:05d}.png")
    plt.close()


def shape_identification_dataset(path: Path):
    for example_path in sorted(path.iterdir()):
        example_data = torch.load(example_path / "trial.pt")

        choices = torch.from_numpy(np.stack(example_data["choices"]))  # NHW3
        choices_masks = torch.all(choices == 0, dim=-1)                # NHW

        target = torch.from_numpy(example_data["stimulus_clean"])
        target_mask = torch.all(target == 0, dim=-1)

        distractor = torch.from_numpy(example_data["distractor_clean"])
        distractor_mask = torch.all(distractor == 0, dim=-1)

        sample = {
            "index": torch.tensor(int(example_path.name)),
            "number_of_informative_dots": example_data["number_of_informative_dots"],
            "stimulus": torch.from_numpy(example_data["stimulus"]).unsqueeze(0),  # CTHW
            "choices": choices,
            "choices_masks": choices_masks,
            "target": torch.tensor(example_data["target"]),
            "target_mask": target_mask,
            "distractor_mask": distractor_mask,
        }

        # add batch dimension
        for key, value in sample.items():
            if isinstance(value, torch.Tensor):
                sample[key] = value.unsqueeze(0)
            else:
                sample[key] = [value]

        yield sample


def iou(mask1, mask2):
    intersection = torch.sum(mask1 & mask2)
    union = torch.sum(mask1 | mask2)
    return intersection / union


if __name__ == "__main__":
    main()
