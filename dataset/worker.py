# Worker script that generates a single, random example
#
# Adapted from the MOVi-C worker script:
# https://github.com/google-research/kubric/blob/e291953c3604bb8f1817e1d0b062fa2362562613/challenges/movi/movi_c_worker.py

import logging
from pathlib import Path

import bpy
import coloredlogs
import io_utils
import kubric as kb
import kubric.assets
import numpy as np
from kubric.randomness import rotation_sampler
from kubric.renderer import Blender

LOGGER = logging.getLogger(__name__)
coloredlogs.install(fmt="%(asctime)s %(name)s %(levelname)s %(message)s")


# --- CLI arguments
parser = kb.ArgumentParser()

parser.add_argument("--split", choices=["train", "val"], default="train")
parser.add_argument("--floor_friction", type=float, default=0.3)
parser.add_argument("--floor_restitution", type=float, default=0.5)

parser.add_argument("--kubasic_assets", type=str,
                    default="gs://kubric-public/assets/KuBasic/KuBasic.json")
parser.add_argument("--hdri_assets", type=str,
                    default="gs://kubric-public/assets/HDRI_haven/HDRI_haven.json")
parser.add_argument("--gso_assets", type=str,
                    default="gs://kubric-public/assets/GSO/GSO.json")

parser.set_defaults(
    frame_start=1,
    frame_end=90,
    frame_rate=30,
    resolution="256x256",
)

FLAGS = parser.parse_args()


# --- Basic setup
scene, rng, output_path, scratch_dir = kb.setup(FLAGS)
output_path = Path(output_path)
renderer = Blender(scene, scratch_dir, samples_per_pixel=64)
kubasic = kb.AssetSource.from_manifest(FLAGS.kubasic_assets)
gso = kb.AssetSource.from_manifest(FLAGS.gso_assets)
hdri_source = kb.AssetSource.from_manifest(FLAGS.hdri_assets)


# --- Background
LOGGER.info("Setting up the background...")

train_backgrounds, test_backgrounds = hdri_source.get_test_split(fraction=0.1)
if FLAGS.split == "train":
    LOGGER.info(
        "Choosing one of the %d training backgrounds...", len(train_backgrounds)
    )
    hdri_id = rng.choice(train_backgrounds)
else:
    LOGGER.info(
        "Choosing one of the %d held-out backgrounds...", len(test_backgrounds)
    )
    hdri_id = rng.choice(test_backgrounds)
LOGGER.info("Using background %s", hdri_id)
background_hdri = hdri_source.create(asset_id=hdri_id)
scene.metadata["background"] = hdri_id
renderer._set_ambient_light_hdri(background_hdri.filename)

# dome is a half sphere wuth radius 40
dome = kubasic.create(asset_id="dome", name="dome",
                      friction=FLAGS.floor_friction,
                      restitution=FLAGS.floor_restitution,
                      static=True, background=True)
assert isinstance(dome, kb.FileBasedObject)
scene += dome
dome_blender = dome.linked_objects[renderer]
texture_node = dome_blender.data.materials[0].node_tree.nodes["Image Texture"]
texture_node.image = bpy.data.images.load(background_hdri.filename)


# --- Foreground object
LOGGER.info("Setting up the foreground object...")

train_objects, test_objects = gso.get_test_split(fraction=0.1)
if FLAGS.split == "train":
    LOGGER.info("Choosing one of the %d training objects...", len(train_objects))
    object_ = gso.create(asset_id=rng.choice(train_objects))
else:
    LOGGER.info("Choosing one of the %d held-out objects...", len(test_objects))
    object_ = gso.create(asset_id=rng.choice(test_objects))
assert isinstance(object_, kb.FileBasedObject)

scale = rng.uniform(0.75, 3.0)
object_.scale = scale / np.max(object_.bounds[1] - object_.bounds[0])
object_.metadata["scale"] = scale

scene += object_

def sample_object_positions():
    # Sample a random start point around the origin
    start_angle = np.random.random() * 2 * np.pi
    start_distance = np.random.random() * 2
    z = np.random.random() * 4 + 2
    start = [
        np.cos(start_angle) * start_distance,
        np.sin(start_angle) * start_distance,
        z,
    ]

    # Sample a random motion direction
    motion_angle = np.random.random() * 2 * np.pi
    motion_distance = 2.0 + np.random.random() * 2.0 * FLAGS.frame_end / FLAGS.frame_rate  # noqa

    dxs = np.linspace(0.0, np.cos(motion_angle) * motion_distance, FLAGS.frame_end)
    dys = np.linspace(0.0, np.sin(motion_angle) * motion_distance, FLAGS.frame_end)
    xs = start[0] + dxs
    ys = start[1] + dys

    return list((x, y, z) for x, y in zip(xs, ys))

def sample_object_quaternions():
    rotation_sampler()(object_, rng=rng)

    angle_speed = rng.uniform(-np.pi / 4, np.pi / 4)
    angle_distance = angle_speed * FLAGS.frame_end / FLAGS.frame_rate
    angles = object_.quaternion[0] + np.linspace(0.0, angle_distance, FLAGS.frame_end)

    return [(angle, *object_.quaternion[1:]) for angle in angles]


object_positions = sample_object_positions()
object_quaternions = sample_object_quaternions()

# --- Camera
LOGGER.info("Setting up the Camera...")
scene.camera = kb.PerspectiveCamera(focal_length=35., sensor_width=32)

def sample_position_in_sphere(radius: float):
    azimuthal_angle = np.random.uniform(0, 2 * np.pi)
    polar_angle = np.random.uniform(0, np.pi)

    x = np.sin(polar_angle) * np.cos(azimuthal_angle)
    y = np.sin(polar_angle) * np.sin(azimuthal_angle)
    z = np.cos(polar_angle)

    return np.array([x, y, z]) * np.random.uniform(0, radius)

def sample_camera_positions():
    base_offset = np.array(kb.sample_point_in_half_sphere_shell(
        inner_radius=3.0, outer_radius=6.0, offset=2.0
    ))

    start_offset = sample_position_in_sphere(1.0)
    end_offset = sample_position_in_sphere(1.0)

    start = np.array(object_positions[0]) + base_offset + start_offset
    end = np.array(object_positions[-1]) + base_offset + end_offset

    trajectory_length = np.random.uniform(0.25, 1.0)
    trajectory_start = np.random.uniform(0, 1.0 - trajectory_length)
    trajectory_end = trajectory_start + trajectory_length

    start_shortened = start + (end - start) * trajectory_start
    end_shortened = start + (end - start) * trajectory_end

    xs = np.linspace(start_shortened[0], end_shortened[0], FLAGS.frame_end)
    ys = np.linspace(start_shortened[1], end_shortened[1], FLAGS.frame_end)
    zs = np.linspace(start_shortened[2], end_shortened[2], FLAGS.frame_end)

    return list(zip(xs, ys, zs)), trajectory_start, trajectory_end


def sample_camera_look_at_positions(trajectory_start, trajectory_end):
    offset = kb.sample_point_in_half_sphere_shell(
        inner_radius=0.0, outer_radius=1.0, offset=-1.0
    )
    start = np.array(object_positions[0]) + np.array(offset)

    offset = kb.sample_point_in_half_sphere_shell(
        inner_radius=0.0, outer_radius=1.0, offset=-1.0
    )
    end = np.array(object_positions[-1]) + np.array(offset)

    start_shortened = start + (end - start) * trajectory_start
    end_shortened = start + (end - start) * trajectory_end

    xs = np.linspace(start_shortened[0], end_shortened[0], FLAGS.frame_end)
    ys = np.linspace(start_shortened[1], end_shortened[1], FLAGS.frame_end)
    zs = np.linspace(start_shortened[2], end_shortened[2], FLAGS.frame_end)

    return list(zip(xs, ys, zs))

camera_positions, trajectory_start, trajectory_end = sample_camera_positions()
camera_look_at_positions = sample_camera_look_at_positions(
    trajectory_start, trajectory_end
)


# --- Rendering
LOGGER.info("Rendering...")

for frame_index in range(FLAGS.frame_start, FLAGS.frame_end+1):
    object_.position = object_positions[frame_index-1]
    object_.quaternion = object_quaternions[frame_index-1]
    object_.keyframe_insert("position", frame_index)
    object_.keyframe_insert("quaternion", frame_index)

    scene.camera.position = camera_positions[frame_index-1]
    scene.camera.look_at(camera_look_at_positions[frame_index-1])
    scene.camera.keyframe_insert("position", frame_index)
    scene.camera.keyframe_insert("quaternion", frame_index)

data = renderer.render()


# --- Postprocessing
data["rgb"] = data["rgba"][..., :3]

kb.compute_visibility(data["segmentation"], scene.assets)
visible_foreground_assets = [
    asset for asset in scene.foreground_assets
    if np.max(asset.metadata["visibility"]) > 0
]
visible_foreground_assets = sorted(  # sort assets by their visibility
    visible_foreground_assets,
    key=lambda asset: np.sum(asset.metadata["visibility"]),
    reverse=True,
)
data["segmentation"] = kb.adjust_segmentation_idxs(
    data["segmentation"],
    scene.assets,
    visible_foreground_assets,
)
data["segmentation"] = data["segmentation"].astype(np.uint8)

# For consistency with optical flow estimation methods we discard the flow for which
# any rgb frame is missing.
data["forward_flow"] = data["forward_flow"][:-1]
data["backward_flow"] = data["backward_flow"][1:]

# Optical flow channels are ordered as `flow_y, flow_x` in Kubric. Following the
# conventions used in this codebase, we swap the channels to `flow_x, flow_y`.
data["forward_flow"] = data["forward_flow"][..., ::-1]
data["forward_flow"] = data["forward_flow"][..., ::-2]


# --- Save output
LOGGER.info("Saving output data...")
io_utils.save_video(data["rgb"], output_path / "rgb.png.zip")
io_utils.save_segmentation(data["segmentation"], output_path / "segmentation.json")
io_utils.save_flow(data["forward_flow"], output_path / "forward_flow.npz")
io_utils.save_flow(data["backward_flow"], output_path / "backward_flow.npz")

# kb.done() raises an error for me. The following seems to be the only relevant call. 
kubric.assets.ClosableResource.close_all()
