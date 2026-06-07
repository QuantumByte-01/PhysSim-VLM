"""
Generate one sample scene for each task type (TTC, stability, trajectory).
Run: python scripts/generate_sample.py
Output: data/generated/{ttc,stability,trajectory}/sample_000001/
"""

import json
import sys
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import mujoco
from simulation.verifier import (
    TTCConfig, StabilityConfig, TrajectoryConfig,
    ObjectConfig, verify_ttc, verify_stability, verify_trajectory,
    _build_ttc_xml, _build_stability_xml, _build_trajectory_xml,
    _set_velocity,
)

OUT_DIR = Path(__file__).parent.parent / "data" / "generated"
WIDTH, HEIGHT = 640, 480


# ---------------------------------------------------------------------------
# Frame rendering helpers
# ---------------------------------------------------------------------------

def render_frames_from_scratch(xml: str, velocities: dict,
                                n_frames: int, frame_interval: float) -> list:
    """
    Build model fresh, set velocities, render n_frames at frame_interval seconds apart.
    velocities: {body_name: [vx, vy, vz]}
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    for body_name, vel in velocities.items():
        _set_velocity(model, data, body_name, vel)

    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    steps_per_frame = max(1, int(frame_interval / model.opt.timestep))
    frames = []

    for f in range(n_frames):
        mujoco.mj_forward(model, data)
        renderer.update_scene(data)
        pixels = renderer.render()
        frames.append(pixels.copy())
        if f < n_frames - 1:
            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)

    renderer.close()
    del data, model
    return frames


def render_single(xml: str) -> np.ndarray:
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    renderer.update_scene(data)
    pixels = renderer.render()
    renderer.close()
    del data, model
    return pixels


def add_label(img_array: np.ndarray, text: str,
              pos: tuple = (10, 10), color=(255, 255, 100)) -> np.ndarray:
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)
    draw.text(pos, text, fill=color)
    return np.array(img)


def save_frames(frames: list, out_dir: Path, prefix: str = "frame") -> None:
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(frames_dir / f"{prefix}_{i:03d}.png")
    print(f" Saved {len(frames)} frames to {frames_dir}")


# ---------------------------------------------------------------------------
# TTC sample
# ---------------------------------------------------------------------------

def generate_ttc_sample(scene_id: str = "ttc_000001") -> None:
    print(f"\n[TTC] Generating {scene_id}...")
    out = OUT_DIR / "ttc" / scene_id
    out.mkdir(parents=True, exist_ok=True)

    cfg = TTCConfig(
        obj1=ObjectConfig(shape="sphere", size=0.12, mass=1.2, color="red",
                          label="Object A",
                          position=[-1.8, 0.0, 0.13],
                          velocity=[2.5, 0.0, 0.0]),
        obj2=ObjectConfig(shape="box", size=0.10, mass=0.9, color="blue",
                          label="Object B",
                          position=[1.5, 0.0, 0.11],
                          velocity=[-0.6, 0.0, 0.0]),
        surface_friction=0.4,
    )

    # Ground truth
    gt = verify_ttc(cfg)
    print(f" TTC = {gt['time_to_collision']}s, collision={gt['collision_occurred']}")

    # Render frames - show 70% of pre-collision timeline
    ttc = gt["time_to_collision"] or 2.0
    show_duration = ttc * 0.70
    n_frames = 8
    frame_interval = show_duration / (n_frames - 1)

    xml = _build_ttc_xml(cfg)
    velocities = {
        "obj1": cfg.obj1.velocity,
        "obj2": cfg.obj2.velocity,
    }
    frames = render_frames_from_scratch(xml, velocities, n_frames, frame_interval)

    # Label frames
    labeled = []
    for i, f in enumerate(frames):
        t = i * frame_interval
        f = add_label(f, f"t={t:.2f}s", (10, 10))
        f = add_label(f, "Object A (red sphere)", (10, HEIGHT - 50))
        f = add_label(f, "Object B (blue box)", (10, HEIGHT - 30))
        labeled.append(f)

    save_frames(labeled, out)

    # Config
    config = {
        "scene_id": scene_id,
        "task_type": "ttc",
        "difficulty": "medium",
        "input_modality": "video",
        "generation_seed": 42,
        "simulator": "mujoco",
        "video": {"n_frames": n_frames, "fps": 10,
                  "duration_s": round(show_duration, 3),
                  "frame_interval_s": round(frame_interval, 3),
                  "show_ratio": 0.70},
        "obj1": {"shape": cfg.obj1.shape, "size": cfg.obj1.size,
                 "mass": cfg.obj1.mass, "color": cfg.obj1.color,
                 "label": cfg.obj1.label, "position": cfg.obj1.position,
                 "velocity": cfg.obj1.velocity},
        "obj2": {"shape": cfg.obj2.shape, "size": cfg.obj2.size,
                 "mass": cfg.obj2.mass, "color": cfg.obj2.color,
                 "label": cfg.obj2.label, "position": cfg.obj2.position,
                 "velocity": cfg.obj2.velocity},
        "surface_friction": cfg.surface_friction,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    (out / "ground_truth.json").write_text(json.dumps(gt, indent=2))

    prompt = f"""You are watching a short video clip of two objects moving in a physics scene.
The clip shows {show_duration:.1f} seconds of footage at 10 frames per second.

Object A is the red sphere.
Object B is the blue box.

By observing how the objects move across frames, estimate their speeds and predict:
how many seconds from the START of the video will these two objects collide?

<reasoning>Describe the motion you observe and your calculation</reasoning>
<answer>X.XX</answer>"""
    (out / "prompt.txt").write_text(prompt)

    print(f" Files saved to {out}")
    print(f" Answer key: {gt['time_to_collision']}s")


# ---------------------------------------------------------------------------
# Stability sample
# ---------------------------------------------------------------------------

def generate_stability_sample(scene_id: str = "stability_000001") -> None:
    print(f"\n[STABILITY] Generating {scene_id}...")
    out = OUT_DIR / "stability" / scene_id
    out.mkdir(parents=True, exist_ok=True)

    cfg = StabilityConfig(
        objects=[
            ObjectConfig(shape="box", size=0.22, height=0.05, mass=2.0, color="blue",
                         label="Base Block", position=[0.0, 0.0, 0.05]),
            ObjectConfig(shape="cylinder", size=0.07, height=0.18, mass=0.5, color="green",
                         label="Middle Cylinder", position=[0.06, 0.0, 0.23]),
            ObjectConfig(shape="sphere", size=0.06, mass=0.3, color="red",
                         label="Top Sphere", position=[0.10, 0.02, 0.43]),
        ],
        surface_friction=0.7,
    )

    gt = verify_stability(cfg)
    print(f" Stable={gt['is_stable']}, max_disp={gt['max_displacement_m']}m")

    # Single image render
    xml = _build_stability_xml(cfg)
    frame = render_single(xml)
    frame = add_label(frame, "Base Block (blue)", (10, HEIGHT - 70))
    frame = add_label(frame, "Middle Cylinder (green)", (10, HEIGHT - 50))
    frame = add_label(frame, "Top Sphere (red)", (10, HEIGHT - 30))
    Image.fromarray(frame).save(out / "scene.png")
    print(f" Saved scene.png to {out}")

    config = {
        "scene_id": scene_id,
        "task_type": "stability",
        "difficulty": "hard",
        "input_modality": "image",
        "generation_seed": 42,
        "simulator": "mujoco",
        "objects": [
            {"shape": o.shape, "size": o.size, "height": o.height,
             "mass": o.mass, "color": o.color, "label": o.label,
             "position": o.position}
            for o in cfg.objects
        ],
        "surface_friction": cfg.surface_friction,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    (out / "ground_truth.json").write_text(json.dumps(gt, indent=2))

    answer = "stable" if gt["is_stable"] else "unstable"
    prompt = """Analyze this arrangement of objects. Will the stack remain stable, or will it topple/collapse?

The scene contains:
- Base Block (blue, bottom)
- Middle Cylinder (green, middle)
- Top Sphere (red, top)

Consider the sizes, shapes, and how they are balanced. Is the center of mass supported?

<reasoning>Your stability analysis</reasoning>
<answer>stable OR unstable</answer>
<confidence>XX%</confidence>"""
    (out / "prompt.txt").write_text(prompt)

    print(f" Files saved to {out}")
    print(f" Answer key: {answer}")


# ---------------------------------------------------------------------------
# Trajectory sample
# ---------------------------------------------------------------------------

def generate_trajectory_sample(scene_id: str = "trajectory_000001") -> None:
    print(f"\n[TRAJECTORY] Generating {scene_id}...")
    out = OUT_DIR / "trajectory" / scene_id
    out.mkdir(parents=True, exist_ok=True)

    cfg = TrajectoryConfig(
        obj=ObjectConfig(shape="sphere", size=0.06, mass=0.3, color="red",
                         label="Ball",
                         position=[0.0, 0.0, 0.5],
                         velocity=[2.5, 0.0, 3.5]),
        surface_friction=0.5,
        restitution=0.3,
    )

    gt = verify_trajectory(cfg)
    print(f" Landing x={gt['landing_position']['x']}m, y={gt['landing_position']['y']}m")
    print(f" Flight time={gt['flight_time_s']}s, bounces={gt['n_bounces']}")

    # Show 40% of flight time
    flight_time = gt["flight_time_s"] or 1.0
    show_duration = flight_time * 0.40
    show_duration = max(0.3, min(1.5, show_duration))
    n_frames = 5
    frame_interval = show_duration / (n_frames - 1)

    xml = _build_trajectory_xml(cfg)
    velocities = {"obj": cfg.obj.velocity}
    frames = render_frames_from_scratch(xml, velocities, n_frames, frame_interval)

    labeled = []
    for i, f in enumerate(frames):
        t = i * frame_interval
        f = add_label(f, f"t={t:.2f}s", (10, 10))
        f = add_label(f, "Ball (red sphere) - predict landing position", (10, HEIGHT - 30))
        labeled.append(f)

    save_frames(labeled, out)

    config = {
        "scene_id": scene_id,
        "task_type": "trajectory",
        "difficulty": "easy",
        "input_modality": "video",
        "generation_seed": 42,
        "simulator": "mujoco",
        "video": {"n_frames": n_frames, "fps": 10,
                  "duration_s": round(show_duration, 3),
                  "frame_interval_s": round(frame_interval, 3),
                  "show_ratio": 0.40},
        "object": {"shape": cfg.obj.shape, "size": cfg.obj.size,
                   "mass": cfg.obj.mass, "color": cfg.obj.color,
                   "label": cfg.obj.label, "position": cfg.obj.position,
                   "velocity": cfg.obj.velocity},
        "surface_friction": cfg.surface_friction,
        "restitution": cfg.restitution,
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    (out / "ground_truth.json").write_text(json.dumps(gt, indent=2))

    prompt = f"""This video shows the first {show_duration:.1f} seconds of a projectile's flight (5 frames at 10 fps).

The red ball has been launched and you can observe its initial trajectory.

Based on the motion you observe, predict where the ball will come to rest after it lands and stops bouncing.
Give coordinates relative to the launch point:
  x = forward distance (meters)
  y = sideways distance (meters)

<reasoning>Analyze the trajectory from the video frames</reasoning>
<answer>x=X.XX, y=X.XX</answer>"""
    (out / "prompt.txt").write_text(prompt)

    print(f" Files saved to {out}")
    print(f" Answer key: x={gt['landing_position']['x']}, y={gt['landing_position']['y']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating one sample per task type...")
    generate_ttc_sample()
    generate_stability_sample()
    generate_trajectory_sample()
    print("\nDone. Check: data/generated/{ttc,stability,trajectory}/")
