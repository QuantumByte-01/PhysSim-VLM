#!/usr/bin/env python3
"""
PhysSim-VLM: Training Data Generator
===========================================
Generates physics scenes for GRPO training (5 000 per task by default).

Tasks:
  - ttc Time-to-Collision: 8-frame video, objects approaching
  - stability Stability Assessment: single image of stacked objects
  - trajectory Trajectory Prediction: 5-frame video of projectile

Input modality (from docs):
  TTC → 8 frames showing 60-80% of pre-collision timeline
  Stability → single frame after brief settling phase
  Trajectory → 4-6 frames showing 30-50% of flight

Usage:
  # Full generation
  python scripts/generate_training_data.py

  # One task, small run
  python scripts/generate_training_data.py --task ttc --count 200

  # Verify rendering looks correct before full run
  python scripts/generate_training_data.py --test_render

  # Inspect existing data
  python scripts/generate_training_data.py --inspect_only --inspect 5
"""

# ── Rendering backend (must be set before mujoco import) ──────────────────────
import os
import platform
if platform.system() != "Windows":
    os.environ["MUJOCO_GL"] = "osmesa" # headless Linux (MI300X)

import json
import math
import random
import argparse
import sys
import time
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw
import mujoco
from tqdm import tqdm

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "generated"
sys.path.insert(0, str(ROOT))

from simulation.verifier import (
    TTCConfig, StabilityConfig, TrajectoryConfig, ObjectConfig,
    verify_ttc, verify_stability, verify_trajectory,
    _set_velocity, _get_position,
)

# ── Visual constants ───────────────────────────────────────────────────────────
SHAPES = ["sphere", "box", "cylinder"]
COLORS = ["red", "blue", "green", "yellow", "orange", "purple", "white"]

COLOR_RGBA = {
    "red": "0.9 0.2 0.2 1",
    "blue": "0.2 0.4 0.9 1",
    "green": "0.2 0.8 0.3 1",
    "yellow": "0.9 0.8 0.1 1",
    "orange": "0.9 0.5 0.1 1",
    "purple": "0.6 0.2 0.8 1",
    "white": "0.9 0.9 0.9 1",
    "gray": "0.5 0.5 0.5 1",
}

W, H = 640, 480 # render resolution


# ══════════════════════════════════════════════════════════════════════════════
# 1. PARAMETER SAMPLERS
# ══════════════════════════════════════════════════════════════════════════════

def sample_difficulty() -> str:
    r = random.random()
    if r < 0.35: return "easy"
    if r < 0.75: return "medium"
    return "hard"


def _color(exclude: Optional[str] = None) -> str:
    choices = [c for c in COLORS if c != exclude]
    return random.choice(choices)

def _size(lo=0.07, hi=0.20) -> float:
    return round(random.uniform(lo, hi), 3)

def _mass(lo=0.3, hi=2.5) -> float:
    return round(random.uniform(lo, hi), 2)

def _shape() -> str:
    return random.choice(SHAPES)

def _height(shape: str, size: float) -> float:
    if shape == "sphere":
        return 0.0
    return round(random.uniform(0.04, 0.14), 3)

def _z_floor(shape: str, size: float, height: float) -> float:
    """Bottom clearance above floor so object rests on it."""
    half = size if shape == "sphere" else height
    return round(half + 0.003, 4)


# ── TTC ───────────────────────────────────────────────────────────────────────
PARAMS_TTC = {
    "easy": dict(v1=(0.7, 1.8), v2=(0.2, 0.8), sep=(1.2, 2.5), yoff=0.02),
    "medium": dict(v1=(1.5, 3.5), v2=(0.2, 1.5), sep=(1.5, 3.5), yoff=0.20),
    "hard": dict(v1=(2.5, 5.5), v2=(0.3, 2.5), sep=(2.0, 4.5), yoff=0.35),
}

def sample_ttc_config(diff: str) -> dict:
    p = PARAMS_TTC[diff]
    c1 = _color(); c2 = _color(c1)
    s1 = _shape(); s2 = _shape()
    sz1 = _size(); sz2 = _size()
    h1 = _height(s1, sz1); h2 = _height(s2, sz2)
    z1 = _z_floor(s1, sz1, h1); z2 = _z_floor(s2, sz2, h2)

    sep = round(random.uniform(*p["sep"]), 2)
    yoff = round(random.uniform(-p["yoff"], p["yoff"]), 3)
    v1 = round(random.uniform(*p["v1"]), 2)
    v2 = round(random.uniform(*p["v2"]), 2)
    # 15% chance: one object nearly stationary
    if random.random() < 0.15:
        v2 = round(random.uniform(0.0, 0.15), 2)

    friction = round(random.uniform(0.3, 0.8), 2)

    return {
        "task_type": "ttc",
        "difficulty": diff,
        "obj1": {
            "shape": s1, "size": sz1, "height": h1, "mass": _mass(),
            "color": c1, "label": "Object A",
            "position": [round(-sep / 2, 3), 0.0, z1],
            "velocity": [v1, 0.0, 0.0],
        },
        "obj2": {
            "shape": s2, "size": sz2, "height": h2, "mass": _mass(),
            "color": c2, "label": "Object B",
            "position": [round(sep / 2, 3), yoff, z2],
            "velocity": [-v2, 0.0, 0.0],
        },
        "surface_friction": friction,
    }


# ── Stability ─────────────────────────────────────────────────────────────────
PARAMS_STAB = {
    "easy": dict(n=(2, 2), off_max=0.04, ratio=(0.65, 0.90)),
    "medium": dict(n=(2, 3), off_max=0.12, ratio=(0.45, 0.80)),
    "hard": dict(n=(3, 4), off_max=0.28, ratio=(0.30, 0.70)),
}

def sample_stability_config(diff: str) -> dict:
    p = PARAMS_STAB[diff]
    n = random.randint(*p["n"])
    friction = round(random.uniform(0.5, 0.9), 2)

    base_sz = round(random.uniform(0.14, 0.28), 3)
    base_sh = random.choice(["box", "cylinder"])
    base_h = round(random.uniform(0.04, 0.10), 3)
    base_c = _color()

    objects = [{
        "shape": base_sh, "size": base_sz, "height": base_h,
        "mass": round(random.uniform(1.5, 3.0), 2),
        "color": base_c, "label": "Base",
        "position": [0.0, 0.0, round(base_h + 0.003, 4)],
    }]

    top_z = 2 * base_h + 0.003
    prev_sz = base_sz
    used = [base_c]

    for i in range(1, n):
        ratio = random.uniform(*p["ratio"])
        sz = max(0.04, round(prev_sz * ratio, 3))
        sh = _shape()
        h = _height(sh, sz)
        half = sz if sh == "sphere" else h
        c = _color(used[-1]); used.append(c)

        ox = round(random.uniform(-p["off_max"], p["off_max"]), 3)
        oy = round(random.uniform(-p["off_max"], p["off_max"]), 3)
        z = round(top_z + half + 0.003, 4)

        objects.append({
            "shape": sh, "size": sz, "height": h,
            "mass": round(random.uniform(0.2, 1.2), 2),
            "color": c, "label": f"Layer {i+1}",
            "position": [ox, oy, z],
        })
        top_z = z + half
        prev_sz = sz

    return {
        "task_type": "stability",
        "difficulty": diff,
        "objects": objects,
        "surface_friction": friction,
    }


# ── Trajectory ────────────────────────────────────────────────────────────────
PARAMS_TRAJ = {
    "easy": dict(vx=(1.5, 3.0), vz=(2.0, 4.0), h=(0.5, 1.5), fr=(0.4, 0.7)),
    "medium": dict(vx=(1.0, 4.5), vz=(1.5, 5.5), h=(0.3, 2.0), fr=(0.2, 0.8)),
    "hard": dict(vx=(0.5, 6.0), vz=(0.8, 7.0), h=(0.2, 2.5), fr=(0.1, 0.9)),
}

def sample_trajectory_config(diff: str) -> dict:
    p = PARAMS_TRAJ[diff]
    sh = _shape()
    sz = _size(0.04, 0.10)
    h = _height(sh, sz)
    c = _color()
    lh = round(random.uniform(*p["h"]), 3) # launch height
    vx = round(random.uniform(*p["vx"]), 2)
    vz = round(random.uniform(*p["vz"]), 2)
    vy = round(random.uniform(-0.3, 0.3), 2) # small sideways component
    z0 = lh + (sz if sh == "sphere" else h) + 0.003

    return {
        "task_type": "trajectory",
        "difficulty": diff,
        "object": {
            "shape": sh, "size": sz, "height": h, "mass": _mass(0.1, 1.0),
            "color": c, "label": "Ball",
            "position": [0.0, 0.0, round(z0, 4)],
            "velocity": [vx, vy, vz],
        },
        "surface_friction": round(random.uniform(*p["fr"]), 2),
        "restitution": round(random.uniform(0.10, 0.50), 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2. XML BUILDERS (no named camera - camera set programmatically)
# ══════════════════════════════════════════════════════════════════════════════

def _body_xml(obj: dict, name: str, friction: float) -> str:
    pos = " ".join(str(v) for v in obj["position"])
    rgba = COLOR_RGBA.get(obj["color"], "0.7 0.7 0.7 1")
    s, sh = obj["size"], obj["shape"]
    if sh == "sphere":
        geom = f'type="sphere" size="{s}"'
    elif sh == "cylinder":
        geom = f'type="cylinder" size="{s} {obj.get("height", s)}"'
    else:
        h = obj.get("height", s)
        geom = f'type="box" size="{s} {s} {h}"'
    return (
        f' <body name="{name}" pos="{pos}">\n'
        f' <freejoint/>\n'
        f' <geom {geom} mass="{obj.get("mass", 1.0)}" rgba="{rgba}" '
        f'friction="{friction} 0.005 0.0001"/>\n'
        f' </body>\n'
    )


def _lights_xml() -> str:
    return (
        ' <light pos="0 0 5" dir="0 0 -1" '
        'diffuse="0.85 0.85 0.85" specular="0.05 0.05 0.05"/>\n'
        ' <light pos="-3 -3 4" dir="0.5 0.5 -0.7" '
        'diffuse="0.35 0.35 0.35" specular="0 0 0"/>\n'
    )


def build_ttc_xml(cfg: dict) -> str:
    f = cfg["surface_friction"]
    return (
        f'<mujoco model="ttc">\n'
        f' <option timestep="0.001" gravity="0 0 -9.81"/>\n'
        f' <visual><global offwidth="{W}" offheight="{H}"/></visual>\n'
        f' <worldbody>\n'
        + _lights_xml()
        + f' <geom name="floor" type="plane" size="12 12 0.1" '
          f'rgba="0.80 0.80 0.80 1" friction="{f} 0.005 0.0001"/>\n'
        + _body_xml(cfg["obj1"], "obj1", f)
        + _body_xml(cfg["obj2"], "obj2", f)
        + ' </worldbody>\n</mujoco>'
    )


def build_stability_xml(cfg: dict) -> str:
    f = cfg["surface_friction"]
    bodies = "".join(_body_xml(o, f"obj{i}", f)
                     for i, o in enumerate(cfg["objects"]))
    return (
        f'<mujoco model="stability">\n'
        f' <option timestep="0.001" gravity="0 0 -9.81"/>\n'
        f' <visual><global offwidth="{W}" offheight="{H}"/></visual>\n'
        f' <worldbody>\n'
        + _lights_xml()
        + f' <geom name="floor" type="plane" size="8 8 0.1" '
          f'rgba="0.80 0.80 0.80 1" friction="{f} 0.005 0.0001"/>\n'
        + bodies
        + ' </worldbody>\n</mujoco>'
    )


def build_trajectory_xml(cfg: dict) -> str:
    f = cfg["surface_friction"]
    return (
        f'<mujoco model="trajectory">\n'
        f' <option timestep="0.001" gravity="0 0 -9.81"/>\n'
        f' <visual><global offwidth="{W}" offheight="{H}"/></visual>\n'
        f' <worldbody>\n'
        + _lights_xml()
        + f' <geom name="floor" type="plane" size="20 20 0.1" '
          f'rgba="0.80 0.80 0.80 1" '
          f'friction="{f} 0.005 0.0001" '
          f'solimp="0.9 0.95 0.001" solref="0.02 1"/>\n'
        + _body_xml(cfg["object"], "obj", f)
        + ' </worldbody>\n</mujoco>'
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3. CAMERA SETUP
# ══════════════════════════════════════════════════════════════════════════════

def _free_cam(lookat, distance, azimuth, elevation) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = lookat
    cam.distance = distance
    cam.azimuth = azimuth
    cam.elevation = elevation
    return cam


def ttc_camera(cfg: dict) -> mujoco.MjvCamera:
    """Side view - objects approach along x-axis, camera from +y."""
    x_mid = (cfg["obj1"]["position"][0] + cfg["obj2"]["position"][0]) / 2
    sep = abs(cfg["obj1"]["position"][0] - cfg["obj2"]["position"][0])
    dist = max(5.0, sep * 1.6)
    return _free_cam(
        lookat = [x_mid, 0.0, 0.25],
        distance = dist,
        azimuth = 90.0,
        elevation= -15.0,
    )


def stability_camera(cfg: dict) -> mujoco.MjvCamera:
    """3/4 front-left view looking at the stack."""
    objs = cfg.get("objects", [])
    top_z = max((o["position"][2] for o in objs), default=0.5)
    return _free_cam(
        lookat = [0.0, 0.0, top_z * 0.45],
        distance = 3.2,
        azimuth = 200.0,
        elevation= -22.0,
    )


def trajectory_camera(cfg: dict) -> mujoco.MjvCamera:
    """Side view - projectile moves along x-axis, camera from +y.
    Lookat adapts to the launch velocity so the ball stays in frame."""
    obj = cfg["object"]
    z0 = obj["position"][2]
    vx = obj["velocity"][0]
    vz = obj["velocity"][2]
    # Estimate ball position at t ≈ 0.15s (early in clip)
    t = 0.15
    x_m = vx * t
    z_m = max(0.15, z0 + vz * t - 0.5 * 9.81 * t ** 2)
    dist = max(4.5, z0 * 3.0 + vx * 0.5)
    return _free_cam(
        lookat = [max(0.1, x_m * 0.5), 0.0, z_m * 0.7],
        distance = dist,
        azimuth = 90.0,
        elevation= -10.0,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 4. RENDERING UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def _set_vel(model, data, body_name: str, vel: list):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0: return
    jid = model.body_jntadr[bid]
    if jid < 0: return
    qa = model.jnt_dofadr[jid]
    data.qvel[qa:qa+3] = vel[:3]


def _add_sky(frame: np.ndarray) -> np.ndarray:
    """Replace near-black sky pixels with a blue-gray gradient.
    Works by thresholding: all near-black pixels (sum < 40) become sky.
    Our objects are red/blue/green/yellow/orange/purple/white - none are black."""
    h, w = frame.shape[:2]
    arr = frame.astype(np.float32)

    # Build sky gradient: deep blue-gray at top → lighter near horizon
    sky = np.zeros((h, w, 3), dtype=np.float32)
    for y in range(h):
        t = y / h # 0 = top, 1 = bottom
        sky[y, :, 0] = 78 + 85 * t # R
        sky[y, :, 1] = 100 + 85 * t # G
        sky[y, :, 2] = 140 + 55 * t # B

    mask = (arr[:, :, 0] + arr[:, :, 1] + arr[:, :, 2]) < 40
    arr[mask] = sky[mask]
    return arr.astype(np.uint8)


def _render(renderer, data, cam) -> np.ndarray:
    renderer.update_scene(data, camera=cam)
    return _add_sky(renderer.render())


def add_overlay(frame: np.ndarray, lines: list[str],
                t: Optional[float] = None) -> np.ndarray:
    """Adds info bar at top and optional timestamp at bottom-left."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img, "RGBA")

    bar_h = 22 * len(lines) + 8
    draw.rectangle([(0, 0), (W, bar_h)], fill=(0, 0, 0, 155))
    for i, line in enumerate(lines):
        draw.text((6, 4 + i * 22), line, fill=(255, 255, 255, 220))

    if t is not None:
        draw.rectangle([(0, H - 22), (100, H)], fill=(0, 0, 0, 140))
        draw.text((4, H - 20), f"t = {t:.2f}s", fill=(220, 220, 220, 220))

    return np.array(img)


# ── TTC render ───────────────────────────────────────────────────────────────
def render_ttc_frames(cfg: dict, gt: dict) -> tuple[list[np.ndarray], dict]:
    ttc = gt["time_to_collision"]
    show_ratio = random.uniform(0.60, 0.80)
    duration = max(0.30, min(2.0, min(ttc * show_ratio, ttc - 0.12)))
    n_frames = max(4, min(8, round(duration / 0.1) + 1))
    fps = 10

    xml = build_ttc_xml(cfg)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    _set_vel(model, data, "obj1", cfg["obj1"]["velocity"])
    _set_vel(model, data, "obj2", cfg["obj2"]["velocity"])

    cam = ttc_camera(cfg)
    renderer = mujoco.Renderer(model, height=H, width=W)

    o1, o2 = cfg["obj1"], cfg["obj2"]
    overlay_lines = [
        f"Object A: {o1['color']} {o1['shape']} → Object B: {o2['color']} {o2['shape']} ←",
        f"Clip shows {duration:.2f}s of {ttc:.2f}s total | Predict time of collision",
    ]

    step_interval = max(1, round(duration / (n_frames - 1) / model.opt.timestep))
    frame_dt = step_interval * model.opt.timestep

    frames = []
    for i in range(n_frames):
        if i > 0:
            for _ in range(step_interval):
                mujoco.mj_step(model, data)
        raw = _render(renderer, data, cam)
        frames.append(add_overlay(raw, overlay_lines, t=i * frame_dt))

    renderer.close()
    del data, model

    return frames, {
        "n_frames": n_frames, "fps": fps,
        "duration_s": round(duration, 3),
        "show_ratio": round(show_ratio, 3),
        "time_remaining_s": round(ttc - duration, 3),
    }


# ── Stability render ──────────────────────────────────────────────────────────
def render_stability_frame(cfg: dict) -> np.ndarray:
    xml = build_stability_xml(cfg)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # brief settling so objects rest naturally
    for _ in range(int(0.12 / model.opt.timestep)):
        mujoco.mj_step(model, data)

    cam = stability_camera(cfg)
    renderer = mujoco.Renderer(model, height=H, width=W)

    objs = cfg["objects"]
    desc = " | ".join(f"{o['label']}: {o['color']} {o['shape']}" for o in objs)
    overlay_lines = [f"Stack ({len(objs)} objects): {desc}", "Is this arrangement stable?"]

    raw = _render(renderer, data, cam)
    renderer.close()
    del data, model

    return add_overlay(raw, overlay_lines)


# ── Trajectory render ─────────────────────────────────────────────────────────
def render_trajectory_frames(cfg: dict, gt: dict) -> tuple[list[np.ndarray], dict]:
    flight_t = gt["flight_time_s"]
    show_ratio = random.uniform(0.30, 0.50)
    duration = max(0.30, min(1.5, min(flight_t * show_ratio, flight_t - 0.25)))
    n_frames = max(4, min(6, round(duration / 0.1) + 1))
    fps = 10

    xml = build_trajectory_xml(cfg)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    _set_vel(model, data, "obj", cfg["object"]["velocity"])

    cam = trajectory_camera(cfg)
    renderer = mujoco.Renderer(model, height=H, width=W)

    obj = cfg["object"]
    overlay_lines = [
        f"Projectile: {obj['color']} {obj['shape']}",
        "Predict final resting position (x=forward, y=sideways from launch)",
    ]

    step_interval = max(1, round(duration / (n_frames - 1) / model.opt.timestep))
    frame_dt = step_interval * model.opt.timestep

    frames = []
    for i in range(n_frames):
        if i > 0:
            for _ in range(step_interval):
                mujoco.mj_step(model, data)
        raw = _render(renderer, data, cam)
        frames.append(add_overlay(raw, overlay_lines, t=i * frame_dt))

    renderer.close()
    del data, model

    return frames, {
        "n_frames": n_frames, "fps": fps,
        "duration_s": round(duration, 3),
        "show_ratio": round(show_ratio, 3),
        "time_remaining_s": round(gt["total_time_s"] - duration, 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. GROUND TRUTH via verifier.py
# ══════════════════════════════════════════════════════════════════════════════

def _obj_cfg(d: dict) -> ObjectConfig:
    return ObjectConfig(
        shape=d["shape"], size=d["size"], height=d.get("height", 0.0),
        mass=d.get("mass", 1.0), color=d["color"],
        position=list(d["position"]), velocity=list(d.get("velocity", [0, 0, 0])),
    )

def gt_ttc(cfg: dict) -> dict:
    c = TTCConfig(
        obj1=_obj_cfg(cfg["obj1"]),
        obj2=_obj_cfg(cfg["obj2"]),
        surface_friction=cfg["surface_friction"],
        max_sim_time=10.0,
    )
    return verify_ttc(c)

def gt_stability(cfg: dict) -> dict:
    c = StabilityConfig(
        objects=[_obj_cfg(o) for o in cfg["objects"]],
        surface_friction=cfg["surface_friction"],
        sim_duration=3.0, settling_time=0.1,
        displacement_threshold=0.02,
    )
    return verify_stability(c)

def gt_trajectory(cfg: dict) -> dict:
    c = TrajectoryConfig(
        obj=_obj_cfg(cfg["object"]),
        surface_friction=cfg["surface_friction"],
        restitution=cfg.get("restitution", 0.3),
        max_sim_time=10.0,
    )
    return verify_trajectory(c)


# ══════════════════════════════════════════════════════════════════════════════
# 6. QUALITY CONTROL
# ══════════════════════════════════════════════════════════════════════════════

def qc_ttc(gt: dict) -> bool:
    return gt["collision_occurred"] and 0.40 <= gt["time_to_collision"] <= 8.0

def qc_stability(_: dict) -> bool:
    return True # all stability outcomes are valid

def qc_trajectory(gt: dict) -> bool:
    ft = gt.get("flight_time_s")
    if ft is None or ft < 0.40:
        return False
    lx = gt["landing_position"]["x"]
    ly = gt["landing_position"]["y"]
    return math.sqrt(lx**2 + ly**2) < 15.0


# ══════════════════════════════════════════════════════════════════════════════
# 7. PROMPT TEMPLATES
# ══════════════════════════════════════════════════════════════════════════════

TTC_PROMPTS = [
    """\
You are watching a short video clip of two objects moving in a physics scene.
The clip shows {dur:.1f} seconds of footage at {fps} fps ({nf} frames).

Object A is the {c1} {s1}.
Object B is the {c2} {s2}.

By observing how the objects move across frames, estimate their speeds and predict:
How many seconds from the START of the video will these two objects collide?

If they will not collide within 10 seconds, answer "no collision".

<reasoning>Describe the motion you observe and your calculation</reasoning>
<answer>X.XX</answer>""",

    """\
This video shows two objects over {dur:.1f} seconds ({nf} frames at {fps} fps).

Object A: {c1} {s1} | Object B: {c2} {s2}

Watch carefully:
1. How far does each object move between frames?
2. What direction is each moving?
3. At what rate are they closing the gap?

Predict the total time from the first frame until collision.

<reasoning>Your frame-by-frame analysis</reasoning>
<answer>X.XX</answer>""",

    """\
Two objects are moving in this {dur:.1f}-second video.

Object A: {c1} {s1}
Object B: {c2} {s2}

When will they collide? Give seconds from the start of the video.

<reasoning>Your physics analysis</reasoning>
<answer>X.XX</answer>""",
]

STABILITY_PROMPTS = [
    """\
Analyze this arrangement of stacked objects. Will it remain stable, or topple/collapse?

Objects (bottom to top):
{obj_list}

Consider:
- The shape and size of each object
- How centered each object is on the one below
- Whether the combined center of mass is supported

<reasoning>Your step-by-step stability analysis</reasoning>
<answer>stable OR unstable</answer>
<confidence>XX%</confidence>""",

    """\
Look at this stack of objects. Will it fall over?

{obj_list}

<reasoning>Your stability analysis</reasoning>
<answer>stable OR unstable</answer>
<confidence>XX%</confidence>""",

    """\
This image shows {n} objects arranged in a stack. Predict whether the arrangement will \
remain stable over the next few seconds, or collapse.

Objects (bottom to top): {inline}

<reasoning>Analyze the center of mass and support geometry</reasoning>
<answer>stable OR unstable</answer>
<confidence>XX%</confidence>""",
]

TRAJECTORY_PROMPTS = [
    """\
This video shows the first {dur:.1f} seconds of a projectile's flight ({nf} frames at {fps} fps).

The {c} {s} has been launched - you can see its initial arc in the clip.

Based on the motion pattern:
1. Estimate launch angle from the trajectory curve
2. Estimate speed from frame-to-frame displacement
3. Predict where it will come to rest after landing and stopping

Give coordinates relative to the launch point:
x = forward distance (meters), y = sideways distance (meters)

<reasoning>Your trajectory analysis</reasoning>
<answer>x=X.XX, y=X.XX</answer>""",

    """\
Watch this projectile's initial flight ({dur:.1f}s shown). Based on its trajectory, \
predict the final resting position.

The {c} {s} was launched from the start position.

<reasoning>Analysis of observed motion arc</reasoning>
<answer>x=X.XX, y=X.XX</answer>""",

    """\
You see {nf} frames ({dur:.1f}s) of a {c} {s} in flight.

The object will land and come to rest. From what you observe:
- Estimate horizontal velocity from lateral movement per frame
- Estimate vertical velocity from height change per frame
- Account for gravity (9.81 m/s²) and the landing

Predict landing position (x=forward, y=sideways from launch point):

<reasoning>Step-by-step trajectory calculation</reasoning>
<answer>x=X.XX, y=X.XX</answer>""",
]


def build_ttc_prompt(cfg: dict, video_info: dict) -> str:
    o1, o2 = cfg["obj1"], cfg["obj2"]
    t = random.choice(TTC_PROMPTS)
    return t.format(dur=video_info["duration_s"], fps=video_info["fps"],
                    nf=video_info["n_frames"],
                    c1=o1["color"], s1=o1["shape"],
                    c2=o2["color"], s2=o2["shape"])

def build_stability_prompt(cfg: dict) -> str:
    objs = cfg["objects"]
    obj_list = "\n".join(
        f"- {o['label']}: {o['color']} {o['shape']} (size {o['size']:.2f}m)"
        for o in objs
    )
    inline = ", ".join(f"{o['color']} {o['shape']}" for o in objs)
    t = random.choice(STABILITY_PROMPTS)
    return t.format(n=len(objs), obj_list=obj_list, inline=inline)

def build_trajectory_prompt(cfg: dict, video_info: dict) -> str:
    obj = cfg["object"]
    t = random.choice(TRAJECTORY_PROMPTS)
    return t.format(dur=video_info["duration_s"], fps=video_info["fps"],
                    nf=video_info["n_frames"],
                    c=obj["color"], s=obj["shape"])


# ══════════════════════════════════════════════════════════════════════════════
# 8. SCENE SAVERS
# ══════════════════════════════════════════════════════════════════════════════

def _ts() -> str:
    return datetime.utcnow().isoformat() + "Z"

def save_frames(frames: list[np.ndarray], out_dir: Path):
    fd = out_dir / "frames"
    fd.mkdir(exist_ok=True)
    for i, f in enumerate(frames):
        Image.fromarray(f).save(fd / f"frame_{i:03d}.png")
    Image.fromarray(frames[0]).save(out_dir / "thumbnail.png")

def save_scene(scene_id: str, cfg: dict, gt: dict, prompt: str,
               out_dir: Path, seed: int,
               frames: Optional[list] = None,
               frame_img: Optional[np.ndarray] = None,
               video_info: Optional[dict] = None):
    out_dir.mkdir(parents=True, exist_ok=True)

    config_out = {
        **cfg,
        "scene_id": scene_id,
        "seed": seed,
        "generation_timestamp": _ts(),
    }
    if video_info:
        config_out["input_modality"] = "video"
        config_out["video"] = video_info
        save_frames(frames, out_dir)
    else:
        config_out["input_modality"] = "image"
        Image.fromarray(frame_img).save(out_dir / "scene.png")

    gt_out = {**gt}
    if video_info:
        gt_out["video_info"] = video_info

    with open(out_dir / "config.json", "w") as f: json.dump(config_out, f, indent=2)
    with open(out_dir / "ground_truth.json", "w") as f: json.dump(gt_out, f, indent=2)
    with open(out_dir / "prompt.txt", "w") as f: f.write(prompt)


# ══════════════════════════════════════════════════════════════════════════════
# 9. MAIN GENERATOR LOOP
# ══════════════════════════════════════════════════════════════════════════════

def generate_task(task: str, count: int, seed_offset: int = 0,
                  start_idx: int = 0) -> dict:
    task_dir = DATA_DIR / task
    task_dir.mkdir(parents=True, exist_ok=True)

    stats = defaultdict(int)
    diff_counts = defaultdict(int)
    accepted = start_idx # start scene IDs from this offset
    tried = 0
    stable_cnt = 0 # used only for stability balance tracking

    pbar = tqdm(
        total=count, desc=f" {task:12s}[{start_idx}]", unit="scene",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{eta}, {rate_fmt}]{postfix}",
    )

    while accepted < start_idx + count:
        tried += 1
        seed = seed_offset + tried
        random.seed(seed)
        np.random.seed(seed)

        diff = sample_difficulty()

        try:
            # ── Sample + simulate ──────────────────────────────────────────────
            if task == "ttc":
                cfg = sample_ttc_config(diff)
                g = gt_ttc(cfg)
                if not qc_ttc(g):
                    stats["rej_qc"] += 1
                    continue

            elif task == "stability":
                cfg = sample_stability_config(diff)
                g = gt_stability(cfg)
                if not qc_stability(g):
                    stats["rej_qc"] += 1
                    continue
                # Enforce ~50/50 stable/unstable balance
                is_stab = g["is_stable"]
                ratio = stable_cnt / max(1, accepted - start_idx)
                if is_stab and ratio > 0.60:
                    stats["rej_balance"] += 1
                    continue
                if not is_stab and ratio < 0.40:
                    stats["rej_balance"] += 1
                    continue

            elif task == "trajectory":
                cfg = sample_trajectory_config(diff)
                g = gt_trajectory(cfg)
                if not qc_trajectory(g):
                    stats["rej_qc"] += 1
                    continue

            # ── Scene ID & resume check ────────────────────────────────────────
            scene_id = f"{task}_{accepted:05d}"
            out_dir = task_dir / scene_id

            if out_dir.exists() and (out_dir / "ground_truth.json").exists():
                # Resume: count it but don't re-generate
                if task == "stability":
                    try:
                        cached = json.load(open(out_dir / "ground_truth.json"))
                        if cached.get("is_stable"): stable_cnt += 1
                    except Exception:
                        pass
                accepted += 1
                diff_counts[diff] += 1
                stats["resumed"] += 1
                pbar.update(1)
                continue

            # ── Render ────────────────────────────────────────────────────────
            if task == "ttc":
                frames, vinfo = render_ttc_frames(cfg, g)
                prompt = build_ttc_prompt(cfg, vinfo)
                save_scene(scene_id, cfg, g, prompt, out_dir, seed,
                           frames=frames, video_info=vinfo)

            elif task == "stability":
                frame = render_stability_frame(cfg)
                prompt = build_stability_prompt(cfg)
                save_scene(scene_id, cfg, g, prompt, out_dir, seed,
                           frame_img=frame)
                if g["is_stable"]: stable_cnt += 1

            elif task == "trajectory":
                frames, vinfo = render_trajectory_frames(cfg, g)
                prompt = build_trajectory_prompt(cfg, vinfo)
                save_scene(scene_id, cfg, g, prompt, out_dir, seed,
                           frames=frames, video_info=vinfo)

        except Exception as e:
            stats["errors"] += 1
            if stats["errors"] <= 5:
                tqdm.write(f" [ERR] seed={seed} {e}")
            continue

        accepted += 1
        diff_counts[diff] += 1
        stats["accepted"] += 1

        # ── Update progress bar postfix ────────────────────────────────────────
        rate = (accepted - start_idx) / tried * 100
        post = f"rate={rate:.0f}% rej_qc={stats['rej_qc']}"
        if task == "stability":
            post += f" stable={stable_cnt}/{accepted - start_idx}"
        post += (f" | easy={diff_counts['easy']} "
                 f"med={diff_counts['medium']} "
                 f"hard={diff_counts['hard']}")
        pbar.set_postfix_str(post)
        pbar.update(1)

    pbar.close()
    return {
        "task": task,
        "count": accepted - start_idx,
        "tried": tried,
        "acceptance_rate": round((accepted - start_idx) / max(tried, 1) * 100, 1),
        "difficulty": dict(diff_counts),
        **{k: v for k, v in stats.items() if k != "accepted"},
    }


# ══════════════════════════════════════════════════════════════════════════════
# 10. TRAIN / VAL / TEST SPLITS
# ══════════════════════════════════════════════════════════════════════════════

def create_splits(tasks: list[str]):
    splits: dict[str, list] = {"train": [], "val": [], "test": []}

    for task in tasks:
        td = DATA_DIR / task
        scenes = sorted([s for s in td.iterdir()
                         if s.is_dir() and (s / "config.json").exists()])
        random.shuffle(scenes)
        n = len(scenes)
        n_val = max(1, int(n * 0.10))
        n_test = max(1, int(n * 0.10))
        n_tr = n - n_val - n_test

        for s in scenes[:n_tr]:
            splits["train"].append({"task": task, "scene_id": s.name, "path": str(s)})
        for s in scenes[n_tr:n_tr + n_val]:
            splits["val"].append({"task": task, "scene_id": s.name, "path": str(s)})
        for s in scenes[n_tr + n_val:]:
            splits["test"].append({"task": task, "scene_id": s.name, "path": str(s)})

    for name, items in splits.items():
        p = DATA_DIR / f"{name}.json"
        with open(p, "w") as f:
            json.dump(items, f, indent=2)
        print(f" {name:6s}: {len(items):5d} scenes → {p}")


# ══════════════════════════════════════════════════════════════════════════════
# 11. DATASET STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def print_stats(tasks: list[str]):
    print(f"\n{'═' * 62}")
    print(" DATASET STATISTICS")
    print(f"{'═' * 62}")

    total_n, total_mb = 0, 0.0

    for task in tasks:
        td = DATA_DIR / task
        if not td.exists():
            continue
        scenes = [s for s in td.iterdir()
                  if s.is_dir() and (s / "config.json").exists()]
        n = len(scenes)
        total_n += n

        size_bytes = sum(f.stat().st_size
                         for s in scenes
                         for f in s.rglob("*") if f.is_file())
        mb = size_bytes / 1024 / 1024
        total_mb += mb
        print(f"\n {task.upper()} ({n} scenes, {mb:.1f} MB)")

        if not scenes:
            continue

        sample = scenes[:min(500, n)]
        diffs = defaultdict(int)
        ttcs, ranges = [], []
        stables, unstables = 0, 0

        for s in sample:
            try:
                cfg = json.load(open(s / "config.json"))
                g = json.load(open(s / "ground_truth.json"))
                diffs[cfg["difficulty"]] += 1
                if task == "ttc":
                    ttcs.append(g["time_to_collision"])
                elif task == "stability":
                    (stables if g["is_stable"] else unstables).__class__ # trick
                    if g["is_stable"]: stables += 1
                    else: unstables += 1
                elif task == "trajectory":
                    lp = g["landing_position"]
                    ranges.append(math.sqrt(lp["x"]**2 + lp["y"]**2))
            except Exception:
                pass

        samp = sum(diffs.values()) or 1
        diff_str = " | ".join(
            f"{d}={diffs[d]} ({diffs[d]/samp*100:.0f}%)"
            for d in ["easy", "medium", "hard"] if d in diffs
        )
        print(f" Difficulty: {diff_str}")

        if task == "ttc" and ttcs:
            print(f" TTC: {min(ttcs):.2f}s - {max(ttcs):.2f}s | mean={sum(ttcs)/len(ttcs):.2f}s")
        elif task == "stability" and stables + unstables:
            tot = stables + unstables
            print(f" Stable {stables} ({stables/tot*100:.1f}%) | "
                  f"Unstable {unstables} ({unstables/tot*100:.1f}%)")
        elif task == "trajectory" and ranges:
            print(f" Range: {min(ranges):.2f}m - {max(ranges):.2f}m | mean={sum(ranges)/len(ranges):.2f}m")

    print(f"\n Total: {total_n} scenes | {total_mb:.1f} MB")
    print(f"{'═' * 62}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 12. RANDOM SAMPLE INSPECTOR
# ══════════════════════════════════════════════════════════════════════════════

def inspect_samples(tasks: list[str], n: int = 5):
    print(f"\n{'═' * 62}")
    print(f" RANDOM SAMPLE INSPECTION ({n} per task)")
    print(f"{'═' * 62}")

    for task in tasks:
        td = DATA_DIR / task
        if not td.exists():
            continue
        scenes = [s for s in td.iterdir()
                  if s.is_dir() and (s / "ground_truth.json").exists()]
        if not scenes:
            continue

        chosen = random.sample(scenes, min(n, len(scenes)))
        print(f"\n── {task.upper()} ({'─' * 46})")

        for s in chosen:
            cfg = json.load(open(s / "config.json"))
            g = json.load(open(s / "ground_truth.json"))
            prompt = open(s / "prompt.txt").read()

            print(f"\n Scene : {s.name} [{cfg['difficulty']}]")

            if task == "ttc":
                o1, o2 = cfg["obj1"], cfg["obj2"]
                print(f" Obj A : {o1['color']:8s} {o1['shape']:9s} x={o1['position'][0]:+.2f}m v={o1['velocity'][0]:+.2f} m/s")
                print(f" Obj B : {o2['color']:8s} {o2['shape']:9s} x={o2['position'][0]:+.2f}m v={o2['velocity'][0]:+.2f} m/s")
                print(f" TTC : {g['time_to_collision']:.3f}s | collision={g['collision_occurred']}")
                vid = cfg.get("video", {})
                print(f" Video : {vid.get('n_frames')}f @ {vid.get('fps')}fps "
                      f"duration={vid.get('duration_s')}s "
                      f"(remaining={vid.get('time_remaining_s')}s)")

            elif task == "stability":
                objs = cfg["objects"]
                stack = " → ".join(f"{o['color']} {o['shape']}" for o in objs)
                print(f" Stack : {stack}")
                label = "STABLE" if g["is_stable"] else "UNSTABLE"
                print(f" Result : {label} max_disp={g['max_displacement_m']:.4f}m")
                if not g["is_stable"] and g.get("collapse_time_s"):
                    print(f" Collapse : at {g['collapse_time_s']:.2f}s")

            elif task == "trajectory":
                obj = cfg["object"]
                v = obj["velocity"]
                lp = g["landing_position"]
                print(f" Object : {obj['color']} {obj['shape']} z0={obj['position'][2]:.2f}m")
                print(f" Launch : vx={v[0]:+.2f} vy={v[1]:+.2f} vz={v[2]:+.2f} m/s")
                print(f" Landing : x={lp['x']:.3f}m y={lp['y']:.3f}m "
                      f"flight={g['flight_time_s']:.2f}s bounces={g['n_bounces']}")
                vid = cfg.get("video", {})
                print(f" Video : {vid.get('n_frames')}f @ {vid.get('fps')}fps "
                      f"duration={vid.get('duration_s')}s")

            # Show first line of prompt
            first_line = next((l for l in prompt.splitlines() if l.strip()), "")
            print(f" Prompt[0]: {first_line[:80]}")

            # File sizes
            files = list(s.rglob("*"))
            imgs = [f for f in files if f.suffix == ".png"]
            total = sum(f.stat().st_size for f in files if f.is_file()) / 1024
            print(f" Files : {len(imgs)} PNGs | total {total:.1f} KB")

    print(f"\n{'═' * 62}\n")


# ══════════════════════════════════════════════════════════════════════════════
# 13. TEST RENDER (visual sanity-check, 3 samples per task)
# ══════════════════════════════════════════════════════════════════════════════

def run_test_renders(tasks: list[str]):
    out = ROOT / "data" / "test_renders"
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n Test renders → {out}\n")

    for task in tasks:
        for i, diff in enumerate(["easy", "medium", "hard"]):
            random.seed(i + 1); np.random.seed(i + 1)
            td = out / f"{task}_{diff}"
            td.mkdir(exist_ok=True)

            try:
                if task == "ttc":
                    cfg = sample_ttc_config(diff)
                    g = gt_ttc(cfg)
                    if not qc_ttc(g):
                        print(f" {task} {diff}: no collision - retry")
                        continue
                    frames, vinfo = render_ttc_frames(cfg, g)
                    for j, f in enumerate(frames):
                        Image.fromarray(f).save(td / f"frame_{j:03d}.png")
                    print(f" ✓ {task} {diff:7s}: TTC={g['time_to_collision']:.2f}s "
                          f"{len(frames)}f/{vinfo['duration_s']:.2f}s shown → {td}")

                elif task == "stability":
                    cfg = sample_stability_config(diff)
                    g = gt_stability(cfg)
                    f = render_stability_frame(cfg)
                    Image.fromarray(f).save(td / "scene.png")
                    label = "stable" if g["is_stable"] else "UNSTABLE"
                    print(f" ✓ {task} {diff:7s}: {label} "
                          f"max_disp={g['max_displacement_m']:.3f}m → {td}/scene.png")

                elif task == "trajectory":
                    cfg = sample_trajectory_config(diff)
                    g = gt_trajectory(cfg)
                    if not qc_trajectory(g):
                        print(f" {task} {diff}: bad trajectory - retry")
                        continue
                    frames, vinfo = render_trajectory_frames(cfg, g)
                    for j, f in enumerate(frames):
                        Image.fromarray(f).save(td / f"frame_{j:03d}.png")
                    lp = g["landing_position"]
                    print(f" ✓ {task} {diff:7s}: landing=({lp['x']:.2f}, {lp['y']:.2f})m "
                          f"{len(frames)}f/{vinfo['duration_s']:.2f}s shown → {td}")

            except Exception as e:
                import traceback
                print(f" ✗ {task} {diff}: {e}")
                traceback.print_exc()

    print(f"\n Open images in {out} to verify rendering quality.\n")


# ══════════════════════════════════════════════════════════════════════════════
# 14. ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="PhysSim-VLM Training Data Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_training_data.py --test_render # visual check first
  python scripts/generate_training_data.py --count 50 # quick test (50/task)
  python scripts/generate_training_data.py # full run (5000/task)
  python scripts/generate_training_data.py --task ttc --count 200
  python scripts/generate_training_data.py --inspect_only --inspect 8
        """,
    )
    parser.add_argument("--task", choices=["ttc", "stability", "trajectory", "all"],
                        default="all")
    parser.add_argument("--count", type=int, default=5000,
                        help="Scenes per task (default: 5000)")
    parser.add_argument("--inspect", type=int, default=3,
                        help="Random samples to print per task (default: 3)")
    parser.add_argument("--inspect_only", action="store_true")
    parser.add_argument("--no_splits", action="store_true")
    parser.add_argument("--test_render", action="store_true",
                        help="Render 3 sanity-check images per task and exit")
    parser.add_argument("--seed_base", type=int, default=42)
    parser.add_argument("--start_idx", type=int, default=0,
                        help="First scene index (for parallel workers splitting ID space)")
    args = parser.parse_args()

    tasks = ["ttc", "stability", "trajectory"] if args.task == "all" else [args.task]

    if args.test_render:
        run_test_renders(tasks)
        return

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not args.inspect_only:
        print(f"\n{'═' * 62}")
        print(f" PhysSim-VLM · Training Data Generator")
        print(f" Tasks : {tasks}")
        print(f" Count : {args.count} per task ({args.count * len(tasks):,} total)")
        print(f" Output: {DATA_DIR}")
        print(f"{'═' * 62}\n")

        all_stats = []
        for i, task in enumerate(tasks):
            seed_off = args.seed_base * 10_000 + i * 100_000 + args.start_idx
            s = generate_task(task, args.count, seed_offset=seed_off,
                              start_idx=args.start_idx)
            all_stats.append(s)
            print(f" ✓ {task:12s} {s['count']:5d} scenes "
                  f"accept={s['acceptance_rate']}% "
                  f"errors={s.get('errors', 0)}")

        with open(DATA_DIR / "generation_stats.json", "w") as f:
            json.dump({"generated_at": _ts(), "tasks": all_stats}, f, indent=2)

        if not args.no_splits:
            print(f"\n Creating train/val/test splits…")
            create_splits(tasks)

    print_stats(tasks)

    if args.inspect > 0:
        inspect_samples(tasks, n=args.inspect)


if __name__ == "__main__":
    main()
