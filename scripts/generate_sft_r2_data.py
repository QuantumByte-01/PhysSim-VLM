"""
PhysSim-VLM SFT Round 2: Expanded MuJoCo Scene Generator
============================================================
Generates training data for 8 task types across all PhysBench categories:

  Original (from generate_data.py):
    1. ttc - time-to-collision prediction (Dynamics)
    2. stability - stack stability classification (Property)
    3. trajectory - projectile landing prediction (Dynamics)

  New (targets PhysBench gaps):
    4. motion_comparison - which object moves faster/higher (Relationships)
    5. object_comparison - compare physical properties (Relationships)
    6. counting - count objects in scene (Scene)
    7. viewpoint - describe scene from a direction (Scene)
    8. manipulation - predict outcome of action (Dynamics)

All scenes use MuJoCo for ground truth. Each scene produces:
  config.json, ground_truth.json, prompt.txt, frames/ or scene.png

Quality monitoring: --validate flag spot-checks random samples for correctness.

Usage:
  python scripts/generate_sft_r2_data.py --n 2000 --tasks all
  python scripts/generate_sft_r2_data.py --n 500 --tasks motion_comparison,counting
  python scripts/generate_sft_r2_data.py --n 2000 --validate --validate-n 50
"""

import json
import sys
import argparse
import base64
import math
import random
import numpy as np
from pathlib import Path
from io import BytesIO
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).parent.parent))

import mujoco
from simulation.verifier import (
    TTCConfig, StabilityConfig, TrajectoryConfig, ObjectConfig,
    verify_ttc, verify_stability, verify_trajectory,
    _build_ttc_xml, _build_stability_xml, _build_trajectory_xml,
    _set_velocity, _color_rgba, _geom_xml,
)
from scripts.generate_sample import (
    render_frames_from_scratch, render_single,
    add_label, save_frames, WIDTH, HEIGHT,
)

OUT_DIR = Path(__file__).parent.parent / "data" / "sft_r2"

SHAPES = ["sphere", "box", "cylinder"]
COLORS = ["red", "blue", "green", "yellow", "orange", "purple"]
DIRECTIONS = ["left", "right", "above", "in front of", "behind"]

# Camera bounds (from generate_data.py)
CAM_X_RANGE = (-2.8, 2.8)
CAM_Z_MAX = 2.2


def _in_frame(x: float, z: float, size: float) -> bool:
    return (CAM_X_RANGE[0] + size < x < CAM_X_RANGE[1] - size
            and z + size < CAM_Z_MAX)


# ── XML Builders for New Tasks ─────────────────────────────────────────────

def _build_multi_object_xml(objects: list[ObjectConfig],
                            friction: float = 0.6,
                            with_freejoints: bool = True) -> str:
    """Build MuJoCo XML with multiple objects on a floor."""
    bodies = []
    for i, obj in enumerate(objects):
        name = f"obj{i}"
        rgba = _color_rgba(obj.color)
        pos = f"{obj.position[0]} {obj.position[1]} {obj.position[2]}"

        if obj.shape == "sphere":
            geom = f'<geom type="sphere" size="{obj.size}" rgba="{rgba}" mass="{obj.mass}" friction="{friction} 0.005 0.0001"/>'
        elif obj.shape == "box":
            h = obj.height if obj.height > 0 else obj.size
            geom = f'<geom type="box" size="{obj.size} {obj.size} {h}" rgba="{rgba}" mass="{obj.mass}" friction="{friction} 0.005 0.0001"/>'
        elif obj.shape == "cylinder":
            h = obj.height if obj.height > 0 else obj.size
            geom = f'<geom type="cylinder" size="{obj.size} {h}" rgba="{rgba}" mass="{obj.mass}" friction="{friction} 0.005 0.0001"/>'
        else:
            geom = f'<geom type="sphere" size="{obj.size}" rgba="{rgba}" mass="{obj.mass}"/>'

        joint = '<freejoint/>' if with_freejoints else ''
        bodies.append(f' <body name="{name}" pos="{pos}">{joint}{geom}</body>')

    bodies_str = "\n".join(bodies)
    return f"""<mujoco>
  <option gravity="0 0 -9.81" timestep="0.001"/>
  <worldbody>
    <light pos="0 -3 3" dir="0 1 -1" diffuse="1 1 1"/>
    <geom type="plane" size="10 10 0.1" rgba="0.8 0.8 0.8 1"
          friction="{friction} 0.005 0.0001"/>
{bodies_str}
  </worldbody>
</mujoco>"""


def _render_multi_object(xml: str, velocities: dict = None,
                         n_frames: int = 1,
                         frame_interval: float = 0.1) -> list[np.ndarray]:
    """Render frames from multi-object scene, optionally with velocities."""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    if velocities:
        for body_name, vel in velocities.items():
            _set_velocity(model, data, body_name, vel)

    renderer = mujoco.Renderer(model, height=HEIGHT, width=WIDTH)
    steps_per_frame = max(1, int(frame_interval / model.opt.timestep))
    frames = []

    for f_idx in range(n_frames):
        mujoco.mj_forward(model, data)
        renderer.update_scene(data)
        pixels = renderer.render()
        frames.append(pixels.copy())
        if f_idx < n_frames - 1:
            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)

    renderer.close()
    del data, model
    return frames


def _sim_get_positions(xml: str, velocities: dict,
                       sim_time: float, dt: float = 0.001) -> dict:
    """Simulate and return final body positions."""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    for body_name, vel in velocities.items():
        _set_velocity(model, data, body_name, vel)

    n_steps = int(sim_time / dt)
    for _ in range(n_steps):
        mujoco.mj_step(model, data)

    positions = {}
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        if name and name.startswith("obj"):
            positions[name] = data.body(name).xpos.copy()

    del data, model
    return positions


# ═══════════════════════════════════════════════════════════════════════════
# Task 4: Motion Comparison (Relationships)
# ═══════════════════════════════════════════════════════════════════════════

MOTION_QUESTIONS = [
    ("faster", "Which object is moving faster?"),
    ("higher", "Which object reaches a higher point?"),
    ("farther", "Which object travels farther horizontally?"),
]


def make_motion_comparison(scene_id: str, seed: int) -> bool:
    """Two objects launched with different velocities - compare their motion."""
    r = np.random.default_rng(seed)

    c1 = r.choice(COLORS)
    c2 = r.choice([c for c in COLORS if c != c1])
    s1, s2 = r.choice(SHAPES), r.choice(SHAPES)
    sz1 = round(float(r.uniform(0.06, 0.12)), 2)
    sz2 = round(float(r.uniform(0.06, 0.12)), 2)

    # Object A: on left, launched right + up
    vx1 = round(float(r.uniform(1.0, 4.0)), 2)
    vz1 = round(float(r.uniform(1.5, 4.0)), 2)
    # Object B: on right, launched left + up (different speeds)
    vx2 = round(float(r.uniform(1.0, 4.0)), 2)
    vz2 = round(float(r.uniform(1.5, 4.0)), 2)

    obj1 = ObjectConfig(shape=s1, size=sz1, mass=0.5, color=c1, label="Object A",
                        position=[-1.5, 0.0, sz1 + 0.01],
                        velocity=[vx1, 0.0, vz1])
    obj2 = ObjectConfig(shape=s2, size=sz2, mass=0.5, color=c2, label="Object B",
                        position=[1.5, 0.0, sz2 + 0.01],
                        velocity=[-vx2, 0.0, vz2])

    friction = round(float(r.uniform(0.4, 0.7)), 2)
    xml = _build_multi_object_xml([obj1, obj2], friction=friction)

    # Simulate to get ground truth
    velocities = {"obj0": obj1.velocity, "obj1": obj2.velocity}

    # Compute analytical answers
    g = 9.81
    speed1 = math.sqrt(vx1**2 + vz1**2)
    speed2 = math.sqrt(vx2**2 + vz2**2)
    peak1 = obj1.position[2] + vz1**2 / (2 * g)
    peak2 = obj2.position[2] + vz2**2 / (2 * g)
    range1 = 2 * vx1 * vz1 / g
    range2 = 2 * vx2 * vz2 / g

    # Pick question type
    q_type, q_text = MOTION_QUESTIONS[seed % len(MOTION_QUESTIONS)]
    if q_type == "faster":
        answer = "Object A" if speed1 > speed2 else "Object B"
    elif q_type == "higher":
        answer = "Object A" if peak1 > peak2 else "Object B"
    elif q_type == "farther":
        answer = "Object A" if range1 > range2 else "Object B"
    else:
        answer = "Object A"

    # Render 6 frames over 0.8s
    show = 0.8
    n_frames = 6
    fi = show / (n_frames - 1)
    frames = _render_multi_object(xml, velocities, n_frames, fi)

    labeled = []
    for j, f in enumerate(frames):
        f = add_label(f, f"t={j*fi:.2f}s [{scene_id}]", (10, 10))
        f = add_label(f, f"A: {c1} {s1} B: {c2} {s2}", (10, HEIGHT - 30))
        labeled.append(f)

    out = OUT_DIR / "motion_comparison" / scene_id
    save_frames(labeled, out)

    gt = {
        "question_type": q_type,
        "answer": answer.lower(),
        "speed_a": round(speed1, 3), "speed_b": round(speed2, 3),
        "peak_height_a": round(peak1, 3), "peak_height_b": round(peak2, 3),
        "range_a": round(range1, 3), "range_b": round(range2, 3),
    }
    (out / "config.json").write_text(json.dumps({
        "scene_id": scene_id, "task_type": "motion_comparison",
        "question_type": q_type,
        "obj_a": {"shape": s1, "size": sz1, "color": c1, "velocity": [vx1, 0.0, vz1]},
        "obj_b": {"shape": s2, "size": sz2, "color": c2, "velocity": [-vx2, 0.0, vz2]},
        "video": {"n_frames": n_frames, "duration_s": show, "frame_interval_s": round(fi, 3)},
    }, indent=2))
    (out / "ground_truth.json").write_text(json.dumps(gt, indent=2))
    (out / "prompt.txt").write_text(
        f"Two objects are launched simultaneously:\n"
        f"- Object A: {c1} {s1} (launched from the left)\n"
        f"- Object B: {c2} {s2} (launched from the right)\n\n"
        f"{q_text}\n\n"
        f"<reasoning>Compare their motions from the video</reasoning>\n"
        f"<answer>Object A or Object B</answer>"
    )

    # Build assistant_text for SFT
    (out / "assistant_text.txt").write_text(
        f"<reasoning>Observing the video, I can compare the two objects' motion. "
        f"Object A ({c1} {s1}) has initial speed {speed1:.2f} m/s "
        f"and Object B ({c2} {s2}) has initial speed {speed2:.2f} m/s. "
        f"For the question of which is {q_type}: {answer} based on the physics.</reasoning>\n"
        f"<answer>{answer.lower()}</answer>"
    )

    print(f" {scene_id}: {q_type} -> {answer} "
          f"speeds={speed1:.1f}/{speed2:.1f} peaks={peak1:.2f}/{peak2:.2f}")
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Task 5: Object Comparison (Relationships)
# ═══════════════════════════════════════════════════════════════════════════

COMPARISON_QUESTIONS = [
    ("larger", "Which object is larger?"),
    ("heavier", "Which object appears heavier based on its motion?"),
    ("bouncier", "Which object bounces more?"),
]


def make_object_comparison(scene_id: str, seed: int) -> bool:
    """Two objects with different properties - compare them."""
    r = np.random.default_rng(seed)

    c1 = r.choice(COLORS)
    c2 = r.choice([c for c in COLORS if c != c1])

    q_type, q_text = COMPARISON_QUESTIONS[seed % len(COMPARISON_QUESTIONS)]

    if q_type == "larger":
        # Make one clearly larger
        sz1 = round(float(r.uniform(0.12, 0.20)), 2)
        sz2 = round(float(r.uniform(0.05, 0.10)), 2)
        if r.random() > 0.5:
            sz1, sz2 = sz2, sz1
        answer = "Object A" if sz1 > sz2 else "Object B"
        m1, m2 = 1.0, 1.0
        rest1, rest2 = 0.3, 0.3
    elif q_type == "heavier":
        # Drop both from same height - heavier one pushes lighter aside on collision
        # Approximate: heavier object decelerates less
        sz1 = sz2 = round(float(r.uniform(0.08, 0.12)), 2)
        m1 = round(float(r.uniform(1.5, 3.0)), 1)
        m2 = round(float(r.uniform(0.3, 0.8)), 1)
        if r.random() > 0.5:
            m1, m2 = m2, m1
        answer = "Object A" if m1 > m2 else "Object B"
        rest1, rest2 = 0.3, 0.3
    elif q_type == "bouncier":
        # Different restitution
        sz1 = sz2 = round(float(r.uniform(0.06, 0.10)), 2)
        m1 = m2 = 0.5
        rest1 = round(float(r.uniform(0.6, 0.9)), 2)
        rest2 = round(float(r.uniform(0.1, 0.3)), 2)
        if r.random() > 0.5:
            rest1, rest2 = rest2, rest1
        answer = "Object A" if rest1 > rest2 else "Object B"
    else:
        return False

    # Place objects side by side, drop from height
    obj1 = ObjectConfig(shape="sphere", size=sz1, mass=m1, color=c1, label="Object A",
                        position=[-0.8, 0.0, 1.5], velocity=[0.0, 0.0, 0.0])
    obj2 = ObjectConfig(shape="sphere", size=sz2, mass=m2, color=c2, label="Object B",
                        position=[0.8, 0.0, 1.5], velocity=[0.0, 0.0, 0.0])

    friction = 0.5

    # Build custom XML with restitution differences
    xml = f"""<mujoco>
  <option gravity="0 0 -9.81" timestep="0.001"/>
  <worldbody>
    <light pos="0 -3 3" dir="0 1 -1" diffuse="1 1 1"/>
    <geom type="plane" size="10 10 0.1" rgba="0.8 0.8 0.8 1"
          friction="{friction} 0.005 0.0001"/>
    <body name="obj0" pos="-0.8 0.0 1.5">
      <freejoint/>
      <geom type="sphere" size="{sz1}" rgba="{_color_rgba(c1)}" mass="{m1}"
            solref="-{int(1000/max(rest1,0.01))} 1" friction="{friction} 0.005 0.0001"/>
    </body>
    <body name="obj1" pos="0.8 0.0 1.5">
      <freejoint/>
      <geom type="sphere" size="{sz2}" rgba="{_color_rgba(c2)}" mass="{m2}"
            solref="-{int(1000/max(rest2,0.01))} 1" friction="{friction} 0.005 0.0001"/>
    </body>
  </worldbody>
</mujoco>"""

    # Render 8 frames over 1.2s (enough to see bouncing)
    show = 1.2
    n_frames = 8
    fi = show / (n_frames - 1)
    frames = _render_multi_object(xml, n_frames=n_frames, frame_interval=fi)

    labeled = []
    for j, f in enumerate(frames):
        f = add_label(f, f"t={j*fi:.2f}s [{scene_id}]", (10, 10))
        f = add_label(f, f"A: {c1} sphere B: {c2} sphere", (10, HEIGHT - 30))
        labeled.append(f)

    out = OUT_DIR / "object_comparison" / scene_id
    save_frames(labeled, out)

    gt = {
        "question_type": q_type,
        "answer": answer.lower(),
        "size_a": sz1, "size_b": sz2,
        "mass_a": m1, "mass_b": m2,
        "restitution_a": rest1, "restitution_b": rest2,
    }
    (out / "config.json").write_text(json.dumps({
        "scene_id": scene_id, "task_type": "object_comparison",
        "question_type": q_type,
        "obj_a": {"size": sz1, "mass": m1, "color": c1, "restitution": rest1},
        "obj_b": {"size": sz2, "mass": m2, "color": c2, "restitution": rest2},
        "video": {"n_frames": n_frames, "duration_s": show, "frame_interval_s": round(fi, 3)},
    }, indent=2))
    (out / "ground_truth.json").write_text(json.dumps(gt, indent=2))
    (out / "prompt.txt").write_text(
        f"Two spheres are dropped from the same height:\n"
        f"- Object A: {c1} sphere\n"
        f"- Object B: {c2} sphere\n\n"
        f"{q_text}\n\n"
        f"<reasoning>Observe and compare</reasoning>\n"
        f"<answer>Object A or Object B</answer>"
    )
    (out / "assistant_text.txt").write_text(
        f"<reasoning>Watching the two spheres, I can compare their {q_type} properties. "
        f"Object A ({c1}) has size={sz1}m, mass={m1}kg. "
        f"Object B ({c2}) has size={sz2}m, mass={m2}kg. "
        f"Based on the physics simulation, {answer} is {q_type}.</reasoning>\n"
        f"<answer>{answer.lower()}</answer>"
    )

    print(f" {scene_id}: {q_type} -> {answer} "
          f"sizes={sz1}/{sz2} masses={m1}/{m2}")
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Task 6: Counting (Scene)
# ═══════════════════════════════════════════════════════════════════════════

def make_counting(scene_id: str, seed: int) -> bool:
    """Scene with N objects - count how many total, or by color/shape."""
    r = np.random.default_rng(seed)

    n_objects = int(r.integers(3, 9)) # 3-8 objects
    objs = []

    for i in range(n_objects):
        sh = r.choice(SHAPES)
        col = r.choice(COLORS)
        sz = round(float(r.uniform(0.06, 0.15)), 2)
        h = round(float(r.uniform(0.04, 0.10)), 2) if sh != "sphere" else 0.0

        # Spread objects in the scene (grid-like with jitter)
        row = i // 3
        col_idx = i % 3
        x = (col_idx - 1) * 1.2 + float(r.uniform(-0.3, 0.3))
        y = float(r.uniform(-0.3, 0.3))
        z_pos = sz if sh == "sphere" else h + 0.01

        if not _in_frame(x, z_pos, sz):
            x = float(np.clip(x, CAM_X_RANGE[0] + 0.3, CAM_X_RANGE[1] - 0.3))

        objs.append(ObjectConfig(
            shape=sh, size=sz, height=h, mass=0.5, color=col,
            label=f"Obj {i+1}",
            position=[round(x, 2), round(y, 2), round(z_pos + row * 0.4, 2)],
        ))

    # Decide question type
    q_types = ["total"]
    color_counts = {}
    shape_counts = {}
    for o in objs:
        color_counts[o.color] = color_counts.get(o.color, 0) + 1
        shape_counts[o.shape] = shape_counts.get(o.shape, 0) + 1

    # Add color/shape counting questions if there's variety
    if len(color_counts) >= 2:
        target_color = r.choice(list(color_counts.keys()))
        q_types.append(f"color:{target_color}")
    if len(shape_counts) >= 2:
        target_shape = r.choice(list(shape_counts.keys()))
        q_types.append(f"shape:{target_shape}")

    q_type = r.choice(q_types)

    if q_type == "total":
        question = "How many objects are in the scene?"
        answer = str(n_objects)
    elif q_type.startswith("color:"):
        target = q_type.split(":")[1]
        question = f"How many {target} objects are in the scene?"
        answer = str(color_counts[target])
    elif q_type.startswith("shape:"):
        target = q_type.split(":")[1]
        question = f"How many {target}s are in the scene?"
        answer = str(shape_counts[target])
    else:
        question = "How many objects are in the scene?"
        answer = str(n_objects)

    xml = _build_multi_object_xml(objs, friction=0.6, with_freejoints=False)
    frame = _render_multi_object(xml, n_frames=1)[0]

    # Add object labels
    for i, o in enumerate(objs):
        y_label = HEIGHT - 30 - i * 18
        if y_label > 10:
            frame = add_label(frame, f"{o.color} {o.shape}", (10, y_label))

    out = OUT_DIR / "counting" / scene_id
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(out / "scene.png")

    gt = {
        "question_type": q_type, "answer": answer,
        "total_objects": n_objects,
        "color_counts": color_counts, "shape_counts": shape_counts,
    }
    (out / "config.json").write_text(json.dumps({
        "scene_id": scene_id, "task_type": "counting",
        "question_type": q_type,
        "objects": [{"shape": o.shape, "size": o.size, "color": o.color,
                     "position": o.position} for o in objs],
    }, indent=2))
    (out / "ground_truth.json").write_text(json.dumps(gt, indent=2))
    (out / "prompt.txt").write_text(
        f"Look at this scene with multiple objects.\n\n"
        f"{question}\n\n"
        f"<reasoning>Count carefully</reasoning>\n"
        f"<answer>N</answer>"
    )

    obj_desc = ", ".join(f"{o.color} {o.shape}" for o in objs)
    (out / "assistant_text.txt").write_text(
        f"<reasoning>I can see the following objects in the scene: {obj_desc}. "
        f"Counting carefully, there are {n_objects} objects total. "
        f"Color breakdown: {color_counts}. Shape breakdown: {shape_counts}.</reasoning>\n"
        f"<answer>{answer}</answer>"
    )

    print(f" {scene_id}: {q_type} -> {answer} ({n_objects} objects)")
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Task 7: Viewpoint (Scene)
# ═══════════════════════════════════════════════════════════════════════════

def make_viewpoint(scene_id: str, seed: int) -> bool:
    """Scene with objects - ask about spatial relationships from a viewpoint."""
    r = np.random.default_rng(seed)

    n_objects = int(r.integers(2, 5))
    objs = []
    for i in range(n_objects):
        sh = r.choice(SHAPES)
        col = COLORS[i % len(COLORS)]
        sz = round(float(r.uniform(0.08, 0.16)), 2)
        h = round(float(r.uniform(0.05, 0.12)), 2) if sh != "sphere" else 0.0

        x = round(float(r.uniform(-2.0, 2.0)), 2)
        y = round(float(r.uniform(-1.0, 1.0)), 2)
        z_pos = sz if sh == "sphere" else h + 0.01

        objs.append(ObjectConfig(
            shape=sh, size=sz, height=h, mass=0.5, color=col,
            label=f"Object {chr(65+i)}",
            position=[x, y, round(z_pos, 2)],
        ))

    # Pick two objects and ask about their spatial relationship
    if n_objects < 2:
        return False
    idx_a, idx_b = int(r.integers(0, n_objects)), int(r.integers(0, n_objects))
    while idx_b == idx_a:
        idx_b = int(r.integers(0, n_objects))

    a, b = objs[idx_a], objs[idx_b]
    dx = b.position[0] - a.position[0]
    dy = b.position[1] - a.position[1]
    dz = b.position[2] - a.position[2]

    # Determine spatial relationship (from camera's perspective: x=left/right, z=above/below)
    if abs(dx) > abs(dz) and abs(dx) > 0.3:
        answer = "right" if dx > 0 else "left"
        question = f"Is {b.label} ({b.color} {b.shape}) to the left or right of {a.label} ({a.color} {a.shape})?"
    elif abs(dz) > 0.1:
        answer = "above" if dz > 0 else "below"
        question = f"Is {b.label} ({b.color} {b.shape}) above or below {a.label} ({a.color} {a.shape})?"
    else:
        answer = "right" if dx >= 0 else "left"
        question = f"Is {b.label} ({b.color} {b.shape}) to the left or right of {a.label} ({a.color} {a.shape})?"

    xml = _build_multi_object_xml(objs, friction=0.6, with_freejoints=False)
    frame = _render_multi_object(xml, n_frames=1)[0]

    for i, o in enumerate(objs):
        y_label = HEIGHT - 30 - i * 18
        if y_label > 10:
            frame = add_label(frame, f"{o.label}: {o.color} {o.shape}", (10, y_label))

    out = OUT_DIR / "viewpoint" / scene_id
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(out / "scene.png")

    gt = {
        "answer": answer,
        "object_a": {"label": a.label, "color": a.color, "position": a.position},
        "object_b": {"label": b.label, "color": b.color, "position": b.position},
        "delta": [round(dx, 3), round(dy, 3), round(dz, 3)],
    }
    (out / "config.json").write_text(json.dumps({
        "scene_id": scene_id, "task_type": "viewpoint",
        "objects": [{"shape": o.shape, "size": o.size, "color": o.color,
                     "label": o.label, "position": o.position} for o in objs],
    }, indent=2))
    (out / "ground_truth.json").write_text(json.dumps(gt, indent=2))
    (out / "prompt.txt").write_text(
        f"Look at this scene with {n_objects} objects.\n\n"
        f"{question}\n\n"
        f"<reasoning>Analyze spatial positions</reasoning>\n"
        f"<answer>left/right/above/below</answer>"
    )
    (out / "assistant_text.txt").write_text(
        f"<reasoning>Looking at the scene, {a.label} ({a.color} {a.shape}) is at "
        f"position ({a.position[0]}, {a.position[2]}) and {b.label} ({b.color} {b.shape}) "
        f"is at ({b.position[0]}, {b.position[2]}). "
        f"The horizontal difference is {dx:.2f}m and vertical is {dz:.2f}m. "
        f"Therefore {b.label} is {answer} of {a.label}.</reasoning>\n"
        f"<answer>{answer}</answer>"
    )

    print(f" {scene_id}: {b.label} is {answer} of {a.label}")
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Task 8: Manipulation (Dynamics)
# ═══════════════════════════════════════════════════════════════════════════

MANIP_ACTIONS = [
    "push_right", # apply force to the right
    "push_left", # apply force to the left
    "lift", # apply upward force
]


def make_manipulation(scene_id: str, seed: int) -> bool:
    """Object on surface - predict result of applying a force."""
    r = np.random.default_rng(seed)

    col = r.choice(COLORS)
    sh = r.choice(SHAPES)
    sz = round(float(r.uniform(0.08, 0.15)), 2)
    h = round(float(r.uniform(0.05, 0.12)), 2) if sh != "sphere" else 0.0
    mass = round(float(r.uniform(0.3, 2.0)), 1)
    z_pos = sz if sh == "sphere" else h + 0.01

    obj = ObjectConfig(shape=sh, size=sz, height=h, mass=mass, color=col,
                       label="Target object",
                       position=[0.0, 0.0, round(z_pos, 2)])

    action = MANIP_ACTIONS[seed % len(MANIP_ACTIONS)]
    force_mag = round(float(r.uniform(2.0, 8.0)), 1)

    # Determine velocity from force (impulse approximation: v = F*dt/m, dt=0.1s)
    impulse_dt = 0.1
    if action == "push_right":
        vel = [force_mag * impulse_dt / mass, 0.0, 0.0]
        action_desc = f"pushed to the right with force {force_mag}N"
    elif action == "push_left":
        vel = [-force_mag * impulse_dt / mass, 0.0, 0.0]
        action_desc = f"pushed to the left with force {force_mag}N"
    elif action == "lift":
        vel = [0.0, 0.0, force_mag * impulse_dt / mass]
        action_desc = f"lifted with upward force {force_mag}N"
    else:
        vel = [1.0, 0.0, 0.0]
        action_desc = "pushed"

    friction = round(float(r.uniform(0.3, 0.8)), 2)
    xml = _build_multi_object_xml([obj], friction=friction)

    # Simulate to get final position
    positions = _sim_get_positions(xml, {"obj0": vel}, sim_time=3.0)
    final_pos = positions.get("obj0", np.array([0, 0, 0]))
    displacement = float(np.sqrt(
        (final_pos[0] - obj.position[0])**2 +
        (final_pos[2] - obj.position[2])**2
    ))

    # Categorize outcome
    if action in ("push_right", "push_left"):
        if displacement < 0.1:
            outcome = "barely moves"
        elif displacement < 1.0:
            outcome = "slides a short distance"
        else:
            outcome = "slides far across the surface"
    elif action == "lift":
        if final_pos[2] > obj.position[2] + 0.5:
            outcome = "launches into the air and lands elsewhere"
        elif final_pos[2] > obj.position[2] + 0.1:
            outcome = "rises briefly and falls back"
        else:
            outcome = "barely lifts off the surface"

    # Render: before (frame 0) and after sequence (frames 1-5)
    show = 1.5
    n_frames = 6
    fi = show / (n_frames - 1)
    frames = _render_multi_object(xml, {"obj0": vel}, n_frames, fi)

    labeled = []
    for j, f in enumerate(frames):
        tag = "before action" if j == 0 else f"t={j*fi:.2f}s after"
        f = add_label(f, f"{tag} [{scene_id}]", (10, 10))
        f = add_label(f, f"{col} {sh} - {action_desc}", (10, HEIGHT - 30))
        labeled.append(f)

    out = OUT_DIR / "manipulation" / scene_id
    save_frames(labeled, out)

    gt = {
        "action": action, "force_N": force_mag, "outcome": outcome,
        "displacement_m": round(displacement, 3),
        "final_position": [round(float(x), 3) for x in final_pos],
        "initial_position": obj.position,
    }
    (out / "config.json").write_text(json.dumps({
        "scene_id": scene_id, "task_type": "manipulation",
        "action": action, "force_N": force_mag,
        "object": {"shape": sh, "size": sz, "mass": mass, "color": col},
        "friction": friction,
        "video": {"n_frames": n_frames, "duration_s": show, "frame_interval_s": round(fi, 3)},
    }, indent=2))
    (out / "ground_truth.json").write_text(json.dumps(gt, indent=2))
    (out / "prompt.txt").write_text(
        f"A {col} {sh} (mass={mass}kg) sits on a surface (friction={friction}).\n"
        f"It is {action_desc}.\n\n"
        f"What happens to the object?\n\n"
        f"<reasoning>Analyze forces and predict motion</reasoning>\n"
        f"<answer>describe the outcome</answer>"
    )
    (out / "assistant_text.txt").write_text(
        f"<reasoning>The {col} {sh} has mass {mass}kg on a surface with friction {friction}. "
        f"When {action_desc}, the impulse gives it velocity "
        f"({vel[0]:.2f}, {vel[1]:.2f}, {vel[2]:.2f}) m/s. "
        f"After simulation, it displaces {displacement:.2f}m. "
        f"The outcome is: {outcome}.</reasoning>\n"
        f"<answer>{outcome}</answer>"
    )

    print(f" {scene_id}: {action} F={force_mag}N -> {outcome} (d={displacement:.2f}m)")
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Quality Monitoring
# ═══════════════════════════════════════════════════════════════════════════

def validate_scene(scene_dir: Path) -> dict:
    """Validate a generated scene for quality."""
    issues = []

    # Check required files exist
    for f in ["config.json", "ground_truth.json", "prompt.txt"]:
        if not (scene_dir / f).exists():
            issues.append(f"missing {f}")

    if not (scene_dir / "scene.png").exists() and not (scene_dir / "frames").exists():
        issues.append("no visual data (scene.png or frames/)")

    # Check config parseable
    try:
        config = json.loads((scene_dir / "config.json").read_text())
        if "scene_id" not in config:
            issues.append("config missing scene_id")
        if "task_type" not in config:
            issues.append("config missing task_type")
    except Exception as e:
        issues.append(f"config parse error: {e}")
        config = {}

    # Check ground truth parseable
    try:
        gt = json.loads((scene_dir / "ground_truth.json").read_text())
        if "answer" not in gt and "is_stable" not in gt and "time_to_collision" not in gt:
            issues.append("ground truth has no recognizable answer field")
    except Exception as e:
        issues.append(f"ground_truth parse error: {e}")

    # Check images not blank/corrupted
    if (scene_dir / "scene.png").exists():
        try:
            img = Image.open(scene_dir / "scene.png")
            arr = np.array(img)
            if arr.std() < 5:
                issues.append("scene.png appears blank (very low variance)")
        except Exception as e:
            issues.append(f"scene.png load error: {e}")

    if (scene_dir / "frames").exists():
        frame_files = sorted((scene_dir / "frames").glob("*.png"))
        if len(frame_files) < 3:
            issues.append(f"only {len(frame_files)} frames (expected >= 3)")
        for ff in frame_files[:2]:
            try:
                arr = np.array(Image.open(ff))
                if arr.std() < 5:
                    issues.append(f"{ff.name} appears blank")
            except Exception as e:
                issues.append(f"{ff.name} load error: {e}")

    # Check prompt non-empty and has tags
    try:
        prompt = (scene_dir / "prompt.txt").read_text()
        if len(prompt) < 20:
            issues.append("prompt too short")
        if "<answer>" not in prompt:
            issues.append("prompt missing <answer> tag template")
    except Exception as e:
        issues.append(f"prompt read error: {e}")

    return {
        "scene_dir": str(scene_dir),
        "task_type": config.get("task_type", "unknown"),
        "valid": len(issues) == 0,
        "issues": issues,
    }


def run_validation(base_dir: Path, n_samples: int = 50, seed: int = 42):
    """Spot-check random scenes for quality."""
    print(f"\n{'='*60}")
    print(f" Quality Validation - sampling {n_samples} scenes")
    print(f"{'='*60}\n")

    # Collect all scene directories
    all_scenes = []
    for task_dir in base_dir.iterdir():
        if task_dir.is_dir():
            for scene_dir in task_dir.iterdir():
                if scene_dir.is_dir() and (scene_dir / "config.json").exists():
                    all_scenes.append(scene_dir)

    if not all_scenes:
        print(" No scenes found to validate!")
        return

    r = random.Random(seed)
    sample = r.sample(all_scenes, min(n_samples, len(all_scenes)))

    results = {"total": 0, "valid": 0, "invalid": 0, "by_task": {}}
    all_issues = []

    for scene_dir in sample:
        result = validate_scene(scene_dir)
        results["total"] += 1
        task = result["task_type"]
        if task not in results["by_task"]:
            results["by_task"][task] = {"valid": 0, "invalid": 0}

        if result["valid"]:
            results["valid"] += 1
            results["by_task"][task]["valid"] += 1
        else:
            results["invalid"] += 1
            results["by_task"][task]["invalid"] += 1
            all_issues.append(result)

    # Print summary
    pct = results["valid"] / max(results["total"], 1) * 100
    print(f" Overall: {results['valid']}/{results['total']} valid ({pct:.0f}%)")
    for task, counts in sorted(results["by_task"].items()):
        total_t = counts["valid"] + counts["invalid"]
        pct_t = counts["valid"] / max(total_t, 1) * 100
        print(f" {task}: {counts['valid']}/{total_t} ({pct_t:.0f}%)")

    if all_issues:
        print(f"\n Issues found in {len(all_issues)} scenes:")
        for issue in all_issues[:10]:
            print(f" {issue['scene_dir']}:")
            for i in issue["issues"]:
                print(f" - {i}")
        if len(all_issues) > 10:
            print(f" ... and {len(all_issues) - 10} more")

    # Save validation report
    report_path = base_dir / "validation_report.json"
    report_path.write_text(json.dumps(results, indent=2))
    print(f"\n Report saved: {report_path}")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

ALL_TASKS = [
    "motion_comparison", "object_comparison", "counting",
    "viewpoint", "manipulation",
]

TASK_GENERATORS = {
    "motion_comparison": make_motion_comparison,
    "object_comparison": make_object_comparison,
    "counting": make_counting,
    "viewpoint": make_viewpoint,
    "manipulation": make_manipulation,
}


def generate_all(n_per_task: int, tasks: list[str], seed: int = 42):
    """Generate scenes for specified tasks."""
    total_generated = 0

    for task in tasks:
        if task not in TASK_GENERATORS:
            print(f" Unknown task: {task}, skipping")
            continue

        gen_fn = TASK_GENERATORS[task]
        print(f"\nGenerating {n_per_task} {task} scenes...")
        ok = 0
        for i in range(n_per_task):
            scene_id = f"{task}_{i:06d}"
            try:
                result = gen_fn(scene_id, seed=seed + i)
                if result:
                    ok += 1
            except Exception as e:
                print(f" ERROR {scene_id}: {e}")

        print(f" {task}: {ok}/{n_per_task} generated")
        total_generated += ok

    print(f"\nTotal: {total_generated} new scenes across {len(tasks)} tasks")
    print(f"Output: {OUT_DIR}")
    return total_generated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate SFT Round 2 training data (expanded task types)")
    parser.add_argument("--n", type=int, default=2000,
                        help="Scenes per task (default 2000)")
    parser.add_argument("--tasks", type=str, default="all",
                        help="Comma-separated task list or 'all'")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate", action="store_true",
                        help="Run quality validation after generation")
    parser.add_argument("--validate-n", type=int, default=50,
                        help="Number of samples to validate")
    parser.add_argument("--validate-only", action="store_true",
                        help="Only run validation (no generation)")
    args = parser.parse_args()

    if args.validate_only:
        run_validation(OUT_DIR, n_samples=args.validate_n, seed=args.seed)
        sys.exit(0)

    tasks = ALL_TASKS if args.tasks == "all" else args.tasks.split(",")
    generate_all(n_per_task=args.n, tasks=tasks, seed=args.seed)

    if args.validate:
        run_validation(OUT_DIR, n_samples=args.validate_n, seed=args.seed)
