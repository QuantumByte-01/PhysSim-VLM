"""
Generate additional varied samples for all three task types.
Run: python scripts/generate_more.py
"""

import json, sys
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

import mujoco
from simulation.verifier import (
    TTCConfig, StabilityConfig, TrajectoryConfig, ObjectConfig,
    verify_ttc, verify_stability, verify_trajectory,
    _build_ttc_xml, _build_stability_xml, _build_trajectory_xml,
    _set_velocity,
)
from scripts.generate_sample import (
    render_frames_from_scratch, render_single,
    add_label, save_frames, OUT_DIR, WIDTH, HEIGHT,
)

SHAPES = ["sphere", "box", "cylinder"]
COLORS = ["red", "blue", "green", "yellow", "orange", "purple"]
rng = np.random.default_rng(seed=123)


# ---------------------------------------------------------------------------
# TTC
# ---------------------------------------------------------------------------

def make_ttc(scene_id: str, seed: int) -> None:
    r = np.random.default_rng(seed)
    s1, s2 = r.choice(SHAPES), r.choice(SHAPES)
    c1 = r.choice(COLORS)
    c2 = r.choice([c for c in COLORS if c != c1])
    sz1 = round(float(r.uniform(0.08, 0.14)), 2)
    sz2 = round(float(r.uniform(0.08, 0.14)), 2)

    # Keep objects close enough to guarantee collision
    gap = round(float(r.uniform(1.5, 3.0)), 2)
    x1 = -gap / 2
    x2 = gap / 2
    v1 = round(float(r.uniform(2.0, 4.0)), 2) # fast enough to always collide
    v2 = round(float(r.uniform(0.5, 1.5)), 2)

    cfg = TTCConfig(
        obj1=ObjectConfig(shape=s1, size=sz1, mass=1.0, color=c1, label="Object A",
                          position=[x1, 0.0, sz1 + 0.01],
                          velocity=[v1, 0.0, 0.0]),
        obj2=ObjectConfig(shape=s2, size=sz2, mass=1.0, color=c2, label="Object B",
                          position=[x2, 0.0, sz2 + 0.01],
                          velocity=[-v2, 0.0, 0.0]),
        surface_friction=round(float(r.uniform(0.2, 0.6)), 2),
    )
    gt = verify_ttc(cfg)
    if not gt["collision_occurred"]:
        print(f" {scene_id}: no collision, skipping")
        return

    ttc = gt["time_to_collision"]
    show = ttc * float(r.uniform(0.60, 0.78))
    n = 8
    fi = show / (n - 1)

    xml = _build_ttc_xml(cfg)
    frames = render_frames_from_scratch(
        xml, {"obj1": cfg.obj1.velocity, "obj2": cfg.obj2.velocity}, n, fi)
    labeled = []
    for j, f in enumerate(frames):
        f = add_label(f, f"t={j*fi:.2f}s [{scene_id}]", (10, 10))
        f = add_label(f, f"Object A: {c1} {s1} Object B: {c2} {s2}", (10, HEIGHT - 30))
        labeled.append(f)

    out = OUT_DIR / "ttc" / scene_id
    save_frames(labeled, out)
    (out / "config.json").write_text(json.dumps({
        "scene_id": scene_id, "task_type": "ttc",
        "obj1": {"shape": s1, "size": sz1, "color": c1,
                 "position": [x1, 0.0, sz1+0.01], "velocity": [v1, 0.0, 0.0]},
        "obj2": {"shape": s2, "size": sz2, "color": c2,
                 "position": [x2, 0.0, sz2+0.01], "velocity": [-v2, 0.0, 0.0]},
        "surface_friction": cfg.surface_friction,
        "video": {"n_frames": n, "fps": 10, "duration_s": round(show, 3),
                  "frame_interval_s": round(fi, 3)},
    }, indent=2))
    (out / "ground_truth.json").write_text(json.dumps(gt, indent=2))
    (out / "prompt.txt").write_text(
        f"You are watching {c1} {s1} (Object A) and {c2} {s2} (Object B) approach each other.\n"
        f"The clip is {show:.1f}s long at 10 fps.\n\n"
        f"Based on their motion, predict the time of collision from the start of the video.\n\n"
        f"<reasoning>Your analysis</reasoning>\n<answer>X.XX</answer>"
    )
    print(f" {scene_id}: TTC={ttc}s shapes={s1}/{s2} colors={c1}/{c2} gap={gap}m")


# ---------------------------------------------------------------------------
# Stability
# ---------------------------------------------------------------------------

def make_stability(scene_id: str, seed: int) -> None:
    r = np.random.default_rng(seed)
    n_obj = int(r.integers(2, 5))
    objs = []
    z = 0.0
    for j in range(n_obj):
        sh = r.choice(SHAPES)
        sz = round(float(r.uniform(0.06, 0.20)), 2)
        h = round(float(r.uniform(0.04, 0.12)), 2) if sh != "sphere" else 0.0
        mass = round(float(r.uniform(0.3, 2.5)), 1)
        col = COLORS[j % len(COLORS)]
        # First object centered; subsequent ones slightly offset
        ox = 0.0 if j == 0 else round(float(r.uniform(-0.08, 0.18)), 3)
        oy = 0.0 if j == 0 else round(float(r.uniform(-0.04, 0.04)), 3)
        half_h = h if sh != "sphere" else sz
        z += half_h + 0.005
        objs.append(ObjectConfig(shape=sh, size=sz, height=h, mass=mass, color=col,
                                  label=f"Object {j+1}", position=[ox, oy, round(z, 3)]))
        z += half_h

    cfg = StabilityConfig(objects=objs, surface_friction=round(float(r.uniform(0.4, 0.8)), 2))
    gt = verify_stability(cfg)
    xml = _build_stability_xml(cfg)
    frame = render_single(xml)
    for j, o in enumerate(objs):
        frame = add_label(frame, f"{o.label}: {o.color} {o.shape}",
                          (10, HEIGHT - 30 - j * 20))
    out = OUT_DIR / "stability" / scene_id
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(out / "scene.png")
    (out / "config.json").write_text(json.dumps({
        "scene_id": scene_id, "task_type": "stability",
        "objects": [{"shape": o.shape, "size": o.size, "height": o.height,
                     "mass": o.mass, "color": o.color, "position": o.position}
                    for o in objs],
        "surface_friction": cfg.surface_friction,
    }, indent=2))
    (out / "ground_truth.json").write_text(json.dumps(gt, indent=2))
    labels = [f"{o.color} {o.shape}" for o in objs]
    (out / "prompt.txt").write_text(
        f"A stack of {n_obj} objects: {', '.join(labels)} (bottom to top).\n"
        f"Will it stay stable or topple?\n\n"
        f"<reasoning>Your analysis</reasoning>\n"
        f"<answer>stable OR unstable</answer>\n<confidence>XX%</confidence>"
    )
    ans = "stable" if gt["is_stable"] else "unstable"
    print(f" {scene_id}: {ans} max_disp={gt['max_displacement_m']}m {n_obj} objects")


# ---------------------------------------------------------------------------
# Trajectory
# ---------------------------------------------------------------------------

def make_trajectory(scene_id: str, seed: int) -> None:
    r = np.random.default_rng(seed)
    # Constrain velocities to keep landing distance reasonable (< 10m)
    vx = round(float(r.uniform(1.0, 2.5)), 2)
    vz = round(float(r.uniform(2.0, 4.5)), 2)
    sz = round(float(r.uniform(0.05, 0.09)), 2)
    col = r.choice(COLORS)
    friction = round(float(r.uniform(0.5, 0.9)), 2) # high friction = shorter slide

    cfg = TrajectoryConfig(
        obj=ObjectConfig(shape="sphere", size=sz, mass=0.3, color=col,
                         label="Ball", position=[0.0, 0.0, 0.5],
                         velocity=[vx, 0.0, vz]),
        surface_friction=friction,
        restitution=round(float(r.uniform(0.1, 0.4)), 2),
    )
    gt = verify_trajectory(cfg)
    ft = gt["flight_time_s"] or 1.0
    show = ft * float(r.uniform(0.35, 0.50))
    show = max(0.3, min(1.5, show))
    n = 5
    fi = show / (n - 1)

    xml = _build_trajectory_xml(cfg)
    frames = render_frames_from_scratch(xml, {"obj": cfg.obj.velocity}, n, fi)
    labeled = []
    for j, f in enumerate(frames):
        f = add_label(f, f"t={j*fi:.2f}s [{scene_id}]", (10, 10))
        f = add_label(f, f"{col} ball - predict landing", (10, HEIGHT - 30))
        labeled.append(f)

    out = OUT_DIR / "trajectory" / scene_id
    save_frames(labeled, out)
    (out / "config.json").write_text(json.dumps({
        "scene_id": scene_id, "task_type": "trajectory",
        "object": {"shape": "sphere", "size": sz, "color": col,
                   "velocity": [vx, 0.0, vz]},
        "surface_friction": friction, "restitution": cfg.restitution,
        "video": {"n_frames": n, "fps": 10, "duration_s": round(show, 3),
                  "frame_interval_s": round(fi, 3)},
    }, indent=2))
    (out / "ground_truth.json").write_text(json.dumps(gt, indent=2))
    (out / "prompt.txt").write_text(
        f"Watch the {col} ball launch (first {show:.1f}s, 5 frames).\n"
        f"Predict final resting position relative to launch point.\n\n"
        f"<reasoning>Analyze the trajectory</reasoning>\n"
        f"<answer>x=X.XX, y=X.XX</answer>"
    )
    lp = gt["landing_position"]
    print(f" {scene_id}: x={lp['x']}m vx={vx} vz={vz} friction={friction}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Generating 3 more TTC samples...")
    for i, seed in enumerate([200, 300, 400], start=2):
        make_ttc(f"ttc_00000{i}", seed)

    print("\nGenerating 3 more Stability samples...")
    for i, seed in enumerate([500, 600, 700], start=2):
        make_stability(f"stability_00000{i}", seed)

    print("\nGenerating 3 more Trajectory samples...")
    for i, seed in enumerate([800, 900, 1000], start=2):
        make_trajectory(f"trajectory_00000{i}", seed)

    print("\nDone. All samples in data/generated/")
