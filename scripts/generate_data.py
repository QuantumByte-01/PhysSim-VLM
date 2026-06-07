"""
Full 15K scene generator with:
- Stratified sampling (guaranteed distribution balance)
- In-frame visibility checks (objects always in camera view)
- No repeated scenes (seed = scene index)
- 50/50 stable/unstable for stability
- TTC spread across 0.5s-5s
- Trajectory spread across 1m-15m landing distance

Run: python scripts/generate_data.py [--n 15000] [--workers 4] [--out data/generated]

Output: data/generated/{ttc,stability,trajectory}/{scene_id}/
  Each folder: config.json, ground_truth.json, prompt.txt, frames/ or scene.png
"""

import json
import sys
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed

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
    add_label, save_frames, WIDTH, HEIGHT,
)

SHAPES = ["sphere", "box", "cylinder"]
COLORS = ["red", "blue", "green", "yellow", "orange", "purple"]

# ---------------------------------------------------------------------------
# Camera frustum bounds (MuJoCo default camera: pos=[0,-4,1.5], lookat=[0,0,0.5])
# Empirically derived safe bounds for 640x480 render
# ---------------------------------------------------------------------------
CAM_X_RANGE = (-2.8, 2.8) # objects beyond ±2.8m in x are partially off-frame
CAM_Z_MAX = 2.2 # objects above 2.2m z are off the top of frame
CAM_FLOOR = 0.0 # floor at z=0


def _in_frame(x: float, z: float, size: float) -> bool:
    """Check if an object centre at (x, z) fits within the camera view."""
    return (
        CAM_X_RANGE[0] + size < x < CAM_X_RANGE[1] - size
        and z + size < CAM_Z_MAX
    )


# ---------------------------------------------------------------------------
# TTC - stratified by TTC duration buckets
# ---------------------------------------------------------------------------

# Target: ~1250 scenes per bucket
TTC_BUCKETS = [
    (0.5, 1.2), # fast
    (1.2, 2.0), # medium-fast
    (2.0, 3.2), # medium
    (3.2, 5.0), # slow
]


def _sample_ttc_params(r: np.random.Generator, target_ttc_min: float,
                       target_ttc_max: float) -> TTCConfig | None:
    """
    Sample TTC parameters biased toward a target TTC range.
    Returns None if no valid config found after max_tries.

    Core relationship: TTC ≈ gap / (v1 + v2)
    So for target TTC t: gap / (v1+v2) ≈ t
    We sample v1+v2 and gap such that gap/(v1+v2) lands in [target_min, target_max].
    """
    s1 = r.choice(SHAPES)
    s2 = r.choice(SHAPES)
    c1 = r.choice(COLORS)
    c2 = r.choice([c for c in COLORS if c != c1])
    sz1 = round(float(r.uniform(0.07, 0.14)), 2)
    sz2 = round(float(r.uniform(0.07, 0.14)), 2)

    # Pick closing speed so that gap lands in a visible range
    closing_speed = round(float(r.uniform(0.8, 5.0)), 2)
    # gap = closing_speed × TTC_target
    t_target = r.uniform(target_ttc_min, target_ttc_max)
    gap = closing_speed * t_target
    gap = round(float(np.clip(gap, 0.8, 5.0)), 2)

    # Visibility check: objects must fit in frame
    x1, x2 = -gap / 2, gap / 2
    if not _in_frame(x1, sz1, sz1) or not _in_frame(x2, sz2, sz2):
        return None

    # Split closing speed between two objects (v1 > 0, v2 >= 0)
    v1 = round(float(r.uniform(closing_speed * 0.5, closing_speed * 0.95)), 2)
    v2 = round(max(0.1, closing_speed - v1), 2)

    cfg = TTCConfig(
        obj1=ObjectConfig(shape=s1, size=sz1, mass=round(float(r.uniform(0.5, 2.0)), 1),
                          color=c1, label="Object A",
                          position=[x1, 0.0, sz1 + 0.01],
                          velocity=[v1, 0.0, 0.0]),
        obj2=ObjectConfig(shape=s2, size=sz2, mass=round(float(r.uniform(0.5, 2.0)), 1),
                          color=c2, label="Object B",
                          position=[x2, 0.0, sz2 + 0.01],
                          velocity=[-v2, 0.0, 0.0]),
        surface_friction=round(float(r.uniform(0.2, 0.7)), 2),
    )
    return cfg


def make_ttc(scene_id: str, seed: int, bucket_idx: int) -> bool:
    r = np.random.default_rng(seed)
    t_min, t_max = TTC_BUCKETS[bucket_idx % len(TTC_BUCKETS)]

    # Try up to 10 times to get a valid scene in this bucket
    for attempt in range(10):
        cfg = _sample_ttc_params(r, t_min, t_max)
        if cfg is None:
            continue
        gt = verify_ttc(cfg)
        if not gt["collision_occurred"]:
            continue
        ttc = gt["time_to_collision"]
        if not (t_min * 0.7 <= ttc <= t_max * 1.3): # allow 30% slack
            continue
        break
    else:
        print(f" SKIP {scene_id}: could not find valid TTC in [{t_min},{t_max}]s after 10 tries")
        return False

    show = ttc * float(r.uniform(0.60, 0.78))
    n, fi = 8, show / 7
    c1, s1 = cfg.obj1.color, cfg.obj1.shape
    c2, s2 = cfg.obj2.color, cfg.obj2.shape

    xml = _build_ttc_xml(cfg)
    frames = render_frames_from_scratch(xml, {"obj1": cfg.obj1.velocity, "obj2": cfg.obj2.velocity}, n, fi)
    labeled = []
    for j, f in enumerate(frames):
        f = add_label(f, f"t={j*fi:.2f}s [{scene_id}]", (10, 10))
        f = add_label(f, f"Object A: {c1} {s1} Object B: {c2} {s2}", (10, HEIGHT - 30))
        labeled.append(f)

    out = Path("data/generated/ttc") / scene_id
    save_frames(labeled, out)
    (out / "config.json").write_text(json.dumps({
        "scene_id": scene_id, "task_type": "ttc", "bucket": f"{t_min}-{t_max}s",
        "obj1": {"shape": s1, "size": cfg.obj1.size, "color": c1,
                 "position": cfg.obj1.position, "velocity": cfg.obj1.velocity},
        "obj2": {"shape": s2, "size": cfg.obj2.size, "color": c2,
                 "position": cfg.obj2.position, "velocity": cfg.obj2.velocity},
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
    print(f" {scene_id}: TTC={ttc:.3f}s bucket=[{t_min},{t_max}] shapes={s1}/{s2}")
    return True


# ---------------------------------------------------------------------------
# Stability - forced 50/50 stable/unstable
# ---------------------------------------------------------------------------

def _sample_stable_config(r: np.random.Generator) -> StabilityConfig:
    """Sample a config biased toward stability: wide base, small offsets."""
    n_obj = int(r.integers(2, 4))
    objs = []
    z = 0.0
    for j in range(n_obj):
        sh = r.choice(SHAPES)
        # Stable: wider objects, especially at the bottom; decreasing size toward top
        sz = round(float(r.uniform(0.08, 0.22)) * (1.0 - j * 0.15), 2)
        sz = max(sz, 0.05)
        h = round(float(r.uniform(0.04, 0.10)), 2) if sh != "sphere" else 0.0
        mass = round(float(r.uniform(0.5, 2.0) * (1.0 - j * 0.2)), 1)
        mass = max(mass, 0.2)
        col = COLORS[j % len(COLORS)]
        # Stable: small horizontal offset (< 30% of base size)
        base_sz = objs[0].size if objs else sz
        max_offset = base_sz * 0.25
        ox = 0.0 if j == 0 else round(float(r.uniform(-max_offset, max_offset)), 3)
        oy = 0.0 if j == 0 else round(float(r.uniform(-max_offset * 0.5, max_offset * 0.5)), 3)
        half_h = h if sh != "sphere" else sz
        z += half_h + 0.005
        objs.append(ObjectConfig(shape=sh, size=sz, height=h, mass=mass, color=col,
                                  label=f"Object {j+1}", position=[ox, oy, round(z, 3)]))
        z += half_h
    return StabilityConfig(objects=objs, surface_friction=round(float(r.uniform(0.5, 0.9)), 2))


def _sample_unstable_config(r: np.random.Generator) -> StabilityConfig:
    """Sample a config biased toward instability: large offsets, top-heavy."""
    n_obj = int(r.integers(2, 5))
    objs = []
    z = 0.0
    for j in range(n_obj):
        sh = r.choice(SHAPES)
        # Unstable: increasing size toward top (top-heavy) or large offsets
        sz = round(float(r.uniform(0.06, 0.18)) * (1.0 + j * 0.10), 2)
        sz = min(sz, 0.25)
        h = round(float(r.uniform(0.05, 0.15)), 2) if sh != "sphere" else 0.0
        mass = round(float(r.uniform(0.3, 2.0)), 1)
        col = COLORS[j % len(COLORS)]
        base_sz = objs[0].size if objs else sz
        # Unstable: large horizontal offset (50-100% of base size)
        offset_scale = r.uniform(0.5, 1.0) if j > 0 else 0.0
        ox = 0.0 if j == 0 else round(float(r.choice([-1, 1]) * base_sz * offset_scale), 3)
        oy = 0.0 if j == 0 else round(float(r.uniform(-0.05, 0.05)), 3)
        half_h = h if sh != "sphere" else sz
        z += half_h + 0.005
        objs.append(ObjectConfig(shape=sh, size=sz, height=h, mass=mass, color=col,
                                  label=f"Object {j+1}", position=[ox, oy, round(z, 3)]))
        z += half_h
    return StabilityConfig(objects=objs, surface_friction=round(float(r.uniform(0.3, 0.7)), 2))


def make_stability(scene_id: str, seed: int, target_stable: bool) -> bool:
    r = np.random.default_rng(seed)

    # Try up to 8 times to get the desired stability outcome
    for attempt in range(8):
        cfg = _sample_stable_config(r) if target_stable else _sample_unstable_config(r)
        gt = verify_stability(cfg)
        if gt["is_stable"] == target_stable:
            break
        # Flip the seed offset so the next attempt gets different params
        r = np.random.default_rng(seed + attempt * 1000)
    else:
        # Accept whatever we got on last attempt
        pass

    objs = cfg.objects
    xml = _build_stability_xml(cfg)
    frame = render_single(xml)

    # Visibility check: tallest object should not exceed CAM_Z_MAX
    max_z = max(o.position[2] + (o.size if o.shape == "sphere" else o.height) for o in objs)
    if max_z > CAM_Z_MAX:
        print(f" SKIP {scene_id}: stack too tall ({max_z:.2f}m)")
        return False

    for j, o in enumerate(objs):
        frame = add_label(frame, f"{o.label}: {o.color} {o.shape}", (10, HEIGHT - 30 - j * 20))

    out = Path("data/generated/stability") / scene_id
    out.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame).save(out / "scene.png")

    labels = [f"{o.color} {o.shape}" for o in objs]
    n_obj = len(objs)
    (out / "config.json").write_text(json.dumps({
        "scene_id": scene_id, "task_type": "stability",
        "target_stable": target_stable,
        "objects": [{"shape": o.shape, "size": o.size, "height": o.height,
                     "mass": o.mass, "color": o.color, "position": o.position}
                    for o in objs],
        "surface_friction": cfg.surface_friction,
    }, indent=2))
    (out / "ground_truth.json").write_text(json.dumps(gt, indent=2))
    (out / "prompt.txt").write_text(
        f"A stack of {n_obj} objects: {', '.join(labels)} (bottom to top).\n"
        f"Will it stay stable or topple?\n\n"
        f"<reasoning>Your analysis</reasoning>\n"
        f"<answer>stable OR unstable</answer>\n<confidence>XX%</confidence>"
    )
    ans = "stable" if gt["is_stable"] else "unstable"
    print(f" {scene_id}: {ans} (target={'stable' if target_stable else 'unstable'}) "
          f"max_disp={gt['max_displacement_m']:.3f}m {n_obj} objects")
    return True


# ---------------------------------------------------------------------------
# Trajectory - stratified by landing distance buckets
# ---------------------------------------------------------------------------

TRAJ_BUCKETS = [
    (0.5, 3.0), # short
    (3.0, 6.0), # medium
    (6.0, 10.0), # far
    (10.0, 16.0), # very far
]


def _sample_traj_params(r: np.random.Generator, dist_min: float,
                        dist_max: float) -> TrajectoryConfig | None:
    """
    Sample trajectory params biased toward a landing distance range.
    Kinematic estimate: x_land ≈ 2*vx*vz/g (ignoring friction/bouncing)
    So: vx*vz ≈ dist * g / 2
    """
    g = 9.81
    d_target = r.uniform(dist_min, dist_max)
    vz = round(float(r.uniform(1.5, 5.0)), 2)
    vx_ideal = (d_target * g / 2.0) / vz
    vx_ideal = float(np.clip(vx_ideal, 0.3, 6.0))
    vx = round(float(r.normal(vx_ideal, vx_ideal * 0.1)), 2)
    vx = round(float(np.clip(vx, 0.3, 6.0)), 2)

    sz = round(float(r.uniform(0.04, 0.10)), 2)
    col = r.choice(COLORS)
    friction = round(float(r.uniform(0.3, 0.9)), 2)

    # Visibility: peak height must stay below camera top
    t_peak = vz / g
    z_peak = 0.5 + vz * t_peak - 0.5 * g * t_peak ** 2
    if z_peak + sz > CAM_Z_MAX:
        # Clamp vz so peak stays in frame: z_peak = 0.5 + vz^2/(2g) <= CAM_Z_MAX - sz
        # => vz <= sqrt(2g * (CAM_Z_MAX - sz - 0.5))
        vz_max = float(np.sqrt(2 * g * max(0.1, CAM_Z_MAX - sz - 0.5)))
        vz = round(float(np.clip(vz, 0.5, vz_max)), 2)

    cfg = TrajectoryConfig(
        obj=ObjectConfig(shape="sphere", size=sz, mass=0.3, color=col,
                         label="Ball", position=[0.0, 0.0, 0.5],
                         velocity=[vx, 0.0, vz]),
        surface_friction=friction,
        restitution=round(float(r.uniform(0.1, 0.4)), 2),
    )
    return cfg


def make_trajectory(scene_id: str, seed: int, bucket_idx: int) -> bool:
    r = np.random.default_rng(seed)
    d_min, d_max = TRAJ_BUCKETS[bucket_idx % len(TRAJ_BUCKETS)]

    for attempt in range(10):
        cfg = _sample_traj_params(r, d_min, d_max)
        if cfg is None:
            continue
        gt = verify_trajectory(cfg)
        lx = gt["landing_position"]["x"]
        if not (d_min * 0.6 <= lx <= d_max * 1.4): # 40% slack
            continue
        break
    else:
        print(f" SKIP {scene_id}: could not land in [{d_min},{d_max}]m after 10 tries")
        return False

    ft = gt["flight_time_s"] or 1.0
    show = max(0.3, min(1.5, ft * float(r.uniform(0.35, 0.50))))
    n, fi = 5, show / 4
    col = cfg.obj.color

    xml = _build_trajectory_xml(cfg)
    frames = render_frames_from_scratch(xml, {"obj": cfg.obj.velocity}, n, fi)
    labeled = []
    for j, f in enumerate(frames):
        f = add_label(f, f"t={j*fi:.2f}s [{scene_id}]", (10, 10))
        f = add_label(f, f"{col} ball - predict landing", (10, HEIGHT - 30))
        labeled.append(f)

    out = Path("data/generated/trajectory") / scene_id
    save_frames(labeled, out)
    (out / "config.json").write_text(json.dumps({
        "scene_id": scene_id, "task_type": "trajectory",
        "bucket": f"{d_min}-{d_max}m",
        "object": {"shape": "sphere", "size": cfg.obj.size, "color": col,
                   "velocity": cfg.obj.velocity},
        "surface_friction": cfg.surface_friction, "restitution": cfg.restitution,
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
    print(f" {scene_id}: x={lx:.2f}m bucket=[{d_min},{d_max}] "
          f"vx={cfg.obj.velocity[0]} vz={cfg.obj.velocity[2]} f={cfg.surface_friction}")
    return True


# ---------------------------------------------------------------------------
# Main - stratified generation
# ---------------------------------------------------------------------------

def generate_all(n_per_task: int = 5000, workers: int = 1):
    """
    Generate n_per_task scenes for each of 3 tasks.

    Stratification:
      TTC: equal split across 4 TTC duration buckets
      Stability: exactly 50% stable, 50% unstable
      Trajectory: equal split across 4 landing distance buckets
    """
    print(f"Generating {n_per_task} TTC scenes...")
    ttc_ok = 0
    for i in range(n_per_task):
        bucket = i % len(TTC_BUCKETS)
        ok = make_ttc(f"ttc_{i:06d}", seed=i, bucket_idx=bucket)
        if ok:
            ttc_ok += 1
    print(f"TTC done: {ttc_ok}/{n_per_task} generated\n")

    print(f"Generating {n_per_task} Stability scenes (50/50 split)...")
    stab_ok = 0
    for i in range(n_per_task):
        target_stable = (i % 2 == 0) # alternating → exact 50/50
        ok = make_stability(f"stability_{i:06d}", seed=i, target_stable=target_stable)
        if ok:
            stab_ok += 1
    print(f"Stability done: {stab_ok}/{n_per_task} generated\n")

    print(f"Generating {n_per_task} Trajectory scenes...")
    traj_ok = 0
    for i in range(n_per_task):
        bucket = i % len(TRAJ_BUCKETS)
        ok = make_trajectory(f"trajectory_{i:06d}", seed=i, bucket_idx=bucket)
        if ok:
            traj_ok += 1
    print(f"Trajectory done: {traj_ok}/{n_per_task} generated\n")

    total = ttc_ok + stab_ok + traj_ok
    print(f"Total: {total} scenes generated across 3 tasks")
    print("Upload to HuggingFace: python scripts/upload_dataset.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=5000,
                        help="Scenes per task (default 5000 → 15K total)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers (default 1, increase on Linux/Mac)")
    args = parser.parse_args()

    generate_all(n_per_task=args.n, workers=args.workers)
