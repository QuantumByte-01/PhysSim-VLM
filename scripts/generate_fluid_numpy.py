"""
PhysSim-VLM: Numpy/Scipy Fluid Simulation Generator
=====================================================
Pure numpy Verlet particle simulation - no Taichi, no SPH pressure, no NaN.

Visual style: smooth continuous fluid rendered via scipy gaussian density field.
Each particle contributes to a 2-D density grid; gaussian_filter smooths it
into a connected fluid body.

Libraries:
  numpy - particle physics (Verlet integration, gravity, damping)
  scipy - gaussian density smoothing (gaussian_filter) for fluid look
  PIL - wall/label drawing, frame saving
  matplotlib - composite viscosity side-by-side renders

Tasks:
  fluid_direction - block of fluid flows left / right / down
  fluid_viscosity - two fluids side by side, different damping → different spread

fluid_level is kept in generate_taichi_fluid.py (already fixed + generated).

Usage:
  python scripts/generate_fluid_numpy.py --n 500 --tasks fluid_direction
  python scripts/generate_fluid_numpy.py --n 350 --tasks fluid_viscosity
  python scripts/generate_fluid_numpy.py --n 500 --tasks all
  python scripts/generate_fluid_numpy.py --validate --tasks fluid_direction
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter

# ── Paths ──────────────────────────────────────────────────────────────────
OUT_DIR = Path(__file__).parent.parent / "data" / "sft_r2"

# ── Physical constants ──────────────────────────────────────────────────────
DOMAIN_W = 2.0 # m - domain width
DOMAIN_H = 1.5 # m - domain height
PARTICLE_R = 0.01 # m - visual particle radius
RESTITUTION = 0.25 # coefficient of restitution on wall bounce
DT = 0.005 # s - integration timestep
GRAVITY = 9.81 # m/s²

# ── Image constants ─────────────────────────────────────────────────────────
IMG_W, IMG_H = 640, 480
BG_COLOR = (240, 240, 240)
WALL_COLOR = (100, 100, 100)
FLUID_BLUE = np.array([60, 120, 255], dtype=float)
FLUID_ORANGE = np.array([255, 100, 60], dtype=float)

# Gaussian blur sigma in pixels (makes discrete particles look like fluid)
GAUSS_SIGMA = 7


# ═══════════════════════════════════════════════════════════════════════════
# Particle simulation - pure numpy Verlet, no SPH pressure
# ═══════════════════════════════════════════════════════════════════════════

def make_block(cx: float, cy: float, w: float, h: float,
               n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return (positions, velocities) for N particles in a rectangle."""
    nx = max(int(math.sqrt(n * w / h)), 1)
    ny = max(n // nx, 1)
    xs = np.linspace(cx - w / 2 + PARTICLE_R, cx + w / 2 - PARTICLE_R, nx)
    ys = np.linspace(cy - h / 2 + PARTICLE_R, cy + h / 2 - PARTICLE_R, ny)
    xx, yy = np.meshgrid(xs, ys)
    pos = np.column_stack([xx.ravel(), yy.ravel()])[:n].copy()
    # tiny jitter so particles aren't perfectly on-grid
    pos += rng.normal(0, PARTICLE_R * 0.25, pos.shape)
    vel = np.zeros_like(pos)
    return pos, vel


def step_particles(pos: np.ndarray, vel: np.ndarray,
                   gx: float, gy: float, damping: float,
                   left_wall: float = 0.0, right_wall: float = DOMAIN_W,
                   domain_h: float = DOMAIN_H) -> None:
    """In-place Verlet step. No SPH - just gravity + damping + wall bounce."""
    # Gravity
    vel[:, 0] += gx * DT
    vel[:, 1] += gy * DT

    # Viscosity damping (exponential decay of velocity)
    vel *= max(0.0, 1.0 - damping * DT)

    # Integrate
    pos += vel * DT

    # -- Boundary: left container wall
    mask = pos[:, 0] < left_wall + PARTICLE_R
    pos[mask, 0] = left_wall + PARTICLE_R
    vel[mask, 0] = np.abs(vel[mask, 0]) * RESTITUTION

    # -- Boundary: right container wall
    mask = pos[:, 0] > right_wall - PARTICLE_R
    pos[mask, 0] = right_wall - PARTICLE_R
    vel[mask, 0] = -np.abs(vel[mask, 0]) * RESTITUTION

    # -- Boundary: floor
    mask = pos[:, 1] < PARTICLE_R
    pos[mask, 1] = PARTICLE_R
    vel[mask, 1] = np.abs(vel[mask, 1]) * RESTITUTION

    # -- Boundary: ceiling
    mask = pos[:, 1] > domain_h - PARTICLE_R
    pos[mask, 1] = domain_h - PARTICLE_R
    vel[mask, 1] = -np.abs(vel[mask, 1]) * RESTITUTION


def simulate(pos: np.ndarray, vel: np.ndarray,
             gx: float, gy: float, damping: float,
             n_steps: int, capture_steps: list[int],
             left_wall: float = 0.0, right_wall: float = DOMAIN_W,
             domain_h: float = DOMAIN_H) -> dict[int, np.ndarray]:
    """Run particle sim, return dict {step: positions_copy}."""
    captured = {}
    if 0 in capture_steps:
        captured[0] = pos.copy()
    for s in range(1, n_steps + 1):
        step_particles(pos, vel, gx, gy, damping, left_wall, right_wall, domain_h)
        if s in capture_steps:
            captured[s] = pos.copy()
    return captured


# ═══════════════════════════════════════════════════════════════════════════
# Rendering - gaussian density field gives smooth fluid look
# ═══════════════════════════════════════════════════════════════════════════

def _pos_to_pixels(pos: np.ndarray,
                   domain_w: float = DOMAIN_W,
                   domain_h: float = DOMAIN_H) -> tuple[np.ndarray, np.ndarray]:
    """Convert physical (x,y) to pixel (px, py) with y-flip."""
    px = np.clip((pos[:, 0] / domain_w * IMG_W).astype(int), 0, IMG_W - 1)
    py = np.clip(((1 - pos[:, 1] / domain_h) * IMG_H).astype(int), 0, IMG_H - 1)
    return px, py


def render_particles_pil(pos: np.ndarray, color: tuple = (60, 120, 255),
                         domain_w: float = DOMAIN_W,
                         domain_h: float = DOMAIN_H,
                         radius: int = 3) -> np.ndarray:
    """
    Render particles as individual filled circles (PIL).
    Same visual style as original Taichi code - clear and visible.
    Used for fluid_direction where motion path matters.
    """
    img = Image.new("RGB", (IMG_W, IMG_H), BG_COLOR)
    draw = ImageDraw.Draw(img)
    for p in pos:
        px = int(np.clip(p[0] / domain_w * IMG_W, 0, IMG_W - 1))
        py = int(np.clip((1 - p[1] / domain_h) * IMG_H, 0, IMG_H - 1))
        draw.ellipse([px - radius, py - radius, px + radius, py + radius],
                     fill=color)
    return np.array(img)


def render_density(pos: np.ndarray, color: np.ndarray = FLUID_BLUE,
                   domain_w: float = DOMAIN_W, domain_h: float = DOMAIN_H,
                   sigma: float = GAUSS_SIGMA) -> np.ndarray:
    """
    Render particle positions as a smooth continuous density field.

    1. Scatter particles into a 2-D density grid.
    2. Apply gaussian_filter to create a smooth fluid body.
    3. Colorize against the background.
    Used for fluid_viscosity where compact vs spread shape matters.
    """
    density = np.zeros((IMG_H, IMG_W), dtype=np.float32)
    px, py = _pos_to_pixels(pos, domain_w, domain_h)
    np.add.at(density, (py, px), 1.0)
    density = gaussian_filter(density, sigma=sigma)
    # Normalize: peak ~1.0
    peak = density.max()
    if peak > 0:
        density = density / peak
    density = np.clip(density, 0.0, 1.0)

    bg = np.array(BG_COLOR, dtype=float)
    img = np.ones((IMG_H, IMG_W, 3), dtype=float) * bg
    for c in range(3):
        img[:, :, c] = img[:, :, c] * (1 - density) + color[c] * density
    return img.clip(0, 255).astype(np.uint8)


def render_two_density(pos_a: np.ndarray, pos_b: np.ndarray,
                       domain_w: float = DOMAIN_W,
                       domain_h: float = DOMAIN_H) -> np.ndarray:
    """Render two fluids in blue and orange on the same frame."""
    dens_a = np.zeros((IMG_H, IMG_W), dtype=np.float32)
    dens_b = np.zeros((IMG_H, IMG_W), dtype=np.float32)
    pax, pay = _pos_to_pixels(pos_a, domain_w, domain_h)
    pbx, pby = _pos_to_pixels(pos_b, domain_w, domain_h)
    np.add.at(dens_a, (pay, pax), 1.0)
    np.add.at(dens_b, (pby, pbx), 1.0)
    dens_a = gaussian_filter(dens_a, sigma=GAUSS_SIGMA)
    dens_b = gaussian_filter(dens_b, sigma=GAUSS_SIGMA)

    peak_a = dens_a.max()
    peak_b = dens_b.max()
    if peak_a > 0: dens_a /= peak_a
    if peak_b > 0: dens_b /= peak_b

    bg = np.array(BG_COLOR, dtype=float)
    img = np.ones((IMG_H, IMG_W, 3), dtype=float) * bg
    for c in range(3):
        img[:, :, c] = (img[:, :, c]
                        * (1 - dens_a) * (1 - dens_b)
                        + FLUID_BLUE[c] * dens_a * (1 - dens_b)
                        + FLUID_ORANGE[c] * dens_b * (1 - dens_a)
                        + 0.5 * (FLUID_BLUE[c] + FLUID_ORANGE[c]) * dens_a * dens_b)
    return img.clip(0, 255).astype(np.uint8)


def add_walls(img: np.ndarray, left_wall: float, right_wall: float,
              domain_w: float = DOMAIN_W, domain_h: float = DOMAIN_H) -> np.ndarray:
    """Draw container walls and floor on an image."""
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    # Floor
    draw.rectangle([0, IMG_H - 4, IMG_W, IMG_H], fill=WALL_COLOR)
    # Left wall
    lx = int(left_wall / domain_w * IMG_W)
    draw.rectangle([lx - 4, 0, lx, IMG_H], fill=WALL_COLOR)
    # Right wall
    rx = int(right_wall / domain_w * IMG_W)
    draw.rectangle([rx, 0, rx + 4, IMG_H], fill=WALL_COLOR)
    return np.array(pil)


def add_text(img: np.ndarray, text: str,
             pos: tuple = (10, 10), color=(0, 0, 0)) -> np.ndarray:
    pil = Image.fromarray(img)
    ImageDraw.Draw(pil).text(pos, text, fill=color)
    return np.array(pil)


def save_frames(frames: list[np.ndarray], out_dir: Path):
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        Image.fromarray(f).save(frames_dir / f"frame_{i:03d}.png")


# ═══════════════════════════════════════════════════════════════════════════
# Task: Fluid Direction
# ═══════════════════════════════════════════════════════════════════════════
# Gravity is tilted so that the fluid clearly flows left / right / down.
# The label is deterministic from the config (no post-hoc measurement needed).
#
# gx > 0 → rightward force → answer = "right"
# gx < 0 → leftward force → answer = "left"
# gx = 0 → pure downward → answer = "down"
# ──────────────────────────────────────────────────────────────────────────

_DIR_CONFIGS = [
    # (label, gx, gy, n_range, damping)
    ("left", -6.0, -4.0, (500, 900), 0.15),
    ("right", 6.0, -4.0, (500, 900), 0.15),
    ("down", 0.0, -9.81, (600, 1000), 0.15),
]

# Shorter domain height (1.0m) so settled fluid fills more of the image
_DIR_DOMAIN_H = 1.0


def make_fluid_direction(scene_id: str, seed: int) -> bool:
    rng = np.random.default_rng(seed)
    label, gx, gy, n_range, damping = _DIR_CONFIGS[seed % len(_DIR_CONFIGS)]

    n = int(rng.integers(*n_range))
    # Block starts near the top of the shorter domain
    cy0 = 0.80
    pos, vel = make_block(1.0, cy0, 0.35, 0.30, n, rng)
    # Clamp to domain height
    pos[:, 1] = np.clip(pos[:, 1], PARTICLE_R, _DIR_DOMAIN_H - PARTICLE_R)

    n_steps = 1200
    # Capture more frames early (fluid in motion) then at the end (settled)
    captures = sorted(set([0, 80, 180, 320, 600, 1000, 1200]))

    def step_dir(ps, vs):
        step_particles(ps, vs, gx, gy, damping,
                       left_wall=0.0, right_wall=DOMAIN_W)
        # Clamp ceiling to short domain
        mask = ps[:, 1] > _DIR_DOMAIN_H - PARTICLE_R
        ps[mask, 1] = _DIR_DOMAIN_H - PARTICLE_R
        vs[mask, 1] = -abs(vs[mask, 1]) * RESTITUTION

    capt: dict[int, np.ndarray] = {}
    if 0 in captures:
        capt[0] = pos.copy()
    for s in range(1, n_steps + 1):
        step_dir(pos, vel)
        if s in captures:
            capt[s] = pos.copy()

    final_pos = capt[n_steps]

    frames = []
    for step in sorted(capt.keys()):
        p = capt[step]
        t = step * DT
        img = render_particles_pil(p, color=(60, 120, 255),
                                   domain_h=_DIR_DOMAIN_H)
        img = add_walls(img, 0.0, DOMAIN_W, domain_h=_DIR_DOMAIN_H)
        img = add_text(img, f"t={t:.2f}s [{scene_id}]", (10, 10))
        frames.append(img)

    # Compute displacement for metadata
    init_cx = 1.0
    valid_mask = np.ones(len(final_pos), dtype=bool) # no NaN with Verlet
    final_cx = float(np.mean(final_pos[valid_mask, 0]))
    final_cy = float(np.mean(final_pos[valid_mask, 1]))
    dx = final_cx - init_cx
    dy = final_cy - 0.85

    out = OUT_DIR / "fluid_direction" / scene_id
    save_frames(frames, out)

    gt = {
        "answer": label,
        "initial_center": [round(init_cx, 3), 0.85],
        "final_center": [round(final_cx, 3), round(final_cy, 3)],
        "displacement": [round(dx, 3), round(dy, 3)],
        "gx": gx, "gy": gy,
    }
    (out / "config.json").write_text(json.dumps({
        "scene_id": scene_id, "task_type": "fluid_direction",
        "n_particles": n, "gx": gx, "gy": gy, "damping": damping,
    }, indent=2))
    (out / "ground_truth.json").write_text(json.dumps(gt, indent=2))
    (out / "prompt.txt").write_text(
        "A block of fluid (shown in blue) is released.\n"
        "Watch how the fluid flows across the 6 frames.\n\n"
        "In which primary direction does the fluid move?\n\n"
        "Options: left, right, down\n\n"
        "<reasoning>Observe the bulk motion of the fluid mass</reasoning>\n"
        "<answer>left/right/down</answer>"
    )
    (out / "assistant_text.txt").write_text(
        f"<reasoning>The fluid starts centered at x={init_cx:.1f}m. "
        f"After simulation, the center of mass moves to ({final_cx:.2f}, {final_cy:.2f})m "
        f"(displacement: dx={dx:.2f}m, dy={dy:.2f}m). "
        f"The dominant horizontal motion is {label}.</reasoning>\n"
        f"<answer>{label}</answer>"
    )

    print(f" {scene_id}: {label:5s} dx={dx:+.2f} dy={dy:+.2f} n={n}")
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Task: Fluid Viscosity Comparison
# ═══════════════════════════════════════════════════════════════════════════
# Two blobs released side by side with OUTWARD radial velocity (simulates
# fluid splashing / pouring onto a floor).
#
# High viscosity = high damping → outward velocity dies quickly → compact pile
# Low viscosity = low damping → outward velocity persists → spreads wide flat
#
# Damping ranges chosen so the spread difference is visually unmistakable:
# damping=25 (high) → spreads ~0.1m from center after stopping
# damping=0.3 (low) → spreads to domain walls (~0.9m from center)
#
# Ground truth: which side (A=blue/left B=orange/right) is more viscous.
# ──────────────────────────────────────────────────────────────────────────

_VISC_RANGES = {
    "low": (0.2, 0.6), # water / thin fluid
    "high": (18.0, 35.0), # honey / thick fluid (stops in ~0.3s)
}

_OUTWARD_SPEED = 4.0 # m/s radial kick given to all particles at t=0


def _add_outward_velocity(pos: np.ndarray, vel: np.ndarray,
                          cx: float, cy: float, speed: float) -> None:
    """In-place: give each particle velocity pointing away from (cx, cy)."""
    dx = pos[:, 0] - cx
    dy = pos[:, 1] - cy
    dist = np.sqrt(dx ** 2 + dy ** 2)
    dist = np.where(dist < 1e-6, 1e-6, dist)
    vel[:, 0] += speed * dx / dist
    vel[:, 1] += speed * dy / dist


def make_fluid_viscosity(scene_id: str, seed: int) -> bool:
    rng = np.random.default_rng(seed)

    damping_low = round(float(rng.uniform(*_VISC_RANGES["low"])), 2)
    damping_high = round(float(rng.uniform(*_VISC_RANGES["high"])), 2)

    if rng.random() > 0.5:
        damping_a, damping_b = damping_low, damping_high
        answer = "fluid b"
    else:
        damping_a, damping_b = damping_high, damping_low
        answer = "fluid a"

    n = int(rng.integers(400, 700))
    cx_a, cx_b, cy = 0.5, 1.5, 0.65 # lower start so fluid is visible in 1.0m domain

    # Start particles in compact blocks with outward velocity
    pos_a, vel_a = make_block(cx_a, cy, 0.20, 0.20, n, rng)
    pos_b, vel_b = make_block(cx_b, cy, 0.20, 0.20, n, rng)
    # Clamp to 1.0m domain
    for p in (pos_a, pos_b):
        p[:, 1] = np.clip(p[:, 1], PARTICLE_R, 1.0 - PARTICLE_R)
    _add_outward_velocity(pos_a, vel_a, cx_a, cy, _OUTWARD_SPEED)
    _add_outward_velocity(pos_b, vel_b, cx_b, cy, _OUTWARD_SPEED)

    # No side walls within each half - let fluid spread freely
    n_steps = 800
    captures = list(range(0, n_steps + 1, n_steps // 5))

    capt_a = simulate(pos_a, vel_a, 0.0, -GRAVITY, damping_a, n_steps, captures,
                      left_wall=0.0, right_wall=DOMAIN_W / 2 - 0.02, domain_h=1.0)
    capt_b = simulate(pos_b, vel_b, 0.0, -GRAVITY, damping_b, n_steps, captures,
                      left_wall=DOMAIN_W / 2 + 0.02, right_wall=DOMAIN_W, domain_h=1.0)

    def x_spread(pos, cx):
        return float(np.max(np.abs(pos[:, 0] - cx)))

    spread_a = x_spread(capt_a[n_steps], cx_a)
    spread_b = x_spread(capt_b[n_steps], cx_b)

    # Shorter domain so settled fluid fills more vertical space
    visc_domain_h = 1.0

    frames = []
    for step in captures:
        t = step * DT
        # Render A (blue) and B (orange) onto same image using PIL dots
        img_a = render_particles_pil(capt_a[step], color=(60, 120, 255),
                                     domain_h=visc_domain_h)
        img_b = render_particles_pil(capt_b[step], color=(255, 100, 60),
                                     domain_h=visc_domain_h)
        # Composite: use img_a as base, overlay img_b non-bg pixels
        bg = np.array(BG_COLOR, dtype=np.uint8)
        mask_b = ~np.all(img_b == bg, axis=2)
        img_a[mask_b] = img_b[mask_b]
        img = img_a
        img = add_walls(img, 0.0, DOMAIN_W, domain_h=visc_domain_h)
        # Center divider
        pil = Image.fromarray(img)
        draw = ImageDraw.Draw(pil)
        draw.line([(IMG_W // 2, 0), (IMG_W // 2, IMG_H - 4)],
                  fill=(160, 160, 160), width=1)
        img = np.array(pil)
        img = add_text(img,
                       f"t={t:.2f}s A (blue, left) B (orange, right) [{scene_id}]",
                       (10, 10))
        frames.append(img)

    out = OUT_DIR / "fluid_viscosity" / scene_id
    save_frames(frames, out)

    gt = {
        "answer": answer,
        "damping_a": damping_a,
        "damping_b": damping_b,
        "spread_a": round(spread_a, 4),
        "spread_b": round(spread_b, 4),
    }
    (out / "config.json").write_text(json.dumps({
        "scene_id": scene_id, "task_type": "fluid_viscosity",
        "n_particles": n, "damping_a": damping_a, "damping_b": damping_b,
    }, indent=2))
    (out / "ground_truth.json").write_text(json.dumps(gt, indent=2))
    (out / "prompt.txt").write_text(
        "Two fluids are released from the same point with the same force:\n"
        " - Fluid A (blue, left side)\n"
        " - Fluid B (orange, right side)\n\n"
        "A more viscous fluid resists flow - it stops quickly and stays compact. "
        "A less viscous fluid keeps moving and spreads out wide and flat.\n\n"
        "Which fluid is more viscous?\n\n"
        "<reasoning>Compare how far each fluid has spread from its starting point</reasoning>\n"
        "<answer>fluid a or fluid b</answer>"
    )
    more_visc_damp = damping_a if answer == "fluid a" else damping_b
    less_visc_damp = damping_b if answer == "fluid a" else damping_a
    (out / "assistant_text.txt").write_text(
        f"<reasoning>Both fluids start with the same outward velocity. "
        f"The more viscous fluid (high damping={more_visc_damp:.1f}) loses velocity quickly "
        f"and remains compact (spread={spread_a if answer=='fluid a' else spread_b:.3f}m). "
        f"The less viscous fluid (low damping={less_visc_damp:.1f}) "
        f"keeps spreading (spread={spread_b if answer=='fluid a' else spread_a:.3f}m). "
        f"The compact fluid is {answer}.</reasoning>\n"
        f"<answer>{answer}</answer>"
    )

    print(f" {scene_id}: {answer} "
          f"d_a={damping_a:.1f} d_b={damping_b:.1f} "
          f"spread_a={spread_a:.3f} spread_b={spread_b:.3f}")
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

ALL_TASKS = ["fluid_direction", "fluid_viscosity"]

TASK_GENERATORS = {
    "fluid_direction": make_fluid_direction,
    "fluid_viscosity": make_fluid_viscosity,
}


def generate_all(n_per_task: int, tasks: list[str], seed: int = 42,
                 start_idx: int = 0) -> int:
    total = 0
    for task in tasks:
        gen = TASK_GENERATORS[task]
        out_dir = OUT_DIR / task
        out_dir.mkdir(parents=True, exist_ok=True)
        succeeded = 0
        attempts = 0
        idx = start_idx
        while succeeded < n_per_task:
            scene_id = f"{task}_{idx:06d}"
            if (out_dir / scene_id / "ground_truth.json").exists():
                idx += 1
                continue
            ok = gen(scene_id, seed=seed + idx)
            attempts += 1
            if ok:
                succeeded += 1
            idx += 1
            if attempts > n_per_task * 3:
                print(f"WARNING: too many attempts for {task}, stopping at {succeeded}")
                break
        print(f"[{task}] Generated {succeeded}/{n_per_task} scenes")
        total += succeeded
    return total


def validate(tasks: list[str]) -> None:
    """Quick smoke test: generate 3 scenes per task and show stats."""
    for task in tasks:
        gen = TASK_GENERATORS[task]
        print(f"\n-- {task} --")
        answers = {}
        for i in range(9):
            sid = f"{task}_validate_{i:03d}"
            gen(sid, seed=i)
            gt_path = OUT_DIR / task / sid / "ground_truth.json"
            if gt_path.exists():
                gt = json.loads(gt_path.read_text())
                a = gt.get("answer", "?")
                answers[a] = answers.get(a, 0) + 1
        print(f" label distribution: {answers}")


def main():
    parser = argparse.ArgumentParser(description="Generate fluid scenes (numpy/scipy)")
    parser.add_argument("--n", type=int, default=500,
                        help="Scenes per task")
    parser.add_argument("--tasks", default="all",
                        help="Comma-separated tasks or 'all'")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start", type=int, default=0,
                        help="Starting scene index (for resuming)")
    parser.add_argument("--validate", action="store_true",
                        help="Run quick smoke test (9 scenes per task)")
    args = parser.parse_args()

    tasks = ALL_TASKS if args.tasks == "all" else [t.strip() for t in args.tasks.split(",")]
    unknown = [t for t in tasks if t not in TASK_GENERATORS]
    if unknown:
        print(f"ERROR: unknown tasks: {unknown}")
        print(f"Valid tasks: {ALL_TASKS}")
        sys.exit(1)

    if args.validate:
        validate(tasks)
    else:
        total = generate_all(args.n, tasks, seed=args.seed, start_idx=args.start)
        print(f"\nTotal scenes generated: {total}")


if __name__ == "__main__":
    main()
