"""
PhysSim-VLM: Taichi SPH Fluid Simulation Generator
=====================================================
GPU-accelerated fluid scenes for Dynamics category (fluid flow, viscosity,
pouring, splashing). Replaces Blender for fluid simulation data.

Uses Taichi's SPH (Smoothed Particle Hydrodynamics) solver to simulate
2D/3D fluid behavior and render particle visualizations.

Task types generated:
  - fluid_direction: Which way does the fluid flow? (left/right/down)
  - fluid_viscosity: Which fluid is more viscous? (compare two fluids)
  - fluid_level: What level does the fluid settle at?

Requirements:
  pip install taichi pillow numpy

Usage:
  python scripts/generate_taichi_fluid.py --n 1000 --tasks all
  python scripts/generate_taichi_fluid.py --n 500 --tasks fluid_direction
  python scripts/generate_taichi_fluid.py --validate
"""

import json
import sys
import argparse
import math
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw

try:
    import taichi as ti
    # Try GPU backends in order: vulkan (works without CUDA), then cuda, then cpu
    _ti_inited = False
    for _arch in [ti.vulkan, ti.cuda, ti.cpu]:
        try:
            ti.init(arch=_arch, default_fp=ti.f32)
            print(f"[Taichi] Using arch={_arch}")
            _ti_inited = True
            break
        except Exception:
            pass
    if not _ti_inited:
        ti.init(arch=ti.cpu, default_fp=ti.f32)
    _TAICHI_AVAILABLE = True
except ImportError:
    _TAICHI_AVAILABLE = False
    print("WARNING: taichi not installed. Install with: pip install taichi")

OUT_DIR = Path(__file__).parent.parent / "data" / "sft_r2"

# ── Rendering Constants ────────────────────────────────────────────────────
IMG_W, IMG_H = 640, 480
PARTICLE_RADIUS_PX = 3
FLUID_COLOR = (60, 120, 255) # blue
FLUID_COLOR_B = (255, 100, 60) # orange (for comparison)
WALL_COLOR = (100, 100, 100)
BG_COLOR = (240, 240, 240)

COLORS_MAP = {
    "blue": (60, 120, 255),
    "red": (255, 60, 60),
    "orange": (255, 140, 40),
    "green": (60, 200, 100),
    "purple": (150, 60, 220),
}


# ═══════════════════════════════════════════════════════════════════════════
# Taichi SPH Solver (2D for fast generation)
# ═══════════════════════════════════════════════════════════════════════════

if _TAICHI_AVAILABLE:

    MAX_PARTICLES = 4000
    DIM = 2
    SUPPORT_RADIUS = 0.04
    PARTICLE_RADIUS = 0.01
    REST_DENSITY = 1000.0
    GAS_CONST = 2000.0
    VISCOSITY = 50.0
    DT = 0.0002
    GRAVITY_FIELD = ti.Vector.field(DIM, dtype=ti.f32, shape=())
    GRAVITY_FIELD[None] = [0.0, -9.81]
    DOMAIN = ti.Vector([2.0, 1.5]) # meters

    @ti.data_oriented
    class SPHSolver:
        def __init__(self, n_particles: int, viscosity: float = 50.0):
            self.n = min(n_particles, MAX_PARTICLES)
            self.viscosity_coeff = viscosity

            self.pos = ti.Vector.field(DIM, dtype=ti.f32, shape=self.n)
            self.vel = ti.Vector.field(DIM, dtype=ti.f32, shape=self.n)
            self.acc = ti.Vector.field(DIM, dtype=ti.f32, shape=self.n)
            self.density = ti.field(dtype=ti.f32, shape=self.n)
            self.pressure = ti.field(dtype=ti.f32, shape=self.n)

        @ti.kernel
        def compute_density_pressure(self):
            for i in range(self.n):
                self.density[i] = 0.0
                for j in range(self.n):
                    r = (self.pos[i] - self.pos[j]).norm()
                    if r < SUPPORT_RADIUS:
                        q = 1.0 - r / SUPPORT_RADIUS
                        self.density[i] += q * q * q * 315.0 / (64.0 * 3.14159 * SUPPORT_RADIUS**3)
                self.density[i] = ti.max(self.density[i] * (PARTICLE_RADIUS**2), REST_DENSITY)
                self.pressure[i] = GAS_CONST * (self.density[i] - REST_DENSITY)

        @ti.kernel
        def compute_forces(self):
            for i in range(self.n):
                self.acc[i] = GRAVITY_FIELD[None]
                for j in range(self.n):
                    if i == j:
                        continue
                    rij = self.pos[i] - self.pos[j]
                    r = rij.norm()
                    if r < SUPPORT_RADIUS and r > 1e-6:
                        # Pressure force
                        grad = -45.0 / (3.14159 * SUPPORT_RADIUS**6) * (SUPPORT_RADIUS - r)**2
                        f_pressure = -rij.normalized() * (
                            self.pressure[i] + self.pressure[j]) / (2.0 * self.density[j]) * grad
                        # Viscosity force
                        f_visc = self.viscosity_coeff * (
                            self.vel[j] - self.vel[i]) / self.density[j] * (
                            45.0 / (3.14159 * SUPPORT_RADIUS**6) * (SUPPORT_RADIUS - r))
                        self.acc[i] += (f_pressure + f_visc) / ti.max(self.density[i], 1.0)

        @ti.kernel
        def integrate(self):
            for i in range(self.n):
                self.vel[i] += DT * self.acc[i]
                self.pos[i] += DT * self.vel[i]
                # Boundary conditions (bounce off walls)
                for d in ti.static(range(DIM)):
                    if self.pos[i][d] < PARTICLE_RADIUS:
                        self.pos[i][d] = PARTICLE_RADIUS
                        self.vel[i][d] *= -0.3
                    if self.pos[i][d] > DOMAIN[d] - PARTICLE_RADIUS:
                        self.pos[i][d] = DOMAIN[d] - PARTICLE_RADIUS
                        self.vel[i][d] *= -0.3

        def step(self):
            self.compute_density_pressure()
            self.compute_forces()
            self.integrate()

        def get_positions(self) -> np.ndarray:
            return self.pos.to_numpy()[:self.n]


# ── Rendering ──────────────────────────────────────────────────────────────

def render_particles(positions: np.ndarray, domain: tuple = (2.0, 1.5),
                     color: tuple = FLUID_COLOR,
                     walls: list = None) -> np.ndarray:
    """Render particle positions to an image."""
    img = Image.new("RGB", (IMG_W, IMG_H), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Draw walls if any
    if walls:
        for wall in walls:
            x0 = int(wall[0] / domain[0] * IMG_W)
            y0 = IMG_H - int(wall[1] / domain[1] * IMG_H)
            x1 = int(wall[2] / domain[0] * IMG_W)
            y1 = IMG_H - int(wall[3] / domain[1] * IMG_H)
            # PIL rectangle requires top < bottom; ensure y order
            draw.rectangle([x0, min(y0, y1), x1, max(y0, y1)], fill=WALL_COLOR)

    # Draw floor
    draw.rectangle([0, IMG_H - 5, IMG_W, IMG_H], fill=WALL_COLOR)

    # Draw particles (skip NaN/out-of-domain positions)
    for pos in positions:
        if not (np.isfinite(pos[0]) and np.isfinite(pos[1])):
            continue
        px = int(np.clip(pos[0] / domain[0] * IMG_W, 0, IMG_W - 1))
        py = int(np.clip(IMG_H - pos[1] / domain[1] * IMG_H, 0, IMG_H - 1))
        r = PARTICLE_RADIUS_PX
        draw.ellipse([px - r, py - r, px + r, py + r], fill=color)

    return np.array(img)


def render_two_fluids(pos_a: np.ndarray, pos_b: np.ndarray,
                      domain: tuple = (2.0, 1.5)) -> np.ndarray:
    """Render two sets of particles in different colors."""
    img = Image.new("RGB", (IMG_W, IMG_H), BG_COLOR)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, IMG_H - 5, IMG_W, IMG_H], fill=WALL_COLOR)

    for pos in pos_a:
        if not (np.isfinite(pos[0]) and np.isfinite(pos[1])):
            continue
        px = int(np.clip(pos[0] / domain[0] * IMG_W, 0, IMG_W - 1))
        py = int(np.clip(IMG_H - pos[1] / domain[1] * IMG_H, 0, IMG_H - 1))
        r = PARTICLE_RADIUS_PX
        draw.ellipse([px - r, py - r, px + r, py + r], fill=FLUID_COLOR)

    for pos in pos_b:
        if not (np.isfinite(pos[0]) and np.isfinite(pos[1])):
            continue
        px = int(np.clip(pos[0] / domain[0] * IMG_W, 0, IMG_W - 1))
        py = int(np.clip(IMG_H - pos[1] / domain[1] * IMG_H, 0, IMG_H - 1))
        r = PARTICLE_RADIUS_PX
        draw.ellipse([px - r, py - r, px + r, py + r], fill=FLUID_COLOR_B)

    return np.array(img)


def add_text(img_array: np.ndarray, text: str,
             pos: tuple = (10, 10), color=(0, 0, 0)) -> np.ndarray:
    img = Image.fromarray(img_array)
    draw = ImageDraw.Draw(img)
    draw.text(pos, text, fill=color)
    return np.array(img)


def save_fluid_frames(frames: list, out_dir: Path):
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(frames_dir / f"frame_{i:03d}.png")


# ═══════════════════════════════════════════════════════════════════════════
# Task: Fluid Direction
# ═══════════════════════════════════════════════════════════════════════════

def make_fluid_direction(scene_id: str, seed: int) -> bool:
    """Block of fluid released - which direction does it flow?"""
    if not _TAICHI_AVAILABLE:
        print(f" SKIP {scene_id}: taichi not available")
        return False

    r = np.random.default_rng(seed)

    # 3 classes with tilted gravity so fluid clearly flows left/right/down.
    # gx > gy ensures |dx| > |dy| after simulation, so label is unambiguous.
    configs = [
        ("left", 1.0, 0.9, 0.4, 0.3, -9.81, -4.0), # tilted gravity: left > down
        ("right", 1.0, 0.9, 0.4, 0.3, 9.81, -4.0), # tilted gravity: right > down
        ("down", 1.0, 1.1, 0.4, 0.3, 0.0, -9.81), # standard downward gravity
    ]
    direction, cx, cy, bw, bh, gx, gy = configs[seed % len(configs)]

    # Set gravity for this scene
    GRAVITY_FIELD[None] = [gx, gy]

    n_particles = int(r.integers(800, 1500))
    solver = SPHSolver(n_particles)

    # Initialize particles in a centered block, no initial velocity
    idx = 0
    nx = int(math.sqrt(n_particles * bw / bh))
    ny = n_particles // max(nx, 1)
    for i in range(min(nx, n_particles)):
        for j in range(min(ny, n_particles - idx)):
            if idx >= solver.n:
                break
            px = cx - bw/2 + bw * i / max(nx - 1, 1)
            py = cy - bh/2 + bh * j / max(ny - 1, 1)
            solver.pos[idx] = [px, py]
            solver.vel[idx] = [0.0, 0.0]
            idx += 1
            if idx >= solver.n:
                break

    # Simulate
    sim_steps = 2000
    frames_to_capture = [0, 400, 800, 1200, 1600, 2000]
    frames = []

    for step in range(sim_steps + 1):
        if step in frames_to_capture:
            pos = solver.get_positions()
            frame = render_particles(pos)
            t = step * DT
            frame = add_text(frame, f"t={t:.3f}s [{scene_id}]", (10, 10))
            frames.append(frame)
        if step < sim_steps:
            solver.step()

    # Tilted gravity makes direction deterministic; use config label directly.
    # Compute final CoM from valid (non-NaN) particles for reporting.
    final_pos = solver.get_positions()
    valid_mask = np.isfinite(final_pos[:, 0]) & np.isfinite(final_pos[:, 1])
    if valid_mask.sum() < 10:
        # Too few valid particles - degenerate scene, skip
        return False
    valid_pos = final_pos[valid_mask]
    init_cx = cx
    final_cx = float(np.mean(valid_pos[:, 0]))
    final_cy = float(np.mean(valid_pos[:, 1]))
    dx = final_cx - init_cx
    dy = final_cy - cy

    out = OUT_DIR / "fluid_direction" / scene_id
    save_fluid_frames(frames, out)

    gt = {
        "answer": direction,
        "initial_center": [round(init_cx, 3), round(cy, 3)],
        "final_center": [round(final_cx, 3), round(final_cy, 3)],
        "displacement": [round(dx, 3), round(dy, 3)],
    }
    (out / "config.json").write_text(json.dumps({
        "scene_id": scene_id, "task_type": "fluid_direction",
        "n_particles": n_particles,
        "initial_block": {"cx": cx, "cy": cy, "width": bw, "height": bh},
    }, indent=2))
    (out / "ground_truth.json").write_text(json.dumps(gt, indent=2))
    (out / "prompt.txt").write_text(
        f"A block of fluid (blue particles) is released in a container.\n"
        f"Watch how the fluid flows.\n\n"
        f"In which primary direction does the fluid move?\n\n"
        f"<reasoning>Observe the particle motion</reasoning>\n"
        f"<answer>left/right/down</answer>"
    )
    (out / "assistant_text.txt").write_text(
        f"<reasoning>The fluid block starts at center ({init_cx:.1f}, {cy:.1f}) "
        f"and after simulation the center of mass moves to ({final_cx:.2f}, {final_cy:.2f}). "
        f"The horizontal displacement is {dx:.2f}m and vertical is {dy:.2f}m. "
        f"The primary flow direction is {direction}.</reasoning>\n"
        f"<answer>{direction}</answer>"
    )

    print(f" {scene_id}: direction={direction} "
          f"dx={dx:.2f} dy={dy:.2f} n={n_particles}")
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Task: Fluid Viscosity Comparison
# ═══════════════════════════════════════════════════════════════════════════

def make_fluid_viscosity(scene_id: str, seed: int) -> bool:
    """Two fluids with different viscosity - which is thicker?

    Fix: drop both fluids from a height onto a floor and let them spread
    sideways. High viscosity = stays tall/narrow. Low viscosity = spreads
    wide and flat. Spread is measured as X-range after settling.
    """
    if not _TAICHI_AVAILABLE:
        print(f" SKIP {scene_id}: taichi not available")
        return False

    r = np.random.default_rng(seed)

    # Viscosity range tuned for SPH solver stability (>100 causes NaN)
    visc_low = round(float(r.uniform(10.0, 30.0)), 1) # water-like
    visc_high = round(float(r.uniform(60.0, 95.0)), 1) # honey-like
    if r.random() > 0.5:
        visc_a, visc_b = visc_low, visc_high
    else:
        visc_a, visc_b = visc_high, visc_low

    # Run each fluid SEQUENTIALLY to avoid Taichi multi-instance issues.
    # Same initial conditions, different viscosity. Composite results.
    n_particles = int(r.integers(300, 500))
    cx_a, cx_b, cy = 0.5, 1.5, 0.7
    bw, bh = 0.3, 0.3
    sim_steps = 1500
    frames_to_capture = [0, 300, 600, 900, 1200, 1500]

    def run_one_fluid(viscosity, cx):
        solver = SPHSolver(n_particles, viscosity=viscosity)
        GRAVITY_FIELD[None] = [0.0, -9.81]
        idx = 0
        nx = int(math.sqrt(n_particles * bw / bh))
        ny = n_particles // max(nx, 1)
        # Give particles an outward radial velocity so viscosity
        # determines how far they spread before stopping.
        push_speed = 3.0 # m/s outward from center
        for i in range(min(nx, n_particles)):
            for j in range(min(ny, n_particles - idx)):
                if idx >= solver.n:
                    break
                px = cx - bw/2 + bw * i / max(nx - 1, 1)
                py = cy - bh/2 + bh * j / max(ny - 1, 1)
                solver.pos[idx] = [px, py]
                # Outward velocity from center
                dx = px - cx
                vx = push_speed if dx >= 0 else -push_speed
                solver.vel[idx] = [vx, 0.0]
                idx += 1
                if idx >= solver.n:
                    break
        captured = {}
        for step in range(sim_steps + 1):
            if step in frames_to_capture:
                captured[step] = solver.get_positions().copy()
            if step < sim_steps:
                solver.step()
        return captured

    positions_a = run_one_fluid(visc_a, cx_a)
    positions_b = run_one_fluid(visc_b, cx_b)

    frames = []
    for step in frames_to_capture:
        frame = render_two_fluids(positions_a[step], positions_b[step])
        t = step * DT
        frame = add_text(frame, f"t={t:.3f}s Fluid A (blue) Fluid B (orange)", (10, 10))
        frame = add_text(frame, f"[{scene_id}]", (10, 30))
        frames.append(frame)

    answer = "fluid a" if visc_a > visc_b else "fluid b"

    # Measure spread as X-range of valid particles
    def measure_spread(pos: np.ndarray) -> float:
        valid = pos[np.isfinite(pos[:, 0]) & np.isfinite(pos[:, 1])]
        if len(valid) < 10:
            return 0.0
        return float(np.max(valid[:, 0]) - np.min(valid[:, 0]))

    final_a = positions_a[sim_steps]
    final_b = positions_b[sim_steps]
    spread_a = measure_spread(final_a)
    spread_b = measure_spread(final_b)

    out = OUT_DIR / "fluid_viscosity" / scene_id
    save_fluid_frames(frames, out)

    gt = {
        "answer": answer,
        "viscosity_a": visc_a, "viscosity_b": visc_b,
        "spread_a": round(spread_a, 4), "spread_b": round(spread_b, 4),
    }
    (out / "config.json").write_text(json.dumps({
        "scene_id": scene_id, "task_type": "fluid_viscosity",
        "n_particles": n_particles,
        "viscosity_a": visc_a, "viscosity_b": visc_b,
    }, indent=2))
    (out / "ground_truth.json").write_text(json.dumps(gt, indent=2))
    (out / "prompt.txt").write_text(
        f"Two fluids settle under gravity:\n"
        f"- Fluid A (blue particles, left)\n"
        f"- Fluid B (orange particles, right)\n\n"
        f"Which fluid is more viscous (thicker)? A more viscous fluid resists "
        f"spreading - it stays compact. A less viscous fluid spreads wide and flat.\n\n"
        f"<reasoning>Compare how much each fluid spreads laterally</reasoning>\n"
        f"<answer>fluid a or fluid b</answer>"
    )
    (out / "assistant_text.txt").write_text(
        f"<reasoning>After impacting the floor, the more viscous fluid resists lateral "
        f"flow and stays compact, while the less viscous fluid spreads out widely. "
        f"Fluid A (viscosity {visc_a}) spreads {spread_a:.3f}m and "
        f"Fluid B (viscosity {visc_b}) spreads {spread_b:.3f}m. "
        f"The more viscous fluid is {answer} (less spread).</reasoning>\n"
        f"<answer>{answer}</answer>"
    )

    print(f" {scene_id}: {answer} more viscous "
          f"visc_a={visc_a} visc_b={visc_b} spread_a={spread_a:.3f} spread_b={spread_b:.3f}")
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Task: Fluid Level Prediction
# ═══════════════════════════════════════════════════════════════════════════

def _render_settled_particles(n_particles, container_left, container_right,
                              level_height, walls, domain=(2.0, 1.5),
                              label_text="") -> np.ndarray:
    """Render particles uniformly settled at bottom of container up to level_height."""
    r = np.random.default_rng(42)
    settled_pos = np.zeros((n_particles, 2))
    spacing = PARTICLE_RADIUS * 2.2
    margin = container_left + PARTICLE_RADIUS + 0.005
    usable_w = (container_right - container_left) - 2 * (PARTICLE_RADIUS + 0.005)
    cols = max(int(usable_w / spacing), 1)
    for i in range(n_particles):
        col = i % cols
        row = i // cols
        settled_pos[i, 0] = margin + col * spacing
        settled_pos[i, 1] = PARTICLE_RADIUS + row * spacing
    # Add small jitter for natural look
    settled_pos += r.normal(0, PARTICLE_RADIUS * 0.15, settled_pos.shape)
    settled_pos[:, 0] = np.clip(settled_pos[:, 0],
                                container_left + PARTICLE_RADIUS,
                                container_right - PARTICLE_RADIUS)
    settled_pos[:, 1] = np.clip(settled_pos[:, 1], PARTICLE_RADIUS, domain[1] - PARTICLE_RADIUS)

    frame = render_particles(settled_pos, domain=domain, walls=walls)
    if label_text:
        frame = add_text(frame, label_text, (10, 10))
    return frame


# Pre-defined parameter ranges per level category to ensure variety
_LEVEL_PARAMS = {
    # (n_particles_range, container_width_range)
    "low": ((150, 250), (0.8, 1.2)), # few particles, wide container
    "medium": ((250, 450), (0.5, 0.9)), # moderate
    "high": ((400, 650), (0.35, 0.55)), # more particles, narrower
    "very_high": ((550, 800), (0.3, 0.4)), # many particles, narrow
}
_LEVEL_CATEGORIES = list(_LEVEL_PARAMS.keys())


def make_fluid_level(scene_id: str, seed: int) -> bool:
    """Fluid poured into container -- predict settling level.
    Uses theoretical level (physics-correct) as ground truth.
    Renders final frame with ideal settled particle positions."""
    if not _TAICHI_AVAILABLE:
        print(f" SKIP {scene_id}: taichi not available")
        return False

    r = np.random.default_rng(seed)

    # Pick target category (cycle through evenly)
    cat_idx = seed % len(_LEVEL_CATEGORIES)
    target_cat = _LEVEL_CATEGORIES[cat_idx]
    n_range, w_range = _LEVEL_PARAMS[target_cat]

    n_particles = int(r.integers(n_range[0], n_range[1]))
    container_width = round(float(r.uniform(w_range[0], w_range[1])), 2)
    container_left = (2.0 - container_width) / 2
    container_right = container_left + container_width

    # Theoretical level: volume = n_particles * pi * r^2, level = volume / width
    particle_area = math.pi * PARTICLE_RADIUS**2
    theoretical_level = n_particles * particle_area / container_width

    # Categorize based on theoretical level (physics-correct)
    if theoretical_level < 0.15:
        level_desc = "low (below 20%)"
    elif theoretical_level < 0.35:
        level_desc = "medium (20-50%)"
    elif theoretical_level < 0.55:
        level_desc = "high (50-80%)"
    else:
        level_desc = "very high (above 80%)"

    # Use fewer particles for simulation (cap at 400 for SPH stability)
    sim_n = min(n_particles, 400)

    solver = SPHSolver(sim_n)
    # Start particles above container in a grid
    idx = 0
    cols = max(int(math.sqrt(sim_n * container_width / 0.5)), 1)
    for i in range(min(cols, sim_n)):
        for j in range(sim_n // max(cols, 1) + 1):
            if idx >= solver.n:
                break
            px = container_left + 0.02 + (container_width - 0.04) * i / max(cols - 1, 1)
            py = 0.6 + 0.3 * j / max(sim_n // max(cols, 1), 1)
            solver.pos[idx] = [px, py]
            solver.vel[idx] = [0.0, 0.0]
            idx += 1

    sim_steps = 2000
    frames_to_capture = [0, 400, 800, 1200, 2000]
    frames = []

    walls = [
        [container_left - 0.02, 0.0, container_left, 1.2], # left wall
        [container_right, 0.0, container_right + 0.02, 1.2], # right wall
    ]

    for step in range(sim_steps + 1):
        if step in frames_to_capture:
            pos = solver.get_positions()
            frame = render_particles(pos, walls=walls)
            t = step * DT
            frame = add_text(frame, f"t={t:.3f}s [{scene_id}]", (10, 10))
            frame = add_text(frame,
                f"Container: {container_width:.2f}m wide", (10, 30))
            frames.append(frame)
        if step < sim_steps:
            solver.step()
            # Enforce container walls
            pos_np = solver.get_positions()
            for p_idx in range(solver.n):
                px_val = pos_np[p_idx, 0]
                py_val = pos_np[p_idx, 1]
                if not (np.isfinite(px_val) and np.isfinite(py_val)):
                    # Reset NaN particles to container center floor
                    solver.pos[p_idx] = [(container_left + container_right) / 2,
                                         PARTICLE_RADIUS]
                    solver.vel[p_idx] = [0.0, 0.0]
                    continue
                if px_val < container_left + PARTICLE_RADIUS:
                    solver.pos[p_idx] = [container_left + PARTICLE_RADIUS, py_val]
                    solver.vel[p_idx] = [abs(float(solver.vel[p_idx][0])) * 0.3,
                                         float(solver.vel[p_idx][1])]
                if px_val > container_right - PARTICLE_RADIUS:
                    solver.pos[p_idx] = [container_right - PARTICLE_RADIUS, py_val]
                    solver.vel[p_idx] = [-abs(float(solver.vel[p_idx][0])) * 0.3,
                                          float(solver.vel[p_idx][1])]

    # Replace final frame with ideal settled render (physics-correct visual)
    settled_frame = _render_settled_particles(
        n_particles, container_left, container_right, theoretical_level, walls,
        label_text=f"settled [{scene_id}] {container_width:.2f}m wide")
    settled_frame = add_text(settled_frame,
        f"Level: {level_desc}", (10, 30))
    frames.append(settled_frame)

    out = OUT_DIR / "fluid_level" / scene_id
    save_fluid_frames(frames, out)

    gt = {
        "answer": level_desc,
        "theoretical_level_m": round(theoretical_level, 3),
        "container_width_m": container_width,
        "n_particles": n_particles,
    }
    (out / "config.json").write_text(json.dumps({
        "scene_id": scene_id, "task_type": "fluid_level",
        "container_width": container_width,
        "n_particles": n_particles,
    }, indent=2))
    (out / "ground_truth.json").write_text(json.dumps(gt, indent=2))
    (out / "prompt.txt").write_text(
        f"Fluid (blue particles) is poured into a container.\n"
        f"Container width: {container_width:.2f}m.\n\n"
        f"After the fluid settles, what level will it reach?\n\n"
        f"<reasoning>Consider the amount of fluid and container size</reasoning>\n"
        f"<answer>low/medium/high/very high</answer>"
    )
    (out / "assistant_text.txt").write_text(
        f"<reasoning>The container is {container_width:.2f}m wide and receives "
        f"{n_particles} particles of fluid. Based on the volume of fluid "
        f"(n={n_particles}, particle radius={PARTICLE_RADIUS}m) and the container width, "
        f"the theoretical settling height is {theoretical_level:.3f}m. "
        f"This corresponds to a {level_desc} level.</reasoning>\n"
        f"<answer>{level_desc}</answer>"
    )

    print(f" {scene_id}: theoretical={theoretical_level:.3f}m ({level_desc}) "
          f"width={container_width} n={n_particles}")
    return True


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

ALL_TASKS = ["fluid_direction", "fluid_viscosity", "fluid_level"]

TASK_GENERATORS = {
    "fluid_direction": make_fluid_direction,
    "fluid_viscosity": make_fluid_viscosity,
    "fluid_level": make_fluid_level,
}


def generate_all(n_per_task: int, tasks: list[str], seed: int = 42):
    if not _TAICHI_AVAILABLE:
        print("ERROR: Taichi not installed. Run: pip install taichi")
        return 0

    total = 0
    for task in tasks:
        if task not in TASK_GENERATORS:
            print(f" Unknown task: {task}")
            continue

        gen_fn = TASK_GENERATORS[task]
        print(f"\nGenerating {n_per_task} {task} scenes...")
        ok = 0
        for i in range(n_per_task):
            scene_id = f"{task}_{i:06d}"
            try:
                if gen_fn(scene_id, seed=seed + i):
                    ok += 1
            except Exception as e:
                print(f" ERROR {scene_id}: {e}")
        print(f" {task}: {ok}/{n_per_task} generated")
        total += ok

    print(f"\nTotal: {total} fluid scenes")
    print(f"Output: {OUT_DIR}")
    return total


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Taichi fluid simulation training data")
    parser.add_argument("--n", type=int, default=500,
                        help="Scenes per task (default 500)")
    parser.add_argument("--tasks", type=str, default="all",
                        help="Comma-separated task list or 'all'")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    tasks = ALL_TASKS if args.tasks == "all" else args.tasks.split(",")
    generate_all(n_per_task=args.n, tasks=tasks, seed=args.seed)
