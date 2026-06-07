"""
PhysSim-VLM: Physics Simulation Verifier
Computes ground truth for TTC, stability, and trajectory tasks using MuJoCo.
"""

import math
import numpy as np
import mujoco
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Scene config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class ObjectConfig:
    shape: str # "sphere", "box", "cylinder"
    size: float # radius for sphere/cylinder, half-extent for box
    height: float = 0.0 # for cylinder/box (z half-extent)
    mass: float = 1.0
    color: str = "red"
    label: str = ""
    position: list = field(default_factory=lambda: [0.0, 0.0, 0.5])
    velocity: list = field(default_factory=lambda: [0.0, 0.0, 0.0])


@dataclass
class TTCConfig:
    obj1: ObjectConfig
    obj2: ObjectConfig
    surface_friction: float = 0.6
    max_sim_time: float = 10.0
    timestep: float = 0.001


@dataclass
class StabilityConfig:
    objects: list # list of ObjectConfig (bottom to top)
    surface_friction: float = 0.7
    sim_duration: float = 3.0
    settling_time: float = 0.1
    displacement_threshold: float = 0.02
    timestep: float = 0.001


@dataclass
class TrajectoryConfig:
    obj: ObjectConfig
    surface_friction: float = 0.5
    restitution: float = 0.3
    max_sim_time: float = 10.0
    timestep: float = 0.001
    waypoint_interval: float = 0.05


# ---------------------------------------------------------------------------
# XML builders
# ---------------------------------------------------------------------------

def _color_rgba(name: str) -> str:
    colors = {
        "red": "0.9 0.2 0.2 1",
        "blue": "0.2 0.4 0.9 1",
        "green": "0.2 0.8 0.3 1",
        "yellow": "0.9 0.8 0.1 1",
        "orange": "0.9 0.5 0.1 1",
        "purple": "0.6 0.2 0.8 1",
        "white": "0.9 0.9 0.9 1",
        "gray": "0.5 0.5 0.5 1",
    }
    return colors.get(name, "0.7 0.7 0.7 1")


def _geom_xml(obj: ObjectConfig, name: str, friction: float) -> str:
    pos = f"{obj.position[0]} {obj.position[1]} {obj.position[2]}"
    rgba = _color_rgba(obj.color)
    if obj.shape == "sphere":
        geom = f'size="{obj.size}"'
        gtype = "sphere"
    elif obj.shape == "cylinder":
        h = obj.height if obj.height > 0 else obj.size
        geom = f'size="{obj.size} {h}"'
        gtype = "cylinder"
    else: # box
        h = obj.height if obj.height > 0 else obj.size
        geom = f'size="{obj.size} {obj.size} {h}"'
        gtype = "box"

    return (
        f'<body name="{name}" pos="{pos}">'
        f'<freejoint/>'
        f'<geom type="{gtype}" {geom} mass="{obj.mass}" '
        f'rgba="{rgba}" friction="{friction} 0.005 0.0001"/>'
        f'</body>'
    )


def _build_ttc_xml(cfg: TTCConfig) -> str:
    return f"""
<mujoco model="ttc">
  <option timestep="{cfg.timestep}" gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="10 10 0.1"
          rgba="0.8 0.8 0.8 1" friction="{cfg.surface_friction} 0.005 0.0001"/>
    {_geom_xml(cfg.obj1, "obj1", cfg.surface_friction)}
    {_geom_xml(cfg.obj2, "obj2", cfg.surface_friction)}
  </worldbody>
</mujoco>
"""


def _build_stability_xml(cfg: StabilityConfig) -> str:
    bodies = ""
    for i, obj in enumerate(cfg.objects):
        bodies += _geom_xml(obj, f"obj{i}", cfg.surface_friction) + "\n"
    return f"""
<mujoco model="stability">
  <option timestep="{cfg.timestep}" gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="10 10 0.1"
          rgba="0.8 0.8 0.8 1" friction="{cfg.surface_friction} 0.005 0.0001"/>
    {bodies}
  </worldbody>
</mujoco>
"""


def _build_trajectory_xml(cfg: TrajectoryConfig) -> str:
    return f"""
<mujoco model="trajectory">
  <option timestep="{cfg.timestep}" gravity="0 0 -9.81"/>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="20 20 0.1"
          rgba="0.8 0.8 0.8 1"
          friction="{cfg.surface_friction} 0.005 0.0001"
          solimp="0.9 0.95 0.001" solref="0.02 1"/>
    {_geom_xml(cfg.obj, "obj", cfg.surface_friction)}
  </worldbody>
</mujoco>
"""


# ---------------------------------------------------------------------------
# Initial velocity setter
# ---------------------------------------------------------------------------

def _set_velocity(model: mujoco.MjModel, data: mujoco.MjData,
                  body_name: str, velocity: list) -> None:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return
    # Find the freejoint DOF offset for this body
    jnt_id = model.body_jntadr[body_id]
    if jnt_id < 0:
        return
    qvel_addr = model.jnt_dofadr[jnt_id]
    data.qvel[qvel_addr] = velocity[0]
    data.qvel[qvel_addr + 1] = velocity[1]
    data.qvel[qvel_addr + 2] = velocity[2]


def _get_position(model: mujoco.MjModel, data: mujoco.MjData,
                  body_name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return data.xpos[body_id].copy()


# ---------------------------------------------------------------------------
# Verifier functions
# ---------------------------------------------------------------------------

def verify_ttc(cfg: TTCConfig) -> dict:
    """
    Run TTC simulation. Returns collision time and contact details.
    """
    xml = _build_ttc_xml(cfg)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)
    _set_velocity(model, data, "obj1", cfg.obj1.velocity)
    _set_velocity(model, data, "obj2", cfg.obj2.velocity)

    steps_per_sec = int(1.0 / cfg.timestep)
    max_steps = int(cfg.max_sim_time * steps_per_sec)

    collision_time = None
    contact_force = 0.0
    contact_point = [0.0, 0.0, 0.0]

    for step in range(max_steps):
        mujoco.mj_step(model, data)

        if data.ncon > 0:
            t = data.time
            # Find contact between obj1 and obj2
            for i in range(data.ncon):
                c = data.contact[i]
                g1 = model.geom_bodyid[c.geom1]
                g2 = model.geom_bodyid[c.geom2]
                b1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obj1")
                b2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obj2")
                if (g1 == b1 and g2 == b2) or (g1 == b2 and g2 == b1):
                    collision_time = t
                    contact_point = list(c.pos)
                    # Compute contact force magnitude
                    force = np.zeros(6)
                    mujoco.mj_contactForce(model, data, i, force)
                    contact_force = float(np.linalg.norm(force[:3]))
                    break
            if collision_time is not None:
                break

    del data
    del model

    return {
        "collision_occurred": collision_time is not None,
        "time_to_collision": round(collision_time, 4) if collision_time else None,
        "contact_force": round(contact_force, 4),
        "contact_point": [round(v, 4) for v in contact_point],
        "simulation_steps": step + 1,
        "simulation_timestep": cfg.timestep,
    }


def verify_stability(cfg: StabilityConfig) -> dict:
    """
    Run stability simulation. Returns whether the stack is stable.
    """
    xml = _build_stability_xml(cfg)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)

    # Record initial positions after settling
    settling_steps = int(cfg.settling_time / cfg.timestep)
    for _ in range(settling_steps):
        mujoco.mj_step(model, data)

    n_objects = len(cfg.objects)
    initial_pos = []
    for i in range(n_objects):
        initial_pos.append(_get_position(model, data, f"obj{i}").copy())

    # Run full simulation
    sim_steps = int(cfg.sim_duration / cfg.timestep)
    collapse_time = None

    for step in range(sim_steps):
        mujoco.mj_step(model, data)

        # Check if any object has moved significantly
        if collapse_time is None:
            for i in range(n_objects):
                pos = _get_position(model, data, f"obj{i}")
                disp = np.linalg.norm(pos - initial_pos[i])
                if disp > cfg.displacement_threshold:
                    collapse_time = data.time
                    break

    # Final displacement
    final_displacements = []
    for i in range(n_objects):
        pos = _get_position(model, data, f"obj{i}")
        disp = float(np.linalg.norm(pos - initial_pos[i]))
        final_displacements.append(round(disp, 4))

    max_disp = max(final_displacements)
    is_stable = max_disp <= cfg.displacement_threshold

    del data
    del model

    return {
        "is_stable": is_stable,
        "max_displacement_m": round(max_disp, 4),
        "total_displacement_m": round(sum(final_displacements), 4),
        "per_object_displacement": final_displacements,
        "threshold_used_m": cfg.displacement_threshold,
        "simulation_duration_s": cfg.sim_duration,
        "settling_phase_s": cfg.settling_time,
        "collapse_time_s": round(collapse_time, 4) if collapse_time else None,
    }


def verify_trajectory(cfg: TrajectoryConfig) -> dict:
    """
    Run trajectory simulation. Returns landing position and waypoints.
    """
    xml = _build_trajectory_xml(cfg)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    mujoco.mj_forward(model, data)
    _set_velocity(model, data, "obj", cfg.obj.velocity)

    steps_per_sec = int(1.0 / cfg.timestep)
    max_steps = int(cfg.max_sim_time * steps_per_sec)
    waypoint_steps = int(cfg.waypoint_interval / cfg.timestep)

    launch_pos = _get_position(model, data, "obj").copy()
    waypoints = []
    max_height = float(launch_pos[2])
    n_bounces = 0
    last_z_vel = cfg.obj.velocity[2]
    flight_time = None
    first_landing_pos = None # position at first ground contact

    for step in range(max_steps):
        mujoco.mj_step(model, data)

        pos = _get_position(model, data, "obj")
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "obj")
        jnt_id = model.body_jntadr[body_id]
        qvel_addr = model.jnt_dofadr[jnt_id]
        vel = data.qvel[qvel_addr:qvel_addr + 3].copy()
        speed = float(np.linalg.norm(vel))

        # Track max height
        if float(pos[2]) > max_height:
            max_height = float(pos[2])

        # Track bounces (z velocity sign change near floor)
        z_vel = float(vel[2])
        if last_z_vel < 0 and z_vel > 0 and float(pos[2]) < (launch_pos[2] + 0.1):
            n_bounces += 1
            if flight_time is None:
                flight_time = data.time
                first_landing_pos = pos.copy() # record first landing point
        last_z_vel = z_vel

        # Record waypoints at interval
        if step % waypoint_steps == 0:
            waypoints.append({
                "t": round(data.time, 3),
                "x": round(float(pos[0] - launch_pos[0]), 4),
                "y": round(float(pos[1] - launch_pos[1]), 4),
                "z": round(float(pos[2]), 4),
                "speed": round(speed, 4),
            })

        # Stop when object has effectively stopped
        if speed < 0.01 and data.time > 0.5:
            break

    final_pos = _get_position(model, data, "obj")
    total_time = round(float(data.time), 4)
    # Use first landing point if available, else final stopped position
    report_pos = first_landing_pos if first_landing_pos is not None else final_pos

    del data
    del model

    return {
        "landing_position": {
            "x": round(float(report_pos[0] - launch_pos[0]), 4),
            "y": round(float(report_pos[1] - launch_pos[1]), 4),
        },
        "landing_height": round(float(final_pos[2]), 4),
        "total_time_s": total_time,
        "flight_time_s": round(flight_time, 4) if flight_time else None,
        "object_stopped": True,
        "n_bounces": n_bounces,
        "max_height_m": round(max_height, 4),
        "trajectory_waypoints": waypoints,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def render_frames(model: mujoco.MjModel, data: mujoco.MjData,
                  n_frames: int, frame_interval: float,
                  width: int = 640, height: int = 480) -> list:
    """
    Render a sequence of frames from current simulation state.
    Returns list of numpy arrays (H, W, 3).
    """
    renderer = mujoco.Renderer(model, height=height, width=width)
    frames = []
    steps_per_frame = max(1, int(frame_interval / model.opt.timestep))

    for f in range(n_frames):
        mujoco.mj_forward(model, data)
        renderer.update_scene(data)
        pixels = renderer.render()
        frames.append(pixels.copy())
        if f < n_frames - 1:
            for _ in range(steps_per_frame):
                mujoco.mj_step(model, data)

    renderer.close()
    return frames


def render_single_frame(model: mujoco.MjModel, data: mujoco.MjData,
                        width: int = 640, height: int = 480) -> np.ndarray:
    """Render a single frame."""
    renderer = mujoco.Renderer(model, height=height, width=width)
    mujoco.mj_forward(model, data)
    renderer.update_scene(data)
    pixels = renderer.render()
    renderer.close()
    return pixels


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== TTC Test ===")
    cfg = TTCConfig(
        obj1=ObjectConfig(shape="sphere", size=0.1, mass=1.0, color="red",
                          position=[-1.5, 0.0, 0.11], velocity=[2.0, 0.0, 0.0]),
        obj2=ObjectConfig(shape="box", size=0.1, mass=1.0, color="blue",
                          position=[1.5, 0.0, 0.11], velocity=[-1.0, 0.0, 0.0]),
        surface_friction=0.3,
    )
    result = verify_ttc(cfg)
    print(f" Collision: {result['collision_occurred']}")
    print(f" TTC: {result['time_to_collision']}s")

    print("\n=== Stability Test (stable) ===")
    scfg = StabilityConfig(
        objects=[
            ObjectConfig(shape="box", size=0.2, height=0.05, mass=2.0, color="blue",
                         position=[0.0, 0.0, 0.05]),
            ObjectConfig(shape="box", size=0.1, height=0.05, mass=1.0, color="green",
                         position=[0.0, 0.0, 0.15]),
        ]
    )
    result = verify_stability(scfg)
    print(f" Stable: {result['is_stable']}")
    print(f" Max displacement: {result['max_displacement_m']}m")

    print("\n=== Stability Test (unstable) ===")
    scfg2 = StabilityConfig(
        objects=[
            ObjectConfig(shape="box", size=0.1, height=0.05, mass=2.0, color="blue",
                         position=[0.0, 0.0, 0.05]),
            ObjectConfig(shape="sphere", size=0.05, mass=0.3, color="red",
                         position=[0.12, 0.0, 0.16]), # sphere center past box edge
        ]
    )
    result = verify_stability(scfg2)
    print(f" Stable: {result['is_stable']}")
    print(f" Max displacement: {result['max_displacement_m']}m")
    print(f" Collapse time: {result['collapse_time_s']}s")

    print("\n=== Trajectory Test ===")
    tcfg = TrajectoryConfig(
        obj=ObjectConfig(shape="sphere", size=0.06, mass=0.3, color="red",
                         position=[0.0, 0.0, 0.5], velocity=[2.0, 0.0, 3.0]),
        surface_friction=0.5,
        restitution=0.3,
    )
    result = verify_trajectory(tcfg)
    print(f" Landing: x={result['landing_position']['x']}m, y={result['landing_position']['y']}m")
    print(f" Max height: {result['max_height_m']}m")
    print(f" Bounces: {result['n_bounces']}")
    print(f" Waypoints: {len(result['trajectory_waypoints'])}")
