"""gdbot — a bot that learns to play Geometry Dash (GD 2.2).

The stack, bottom to top:

    bridge     the shared-memory + event protocol spoken by the Geode mod: a
               24x16x4 occupancy grid per frame, and a handshake that makes the
               game wait for us instead of us polling for it
    obs        the observation contract — grid + scalars — that both backends emit
    env        GDEnv, and LiveEnv on top of the bridge
    sim_env    SimEnv, the same contract with no game running
    policy     the conv actor-critic, and the activations the viewer draws
    ppo        rollout buffer, GAE, clipped-surrogate update
    telemetry  the one-way channel to the browser viewer, off the hot path

Training never renders. `train.py` runs headless at full speed and publishes a
snapshot only while a viewer is connected; `viewer/index.html` draws it.
"""

__all__ = ["bridge", "obs", "env", "sim_env", "policy", "ppo", "telemetry"]
