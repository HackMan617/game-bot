"""gdbot — a NEAT-driven bot that learns to play Geometry Dash (GD 2.2).

Architecture: one Gym-style environment interface (`env_base.GDEnv`) with two
interchangeable backends —

    SimEnv   : a fast, deterministic, headless pygame cube simulator used to
               develop and train the network without the real game.
    LiveEnv  : the real Geometry Dash executable, via memory reading (pymem)
               and input injection (Windows SendInput).  [Phase 2+]

The learning code (neat_core, later rl_core) is written against GDEnv only, so
switching from Sim to Live requires no change to the network or training loop.
"""

__all__ = ["env_base", "sim_env", "observation", "game_state", "neat_core"]
