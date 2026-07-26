# GDBot Bridge

A developer/offline utility that connects Geometry Dash to the **gdbot** learning
agent. Each physics frame it publishes the current player state (position,
velocity, gamemode, on-ground, dead, %, level length) into a shared-memory block
and applies a single jump action requested by the agent.

Intended for **local, single-player research** only — it does not touch online
levels or leaderboards.
