"""Python side of the GDBot Bridge contract (protocol v5).

Mirrors `struct GDBotShared` in geode-mod/src/main.cpp. The Geode mod (running
inside GeometryDash.exe) writes state + a 24x16x4 occupancy grid every frame,
signals a named event, and waits a bounded window for us to answer with an
action. That handshake is what lets the game run fast without the agent silently
missing frames the way the old 300Hz polling loop did — and because the mod
waits for us, the game runs exactly as fast as the agent can think.

One rendered frame = one fixed 1/step_hz slice of game time = one decision, so
wall-clock speedup is render_fps / step_hz and the policy is unaffected by it.

Standalone tools:
    python -m gdbot.bridge            # live state readout
    python -m gdbot.bridge --grid     # ASCII render of what the network sees
    python -m gdbot.bridge --bench    # throughput + handshake latency by speed
"""

import ctypes
import mmap
import struct
import time
from ctypes import wintypes

VERSION = 5
GRID_W, GRID_H, GRID_C = 24, 16, 4
GRID_BEHIND = 2            # grid columns behind the player
GRID_CELLS = GRID_C * GRID_H * GRID_W
CELL = 30.0                # GD units per grid cell (1 block)
MAX_CP = 64
_SIZE = 65536
_TAG = "GDBotShared"
MAGIC = 0x54444247         # 'GDBT'

# Command opcodes — must match the enum in main.cpp.
CMD_RESET_START = 1
CMD_LOAD_LEVEL = 2
CMD_SET_PRACTICE = 3
CMD_RESPAWN_CP = 4
CMD_CLEAR_CP = 5

# Field order must match the C++ struct exactly.
_HEAD = ("magic", "version", "state_seq", "action_seq",
         "in_level", "dead", "on_ground", "gamemode", "attempt",
         "upside_down", "level_id", "checkpoint_count",
         "is_practice")                                            # 13 int32
_FLOATS = ("x", "y", "vy", "percent", "length",
           "ground_y", "ceiling_y",
           "player_speed", "gravity_mod", "vehicle_size")          # 10 float32
_TAIL = ("action", "cmd_epoch", "cmd_op", "cmd_arg", "cmd_ack",
         "speed", "fast_respawn", "mute", "attached", "step_hz",
         "grid_w", "grid_h", "grid_c")                             # 13 int32

_FMT = struct.Struct("<13i10f13i%df%dB" % (MAX_CP, GRID_CELLS))
_N_SCALARS = len(_HEAD) + len(_FLOATS) + len(_TAIL)


_FIELDS = _HEAD + _FLOATS + _TAIL


def field_offset(name: str) -> int:
    """Byte offset of a scalar field, derived from the declared order.

    Every scalar in the struct is 4 bytes, so position determines offset. Deriving
    them means adding a field can never silently leave a hand-written offset
    pointing at its neighbour.
    """
    return _FIELDS.index(name) * 4


# Fields we write into. Never write the whole struct back — the mod owns most of it.
_O_ACTION_SEQ = field_offset("action_seq")
_O_ACTION = field_offset("action")
_O_CMD_EPOCH = field_offset("cmd_epoch")
_O_CMD_OP = field_offset("cmd_op")
_O_CMD_ARG = field_offset("cmd_arg")
_O_CMD_ACK = field_offset("cmd_ack")
_O_SPEED = field_offset("speed")
_O_FAST_RESPAWN = field_offset("fast_respawn")
_O_MUTE = field_offset("mute")
_O_ATTACHED = field_offset("attached")
_O_STEP_HZ = field_offset("step_hz")
_O_CP_PCT = len(_FIELDS) * 4
_O_GRID = _O_CP_PCT + MAX_CP * 4

assert _O_GRID + GRID_CELLS == _FMT.size, "offset table disagrees with the struct format"

OFFICIAL_LEVELS = {
    1: "Stereo Madness", 2: "Back on Track", 3: "Polargeist", 4: "Dry Out",
    5: "Base After Base", 6: "Can't Let Go", 7: "Jumper", 8: "Time Machine",
    9: "Cycles", 10: "xStep", 11: "Clutterfunk", 12: "Theory of Everything",
    13: "Electroman Adventures", 14: "Clubstep", 15: "Electrodynamix",
    16: "Hexagon Force", 17: "Blast Processing", 18: "Theory of Everything 2",
    19: "Geometrical Dominator", 20: "Deadlocked", 21: "Fingerdash",
}

# --- Win32 event handshake ---------------------------------------------------
_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_k32.CreateEventW.argtypes = (wintypes.LPVOID, wintypes.BOOL, wintypes.BOOL,
                              wintypes.LPCWSTR)
_k32.CreateEventW.restype = wintypes.HANDLE
_k32.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
_k32.WaitForSingleObject.restype = wintypes.DWORD
_k32.SetEvent.argtypes = (wintypes.HANDLE,)
_k32.SetEvent.restype = wintypes.BOOL
_k32.CloseHandle.argtypes = (wintypes.HANDLE,)
_WAIT_OBJECT_0 = 0


class BridgeLost(RuntimeError):
    """The mod stopped publishing frames (game closed, or left the level)."""


class VersionMismatch(RuntimeError):
    """The installed mod speaks a different protocol version — rebuild it."""


class Bridge:
    """Reader/writer for the mod's shared block, with the frame handshake."""

    def __init__(self, attach: bool = True, tag: str = _TAG):
        # `tag` exists so tests can run against a fake mod while the real one is
        # live in GD — otherwise both write the same block and corrupt each other.
        self.mm = mmap.mmap(-1, _SIZE, tagname=tag)
        # Auto-reset events; CreateEventW opens the mod's if it already exists.
        self._ev_state = _k32.CreateEventW(None, False, False, tag + "StateReady")
        self._ev_action = _k32.CreateEventW(None, False, False, tag + "ActionReady")
        self._last_seq = -1
        self.missed = 0          # physics frames that advanced by more than one
        self.frames = 0
        if attach:
            self.attach()

    # --- connection ----------------------------------------------------------
    def connected(self) -> bool:
        try:
            magic, version = struct.unpack("<2i", self.mm[:8])
        except Exception:
            return False
        if magic != MAGIC:
            return False
        if version != VERSION:
            raise VersionMismatch(
                f"mod speaks v{version}, this client is v{VERSION} — rebuild "
                f"geode-mod and restart GD")
        # Second guard: the version can match while the layout has drifted, so
        # confirm the dimensions land where we expect to read them.
        st = self.read()
        if (st["grid_w"], st["grid_h"], st["grid_c"]) != (GRID_W, GRID_H, GRID_C):
            raise VersionMismatch(
                f"grid dims read as {st['grid_w']}x{st['grid_h']}x{st['grid_c']}, "
                f"expected {GRID_W}x{GRID_H}x{GRID_C} — shared layout is out of sync")
        return True

    def wait_connected(self, timeout: float = None) -> bool:
        t0 = time.time()
        while not self.connected():
            if timeout is not None and time.time() - t0 > timeout:
                return False
            time.sleep(0.1)
        return True

    def attach(self) -> None:
        """Tell the mod an agent is driving, so it waits for our actions."""
        self._w(_O_ATTACHED, 1)

    def detach(self) -> None:
        """Release the mod: no more waiting, speed back to 1x, jump released."""
        self._w(_O_ATTACHED, 0)
        self._w(_O_SPEED, 1)
        self._w(_O_ACTION, 0)

    # --- raw access ----------------------------------------------------------
    def _w(self, off: int, val: int) -> None:
        self.mm[off:off + 4] = struct.pack("<i", val)

    def read(self) -> dict:
        vals = _FMT.unpack(self.mm[:_FMT.size])
        d = dict(zip(_HEAD + _FLOATS + _TAIL, vals[:_N_SCALARS]))
        d["cp_pct"] = vals[_N_SCALARS:_N_SCALARS + MAX_CP]
        return d

    def read_grid(self):
        """The occupancy grid as a (C, H, W) uint8 array — conv-ready."""
        import numpy as np
        return np.frombuffer(self.mm[_O_GRID:_O_GRID + GRID_CELLS],
                             dtype=np.uint8).reshape(GRID_C, GRID_H, GRID_W)

    # --- the per-frame handshake ---------------------------------------------
    def wait_frame(self, timeout: float = 10.0) -> dict:
        """Block until the mod publishes the next physics frame.

        Falls back to polling `state_seq` if the event is missed, so a dropped
        signal costs latency rather than a hang.
        """
        deadline = time.time() + timeout
        while True:
            _k32.WaitForSingleObject(self._ev_state, 2)
            st = self.read()
            if st["magic"] == MAGIC and st["state_seq"] != self._last_seq:
                if self._last_seq >= 0:
                    gap = st["state_seq"] - self._last_seq
                    if gap > 1:
                        self.missed += gap - 1
                self._last_seq = st["state_seq"]
                self.frames += 1
                return st
            if time.time() > deadline:
                raise BridgeLost(f"no physics frame for {timeout:.0f}s (game closed?)")

    def send_action(self, jump: bool, seq: int) -> None:
        """Answer the frame `seq` with an action and release the mod."""
        self.mm[_O_ACTION:_O_ACTION + 4] = struct.pack("<i", 1 if jump else 0)
        self.mm[_O_ACTION_SEQ:_O_ACTION_SEQ + 4] = struct.pack("<i", seq)
        _k32.SetEvent(self._ev_action)

    # --- config --------------------------------------------------------------
    def set_speed(self, n: int) -> None:
        """Physics steps per rendered frame (1..32). One step = one decision."""
        self._w(_O_SPEED, max(1, min(32, int(n))))

    def set_fast_respawn(self, on: bool) -> None:
        self._w(_O_FAST_RESPAWN, 1 if on else 0)

    def set_mute(self, on: bool) -> None:
        self._w(_O_MUTE, 1 if on else 0)

    def set_step_hz(self, hz: int) -> None:
        """Agent decisions per second of GAME time.

        While attached the mod steps with a fixed dt of 1/hz, so this rate holds
        regardless of the render framerate (which is ~140Hz here, not 60) and
        regardless of `speed`. Changing it changes what the network experiences,
        so keep it fixed across a training run.
        """
        self._w(_O_STEP_HZ, max(1, min(240, int(hz))))

    # --- commands ------------------------------------------------------------
    def command(self, op: int, arg: int = 0, wait: float = 2.0) -> bool:
        """Issue a command and wait for the mod to acknowledge it."""
        epoch = self.read()["cmd_epoch"] + 1
        self._w(_O_CMD_OP, op)
        self._w(_O_CMD_ARG, arg)
        self._w(_O_CMD_EPOCH, epoch)
        deadline = time.time() + wait
        while time.time() < deadline:
            if self.read()["cmd_ack"] == epoch:
                return True
            time.sleep(0.002)
        return False

    def reset_from_start(self):        return self.command(CMD_RESET_START)
    def clear_checkpoints(self):       return self.command(CMD_CLEAR_CP)
    def set_practice(self, on: bool):  return self.command(CMD_SET_PRACTICE, 1 if on else 0)
    def respawn_at(self, index: int):  return self.command(CMD_RESPAWN_CP, int(index))
    def load_level(self, level_id: int):
        return self.command(CMD_LOAD_LEVEL, int(level_id), wait=10.0)

    def leave_to_menu(self):           return self.load_level(0)

    def close(self) -> None:
        try:
            self.detach()
        except Exception:
            pass
        for h in (self._ev_state, self._ev_action):
            if h:
                _k32.CloseHandle(h)
        self.mm.close()


# --- standalone tools --------------------------------------------------------
def _connect(attach=True):
    b = Bridge(attach=attach)
    print("Waiting for GDBot Bridge v2... (start GD, enter a level)")
    if not b.wait_connected(timeout=120):
        raise SystemExit("Bridge never connected — is the mod installed?")
    return b


def _readout():
    b = _connect(attach=False)
    print("connected.")
    last = -1
    while True:
        st = b.read()
        if st["state_seq"] != last:
            last = st["state_seq"]
            print(f"x={st['x']:7.1f} y={st['y']:6.1f} {st['percent']*100:5.1f}% "
                  f"mode={st['gamemode']} dead={st['dead']} grnd={st['on_ground']} "
                  f"cp={st['checkpoint_count']} lvl={st['level_id']}")
        time.sleep(0.05)


def _grid():
    """Render the occupancy grid so you can eyeball that perception is correct."""
    b = _connect(attach=False)
    marks = {0: "#", 1: "!", 2: "o", 3: "P"}   # solid, hazard, orb/pad, portal
    print("connected. # solid  ! hazard  o orb/pad  P portal  @ player\n")
    last = -1
    while True:
        st = b.read()
        if st["state_seq"] == last:
            time.sleep(0.02)
            continue
        last = st["state_seq"]
        g = b.read_grid()
        rows = []
        for r in range(GRID_H - 1, -1, -1):       # top row first
            line = []
            for w in range(GRID_W):
                ch = next((c for c in range(GRID_C) if g[c, r, w]), None)
                if r == GRID_H // 2 and w == GRID_BEHIND:
                    line.append("@")
                else:
                    line.append(marks[ch] if ch is not None else ".")
            rows.append("".join(line))
        print("\033[H\033[J", end="")
        print(f"x={st['x']:7.1f} {st['percent']*100:5.1f}%  mode={st['gamemode']}  "
              f"vy={st['vy']:7.1f}  ground={st['on_ground']}\n")
        print("\n".join(rows))
        time.sleep(0.05)


def _bench():
    """Measure real throughput and handshake latency at each speed multiplier."""
    import numpy as np
    b = _connect()
    b.set_fast_respawn(True)
    b.set_mute(True)
    print(f"{'speed':>6} {'steps/s':>9} {'x real':>7} {'missed':>7} "
          f"{'lat p50':>9} {'lat p99':>9}")
    try:
        for speed in (1, 2, 4, 8, 16):
            b.set_speed(speed)
            # let the game settle at the new rate before measuring
            for _ in range(120):
                st = b.wait_frame()
                b.send_action(False, st["state_seq"])
            b.missed = 0
            lat, n = [], 600
            t0 = time.perf_counter()
            for _ in range(n):
                st = b.wait_frame()
                t = time.perf_counter()
                b.send_action(False, st["state_seq"])
                lat.append((time.perf_counter() - t) * 1e6)
            dt = time.perf_counter() - t0
            sps = n / dt
            lat = np.array(lat)
            print(f"{speed:>6} {sps:>9.0f} {sps/60:>7.1f} {b.missed:>7} "
                  f"{np.percentile(lat,50):>8.0f}us {np.percentile(lat,99):>8.0f}us")
    finally:
        b.set_speed(1)
        b.close()


if __name__ == "__main__":
    import sys
    if "--grid" in sys.argv:
        _grid()
    elif "--bench" in sys.argv:
        _bench()
    else:
        _readout()
