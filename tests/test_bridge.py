"""Protocol tests for the v2 bridge, with no Geometry Dash needed.

A fake mod stands in for geode-mod/src/main.cpp: it creates the same named
shared block and the same two auto-reset events, publishes frames, and waits for
the agent's action exactly as the C++ side does. That exercises the struct
layout, the offsets we write into, the handshake, and the command channel.

    python tests/test_bridge.py
"""

import ctypes
import mmap
import os
import struct
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gdbot import bridge as B  # noqa: E402


# A private name, so these tests are unaffected by a live GD running the real
# mod (both would otherwise write the same block).
TEST_TAG = "GDBotSharedTest"


class FakeMod:
    """Mimics the mod's half of the contract."""

    def __init__(self, hz=2000.0, tag=TEST_TAG):
        self.mm = mmap.mmap(-1, B._SIZE, tagname=tag)
        self.ev_state = B._k32.CreateEventW(None, False, False, tag + "StateReady")
        self.ev_action = B._k32.CreateEventW(None, False, False, tag + "ActionReady")
        self.period = 1.0 / hz
        self.stop = threading.Event()
        self.seq = 0
        self.actions = []          # actions we actually received, in order
        self.answered = 0
        self.commands = []         # (op, arg) the agent issued
        self._last_cmd = 0
        # header the way ensureShared() stamps it
        F = B.field_offset            # never hardcode offsets; derive them
        self._w(F("magic"), B.MAGIC)
        self._w(F("version"), B.VERSION)
        self._w(F("grid_w"), B.GRID_W)
        self._w(F("grid_h"), B.GRID_H)
        self._w(F("grid_c"), B.GRID_C)
        self._w(F("speed"), 1)
        self._w(F("step_hz"), 60)

    def _w(self, off, val):
        self.mm[off:off + 4] = struct.pack("<i", val)

    def _wf(self, off, val):
        self.mm[off:off + 4] = struct.pack("<f", val)

    def run(self):
        t = threading.Thread(target=self._loop, daemon=True)
        t.start()
        return t

    def _loop(self):
        while not self.stop.is_set():
            # a moving player, and one solid cell under it + a hazard ahead
            F = B.field_offset
            x = 400.0 + self.seq * 5.2
            self._w(F("in_level"), 1)
            self._wf(F("x"), x)
            self._wf(F("y"), 105.0)
            self._wf(F("percent"), min(1.0, x / 26724.0))
            self._wf(F("length"), 26724.0)
            grid = bytearray(B.GRID_CELLS)
            # channel 0 (solid), row below player, all columns
            r = B.GRID_H // 2 - 1
            for w in range(B.GRID_W):
                grid[0 * B.GRID_H * B.GRID_W + r * B.GRID_W + w] = 1
            # channel 1 (hazard) at player row, 5 cells ahead
            grid[1 * B.GRID_H * B.GRID_W + (B.GRID_H // 2) * B.GRID_W + 7] = 1
            self.mm[B._O_GRID:B._O_GRID + B.GRID_CELLS] = bytes(grid)

            # commands
            cmd = struct.unpack("<i", self.mm[B._O_CMD_EPOCH:B._O_CMD_EPOCH + 4])[0]
            if cmd != self._last_cmd:
                self._last_cmd = cmd
                op = struct.unpack("<i", self.mm[B._O_CMD_OP:B._O_CMD_OP + 4])[0]
                arg = struct.unpack("<i", self.mm[B._O_CMD_ARG:B._O_CMD_ARG + 4])[0]
                self.commands.append((op, arg))
                self._w(B._O_CMD_ACK, cmd)

            self.seq += 1
            self._w(B.field_offset("state_seq"), self.seq)
            attached = struct.unpack("<i", self.mm[B._O_ATTACHED:B._O_ATTACHED + 4])[0]
            if attached:
                B._k32.SetEvent(self.ev_state)
                if B._k32.WaitForSingleObject(self.ev_action, 2) == 0:
                    ack = struct.unpack("<i", self.mm[B._O_ACTION_SEQ:B._O_ACTION_SEQ + 4])[0]
                    if ack == self.seq:
                        self.answered += 1
                        act = struct.unpack("<i", self.mm[B._O_ACTION:B._O_ACTION + 4])[0]
                        self.actions.append(act)
            time.sleep(self.period)

    def close(self):
        self.stop.set()
        time.sleep(0.05)
        for h in (self.ev_state, self.ev_action):
            B._k32.CloseHandle(h)
        self.mm.close()


# --- tests -------------------------------------------------------------------
_failures = []


def check(name, cond, detail=""):
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond:
        _failures.append(name)


def main():
    mod = FakeMod()
    mod.run()
    time.sleep(0.1)
    b = B.Bridge(tag=TEST_TAG)

    # Mirrors the static_assert in main.cpp — if these disagree the two sides
    # are reading different structs.
    check("struct size matches C++ static_assert", B._FMT.size == 1936, f"{B._FMT.size} bytes")
    check("connects and version-checks", b.connected())

    # --- handshake: every published frame must be answered, none skipped -----
    n = 500
    for _ in range(n):
        st = b.wait_frame(timeout=5.0)
        b.send_action(st["percent"] > 0.5, st["state_seq"])
    check("no frames missed over 500", b.missed == 0, f"missed={b.missed}")
    check("mod saw every action", mod.answered >= n - 2, f"{mod.answered}/{n}")

    # --- state decoding -------------------------------------------------------
    st = b.wait_frame()
    b.send_action(False, st["state_seq"])
    check("in_level decoded", st["in_level"] == 1)
    check("length decoded", abs(st["length"] - 26724.0) < 1e-3, f"{st['length']}")
    check("percent tracks x", abs(st["percent"] - st["x"] / 26724.0) < 1e-4)

    # --- grid -----------------------------------------------------------------
    g = b.read_grid()
    check("grid shape is conv-ready", g.shape == (B.GRID_C, B.GRID_H, B.GRID_W), str(g.shape))
    check("solid floor read back", g[0, B.GRID_H // 2 - 1, :].all())
    check("hazard read at the right cell", g[1, B.GRID_H // 2, 7] == 1)
    check("no phantom occupancy", g[2].sum() == 0 and g[3].sum() == 0)

    # --- action write does not clobber neighbours -----------------------------
    before = b.read()
    b.send_action(True, before["state_seq"])
    after = b.read()
    check("action write is isolated",
          after["speed"] == before["speed"] and after["cmd_epoch"] == before["cmd_epoch"]
          and after["attached"] == 1)

    # --- config + commands ----------------------------------------------------
    b.set_speed(8)
    b.set_fast_respawn(True)
    b.set_mute(True)
    st = b.read()
    check("config fields land", st["speed"] == 8 and st["fast_respawn"] == 1 and st["mute"] == 1)
    b.set_speed(99)
    check("speed clamped to 32", b.read()["speed"] == 32)

    check("reset_from_start acked", b.reset_from_start())
    check("set_practice acked", b.set_practice(True))
    check("respawn_at acked", b.respawn_at(3))
    check("commands arrived in order",
          mod.commands[-3:] == [(B.CMD_RESET_START, 0), (B.CMD_SET_PRACTICE, 1),
                                (B.CMD_RESPAWN_CP, 3)],
          str(mod.commands[-3:]))

    # --- detach ---------------------------------------------------------------
    b.detach()
    st = b.read()
    check("detach releases the mod", st["attached"] == 0 and st["speed"] == 1)

    b.close()
    mod.close()

    print()
    if _failures:
        print(f"{len(_failures)} FAILED: {', '.join(_failures)}")
        return 1
    print("all bridge protocol tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
