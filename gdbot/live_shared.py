"""Python side of the GDBot Bridge shared-memory contract.

Mirrors `struct GDBotShared` in geode-mod/src/main.cpp. The Geode mod (running
inside GeometryDash.exe) writes state + a forward hazard grid every physics
frame; we read it here and write back the jump action.
"""

import mmap
import struct

LOOKAHEAD = 10
# 7 int32, 5 float32, action int32, LOOKAHEAD spike int32s, LOOKAHEAD ground
# float32s, then practice/reset_epoch/checkpoint_count + load_epoch/load_level_id/
# current_level_id int32s.
_FMT = struct.Struct("<7i5fi%di%df6i" % (LOOKAHEAD, LOOKAHEAD))
_SIZE = 4096
_TAG = "GDBotShared"
MAGIC = 0x54444247  # 'GDBT'
_ACTION_OFF = struct.calcsize("<7i5f")                              # action field (48)
_PRACTICE_OFF = struct.calcsize("<7i5fi%di%df" % (LOOKAHEAD, LOOKAHEAD))  # practice (132)
_RESET_OFF = _PRACTICE_OFF + 4                                      # reset_epoch (136)
_LOAD_EPOCH_OFF = struct.calcsize("<7i5fi%di%df3i" % (LOOKAHEAD, LOOKAHEAD))  # load_epoch (144)
_LOAD_ID_OFF = _LOAD_EPOCH_OFF + 4                                  # load_level_id (148)

# Official main levels (id -> name); pass an id to load_level().
OFFICIAL_LEVELS = {
    1: "Stereo Madness", 2: "Back on Track", 3: "Polargeist", 4: "Dry Out",
    5: "Base After Base", 6: "Can't Let Go", 7: "Jumper", 8: "Time Machine",
    9: "Cycles", 10: "xStep", 11: "Clutterfunk", 12: "Theory of Everything",
    13: "Electroman Adventures", 14: "Clubstep", 15: "Electrodynamix",
    16: "Hexagon Force", 17: "Blast Processing", 18: "Theory of Everything 2",
    19: "Geometrical Dominator", 20: "Deadlocked", 21: "Fingerdash",
}

_SCALARS = ("magic", "frame", "in_level", "dead", "on_ground", "gamemode",
            "attempt", "x", "y", "vy", "percent", "length", "action")


class LiveShared:
    """Reader/writer for the mod's shared-memory block."""

    def __init__(self):
        self.mm = mmap.mmap(-1, _SIZE, tagname=_TAG)

    def read(self) -> dict:
        vals = _FMT.unpack(self.mm[:_FMT.size])
        d = dict(zip(_SCALARS, vals[:13]))
        d["spike"] = list(vals[13:13 + LOOKAHEAD])
        d["ground"] = list(vals[13 + LOOKAHEAD:13 + 2 * LOOKAHEAD])
        base = 13 + 2 * LOOKAHEAD
        d["practice"] = vals[base]
        d["reset_epoch"] = vals[base + 1]
        d["checkpoint_count"] = vals[base + 2]
        d["load_epoch"] = vals[base + 3]
        d["load_level_id"] = vals[base + 4]
        d["current_level_id"] = vals[base + 5]
        return d

    def connected(self) -> bool:
        try:
            return _FMT.unpack(self.mm[:_FMT.size])[0] == MAGIC
        except Exception:
            return False

    def set_action(self, jump: bool) -> None:
        """Write only the action field so we never clobber the mod's writes."""
        self.mm[_ACTION_OFF:_ACTION_OFF + 4] = struct.pack("<i", 1 if jump else 0)

    def set_practice(self, on: bool) -> None:
        """Enable/disable practice mode + auto frontier-checkpoints in the mod."""
        self.mm[_PRACTICE_OFF:_PRACTICE_OFF + 4] = struct.pack("<i", 1 if on else 0)

    def request_reset(self) -> None:
        """Ask the mod to clear checkpoints and restart from the level start."""
        epoch = self.read()["reset_epoch"]
        self.mm[_RESET_OFF:_RESET_OFF + 4] = struct.pack("<i", epoch + 1)

    def load_level(self, level_id: int) -> None:
        """Load an official level by id (1-21), leaving the current one.
        level_id 0 leaves to the main menu."""
        self.mm[_LOAD_ID_OFF:_LOAD_ID_OFF + 4] = struct.pack("<i", int(level_id))
        epoch = self.read()["load_epoch"]
        self.mm[_LOAD_EPOCH_OFF:_LOAD_EPOCH_OFF + 4] = struct.pack("<i", epoch + 1)

    def leave_to_menu(self) -> None:
        """Quit the current level back to the main menu."""
        self.load_level(0)

    def close(self) -> None:
        self.mm.close()


if __name__ == "__main__":
    import time
    s = LiveShared()
    print("Waiting for GDBot Bridge mod... (start GD, enter a level)")
    while not s.connected():
        time.sleep(0.5)
    print("connected. Enter a level.")
    last = -1
    while True:
        st = s.read()
        if st["frame"] != last:
            last = st["frame"]
            grid = "".join("#" if x else "." for x in st["spike"])
            print(f"x={st['x']:7.1f} {st['percent']*100:5.1f}% mode={st['gamemode']} "
                  f"dead={st['dead']} grnd={st['on_ground']} spikes[{grid}]")
        time.sleep(0.05)
