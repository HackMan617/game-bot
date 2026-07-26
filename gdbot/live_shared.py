"""Python side of the GDBot Bridge shared-memory contract.

Mirrors `struct GDBotShared` in geode-mod/src/main.cpp. The Geode mod (running
inside GeometryDash.exe) writes state + a forward hazard grid every physics
frame; we read it here and write back the jump action.
"""

import mmap
import struct

LOOKAHEAD = 10
# 7 int32, 5 float32, action int32, LOOKAHEAD spike int32s, LOOKAHEAD ground float32s
_FMT = struct.Struct("<7i5fi%di%df" % (LOOKAHEAD, LOOKAHEAD))
_SIZE = 4096
_TAG = "GDBotShared"
MAGIC = 0x54444247  # 'GDBT'
_ACTION_OFF = struct.calcsize("<7i5f")  # byte offset of the action field (48)

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
        return d

    def connected(self) -> bool:
        try:
            return _FMT.unpack(self.mm[:_FMT.size])[0] == MAGIC
        except Exception:
            return False

    def set_action(self, jump: bool) -> None:
        """Write only the action field so we never clobber the mod's writes."""
        self.mm[_ACTION_OFF:_ACTION_OFF + 4] = struct.pack("<i", 1 if jump else 0)

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
