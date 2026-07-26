"""Python side of the GDBot Bridge shared-memory contract.

Mirrors `struct GDBotShared` in geode-mod/src/main.cpp. The Geode mod (running
inside GeometryDash.exe) writes state every physics frame; we read it here and
write back the jump action. Same-session, page-file-backed named mapping.
"""

import mmap
import struct

# 7 int32, 5 float32, 1 int32  (see GDBotShared)
_FMT = struct.Struct("<7i5fi")
_SIZE = 4096
_TAG = "GDBotShared"
MAGIC = 0x54444247  # 'GDBT'
_ACTION_OFF = _FMT.size - 4  # last field

_FIELDS = ("magic", "frame", "in_level", "dead", "on_ground", "gamemode",
           "attempt", "x", "y", "vy", "percent", "length", "action")


class LiveShared:
    """Reader/writer for the mod's shared-memory block."""

    def __init__(self):
        # tagname opens the same named mapping the mod created (or creates it
        # first if Python starts before the mod).
        self.mm = mmap.mmap(-1, _SIZE, tagname=_TAG)

    def read(self) -> dict:
        return dict(zip(_FIELDS, _FMT.unpack(self.mm[:_FMT.size])))

    def connected(self) -> bool:
        """True once the mod has initialized the block (magic stamped)."""
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
    # Quick live probe: python -m gdbot.live_shared
    import time
    s = LiveShared()
    print("Waiting for GDBot Bridge mod... (start GD with the mod installed)")
    while not s.connected():
        time.sleep(0.5)
    print("connected. Enter a level.")
    last = -1
    while True:
        st = s.read()
        if st["frame"] != last:
            last = st["frame"]
            print(f"in_level={st['in_level']} x={st['x']:.1f} y={st['y']:.1f} "
                  f"{st['percent']*100:.1f}% mode={st['gamemode']} "
                  f"dead={st['dead']} grnd={st['on_ground']} att={st['attempt']}")
        time.sleep(0.05)
