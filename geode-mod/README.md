# GDBot Bridge (Geode mod)

Streams live Geometry Dash state to shared memory and applies the agent's jump
action — the robust, offset-free way to sense/control **64-bit GD 2.2**. Geode
resolves all class member offsets per version, so nothing here is version-pinned
to raw addresses.

The shared-memory layout mirrors [`../gdbot/bridge.py`](../gdbot/bridge.py). Both
sides carry a size assert and a protocol version, so the two halves cannot
silently drift — if you change the struct, bump `GDBOT_VERSION` and update
`bridge.py` in the same commit.

## One-time setup

You already have **Visual Studio 2022** (MSVC + bundled CMake/Ninja). You still need
Geode, its SDK/CLI, and clang.

1. **Install the Geode loader into GD** — download the installer from
   <https://geode-sdk.org/> and run it (it detects your Steam GD install). Launch
   GD once; a **Geode** button on the main menu confirms it loaded.

2. **Install clang** (Geode's Windows toolchain):
   ```powershell
   winget install LLVM.LLVM
   ```
   (or add the "C++ Clang tools for Windows" component in the VS Installer.)

3. **Install the Geode CLI** — grab `geode-cli-*-win.zip` from
   <https://github.com/geode-sdk/cli/releases>, extract `geode.exe` somewhere on
   PATH. Then install the SDK (sets the `GEODE_SDK` env var):
   ```powershell
   geode sdk install
   geode sdk install-binaries
   geode config setup   # point the CLI at your GD install
   ```

4. **Match versions** — set `gd.win` in `mod.json` to the GD version your Geode
   targets (the Geode installer / `geode sdk` output tells you; e.g. `2.2074`).

## Build & install the mod

From this folder, in a **"x64 Native Tools Command Prompt for VS 2022"** (so cl,
cmake, ninja are on PATH):

```powershell
geode build
```
This produces `gdbot.bridge.geode`. Install it (either command works):

```powershell
geode install ./build/gdbot.bridge.geode
# ...or copy the .geode into  <GD>\geode\mods\
```

If `geode build` isn't available, configure manually:
```powershell
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build
```

## Test it

1. Launch Geometry Dash (with the mod installed) and enter a level.
2. From the repo root:
   ```powershell
   python -m gdbot.bridge          # live x / y / % / mode / dead as you play
   python -m gdbot.bridge --grid   # ASCII view of the grid the agent perceives
   python -m gdbot.bridge --bench  # throughput + handshake latency by speed
   ```
   The `--grid` view is the one to trust: if the floor, the hazards and the
   blocks do not appear where you see them on screen, perception is wrong and no
   amount of training will fix it.

The protocol itself is covered by `tests/test_bridge.py`, which runs a fake mod
in-process and needs no game:

```powershell
python tests/test_bridge.py
```

Once the grid looks right, `python train.py --level 1` takes it from here.
