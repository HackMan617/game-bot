/*
 * GDBot Bridge — a Geode mod that connects Geometry Dash to the gdbot agent.
 *
 * Every frame it writes exact PlayLayer/PlayerObject state plus a 2D occupancy
 * grid into a named shared-memory block, then blocks on a named event until the
 * agent answers with an action. Geode resolves all class member offsets per GD
 * version, so this works on the 64-bit 2.2 build with NO manual offsets.
 *
 * This generation adds, over the original:
 *   - a 24x16x4 occupancy grid (solid / hazard / orb-pad / portal) instead of a
 *     single row of 10 (spike, height) pairs, including the ground and ceiling
 *     planes, which are GJGroundLayers rather than GameObjects and so were
 *     previously invisible to the agent
 *   - a per-level spatial index, so a frame touches ~24 buckets instead of every
 *     object in the level
 *   - an event handshake, so the agent never misses a frame
 *   - a fixed timestep while driving, so the decision rate is independent of the
 *     render rate, plus the three throttles that were capping training speed
 *     (background pause, vsync, frame pacing)
 *   - fast respawn, skipping the death animation
 *   - one (epoch, op, arg) command channel replacing the per-command epochs
 *
 * Shared-memory layout MUST match gdbot/bridge.py exactly; the static_assert
 * below and the version field are what keep the two honest.
 */

#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <Geode/modify/CCScheduler.hpp>
#include <Geode/modify/AppDelegate.hpp>
#include <Geode/modify/CCEGLView.hpp>
#include <Geode/binding/GameLevelManager.hpp>
#include <Geode/binding/MenuLayer.hpp>
#include <Geode/binding/FMODAudioEngine.hpp>
#include <Geode/binding/CheckpointObject.hpp>
#include <Geode/binding/PlayerCheckpoint.hpp>
#include <Geode/binding/GJGameLevel.hpp>
#include <windows.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

using namespace geode::prelude;

#define GDBOT_VERSION   5    // bump whenever the shared layout changes
#define GRID_W          24    // cells wide  (GRID_BEHIND behind the player, rest ahead)
#define GRID_H          16    // cells tall  (player sits on row GRID_H/2)
#define GRID_C          4     // channels: 0 solid, 1 hazard, 2 orb/pad, 3 portal
#define GRID_BEHIND     2
#define GRID_CELLS      (GRID_C * GRID_H * GRID_W)
#define CELL            30.0f // 1 block = 30 GD units
#define MAX_CP          64    // checkpoint percents reported to the agent
#define SHARED_BYTES    65536
// Bounded, so a stalled agent can never hang the game. This is deliberately
// generous: with vsync bypassed the handshake IS the rate limiter, so the game
// runs exactly as fast as the agent can think. Too small and a slow inference
// step shows up as dropped frames instead of just a slower game.
#define ACTION_WAIT_MS  20
#define ATTACH_TIMEOUT  600   // consecutive unanswered frames before we detach

// Command opcodes (agent -> mod). Bump cmd_epoch to execute; the mod echoes it
// back in cmd_ack once done.
enum : int32_t {
    CMD_NONE          = 0,
    CMD_RESET_START   = 1,  // clear checkpoints, restart from 0%
    CMD_LOAD_LEVEL    = 2,  // arg = official level id 1-21, or 0 for the menu
    CMD_SET_PRACTICE  = 3,  // arg = 1 on / 0 off
    CMD_RESPAWN_CP    = 4,  // arg = checkpoint index to respawn at (truncates later ones)
    CMD_CLEAR_CP      = 5,  // drop all checkpoints
};

#pragma pack(push, 1)
struct GDBotShared {
    // --- header / handshake ---
    int32_t magic;            // 'GDBT' = 0x54444247
    int32_t version;          // GDBOT_VERSION; the agent refuses a mismatch
    int32_t state_seq;        // mod -> agent: increments every physics frame
    int32_t action_seq;       // agent -> mod: the state_seq this action answers
    // --- state (mod -> agent) ---
    int32_t in_level;
    int32_t dead;
    int32_t on_ground;
    int32_t gamemode;         // matches the gdbot.obs ids (0 = cube)
    int32_t attempt;
    int32_t upside_down;
    int32_t level_id;         // 0 = not in a level
    int32_t checkpoint_count;
    int32_t is_practice;      // mod -> agent: practice mode actually engaged
    float   x, y;
    float   vy;
    float   percent;          // 0..1
    float   length;           // level length in x units
    float   ground_y;         // top of the floor plane (not a GameObject, so the
                              // grid has to synthesise it — see the fill below)
    float   ceiling_y;        // bottom of the ceiling plane, 0 if there isn't one
    float   player_speed;     // speed portals scale how fast x advances
    float   gravity_mod;      // gravity portals flip/scale the fall
    float   vehicle_size;     // mini portals halve the hitbox and jump arc
    // --- action (agent -> mod) ---
    int32_t action;           // 1 = hold jump, 0 = release
    // --- command channel (agent -> mod) ---
    int32_t cmd_epoch;
    int32_t cmd_op;
    int32_t cmd_arg;
    int32_t cmd_ack;          // mod -> agent: last cmd_epoch executed
    // --- config (agent -> mod, read every frame) ---
    int32_t speed;            // physics steps per rendered frame, 1..32
    int32_t fast_respawn;     // 1 = reset immediately on death, skipping the animation
    int32_t mute;             // 1 = silence music (speed makes it unbearable)
    int32_t attached;         // 1 = an agent is driving; 0 disables the handshake
    int32_t step_hz;          // decisions per second of GAME time (fixed dt); 0 = 60
    // --- perception ---
    int32_t grid_w, grid_h, grid_c;
    float   cp_pct[MAX_CP];   // mod -> agent: percent of each active checkpoint
    uint8_t grid[GRID_CELLS]; // channel-first (C, H, W) so it reshapes into a conv
};
#pragma pack(pop)

// If this fires, gdbot/bridge.py's _FMT no longer describes this struct — fix
// both sides together, and bump GDBOT_VERSION so an old client refuses to run.
static_assert(sizeof(GDBotShared) == 1936, "shared layout changed; sync bridge.py");

static GDBotShared* g_shared      = nullptr;
static HANDLE       g_map         = nullptr;
static HANDLE       g_evState     = nullptr;  // mod signals: a frame is ready
static HANDLE       g_evAction    = nullptr;  // agent signals: the action is written
static int          g_lastAction  = 0;
static bool         g_practiceOn  = false;
static int          g_lastCmd     = 0;
static float        g_lastCpX     = -1e9f;
static int          g_currentLvl  = 0;
static bool         g_needReset   = false;
static int          g_unanswered  = 0;
static bool         g_muted       = false;
static int          g_appliedSpeed = 1;
static float        g_lastRespawnX = 0.f;  // how far the last attempt got
static int          g_deadRespawns = 0;    // respawns in a row that went nowhere
static int          g_resetDelay   = 0;    // frames waited so far before respawning
static int          g_menuAttached = 0;    // ticks attached with no level loaded

// Per-level spatial index: objects bucketed by (int)(x / CELL), so a frame only
// scans the ~GRID_W buckets in front of the player instead of every object.
static std::vector<std::vector<GameObject*>> g_cols;
static int g_indexedCount = -1;

static void ensureShared() {
    if (g_shared) return;
    g_map = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
                               0, SHARED_BYTES, "GDBotShared");
    if (!g_map) return;
    g_shared = static_cast<GDBotShared*>(
        MapViewOfFile(g_map, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(GDBotShared)));
    if (!g_shared) return;

    std::memset(g_shared, 0, sizeof(GDBotShared));
    g_shared->magic   = 0x54444247;
    g_shared->version = GDBOT_VERSION;
    g_shared->speed   = 1;
    g_shared->step_hz = 60;
    g_shared->grid_w  = GRID_W;
    g_shared->grid_h  = GRID_H;
    g_shared->grid_c  = GRID_C;

    // Auto-reset events: each SetEvent releases exactly one waiter.
    g_evState  = CreateEventA(nullptr, FALSE, FALSE, "GDBotStateReady");
    g_evAction = CreateEventA(nullptr, FALSE, FALSE, "GDBotActionReady");
    log::info("GDBot Bridge v{}: shared memory + handshake ready ({} bytes)",
              GDBOT_VERSION, (int)sizeof(GDBotShared));
}

// gamemode ids must match gdbot/obs.py
static int currentMode(PlayerObject* p) {
    if (p->m_isShip)   return 1; // SHIP
    if (p->m_isBall)   return 2; // BALL
    if (p->m_isBird)   return 3; // UFO   (bindings call it "bird")
    if (p->m_isDart)   return 4; // WAVE  (bindings call it "dart")
    if (p->m_isRobot)  return 5; // ROBOT
    if (p->m_isSpider) return 6; // SPIDER
    if (p->m_isSwing)  return 7; // SWING (2.2)
    return 0;                    // CUBE
}

// Map a GameObjectType onto a grid channel, or -1 to ignore it. Decoration,
// triggers, coins and collision boxes carry no information the agent can act on.
static int channelOf(GameObjectType t) {
    switch (t) {
        case GameObjectType::Solid:
        case GameObjectType::Slope:
        case GameObjectType::Breakable:
            return 0;
        case GameObjectType::Hazard:
        case GameObjectType::AnimatedHazard:   // saw blades — v1 missed these entirely
            return 1;
        case GameObjectType::YellowJumpPad:
        case GameObjectType::PinkJumpPad:
        case GameObjectType::GravityPad:
        case GameObjectType::RedJumpPad:
        case GameObjectType::SpiderPad:
        case GameObjectType::YellowJumpRing:
        case GameObjectType::PinkJumpRing:
        case GameObjectType::GravityRing:
        case GameObjectType::RedJumpRing:
        case GameObjectType::GreenRing:
        case GameObjectType::DropRing:
        case GameObjectType::CustomRing:
        case GameObjectType::DashRing:
        case GameObjectType::GravityDashRing:
        case GameObjectType::SpiderOrb:
        case GameObjectType::TeleportOrb:
            return 2;
        case GameObjectType::InverseGravityPortal:
        case GameObjectType::NormalGravityPortal:
        case GameObjectType::GravityTogglePortal:
        case GameObjectType::ShipPortal:
        case GameObjectType::CubePortal:
        case GameObjectType::BallPortal:
        case GameObjectType::UfoPortal:
        case GameObjectType::WavePortal:
        case GameObjectType::RobotPortal:
        case GameObjectType::SpiderPortal:
        case GameObjectType::SwingPortal:
        case GameObjectType::InverseMirrorPortal:
        case GameObjectType::NormalMirrorPortal:
        case GameObjectType::RegularSizePortal:
        case GameObjectType::MiniSizePortal:
        case GameObjectType::DualPortal:
        case GameObjectType::SoloPortal:
        case GameObjectType::TeleportPortal:
            return 3;
        default:
            return -1;
    }
}

static void clearIndex() {
    g_cols.clear();
    g_indexedCount = -1;
}

// Rebuild the column index. Cheap enough to do on level load only; we detect a
// new level by the object count changing, which also self-heals after a reload.
static void buildIndex(CCArray* objects) {
    g_cols.clear();
    if (!objects) { g_indexedCount = 0; return; }
    g_indexedCount = objects->count();

    int maxCol = 0;
    for (auto obj : CCArrayExt<GameObject*>(objects)) {
        if (channelOf(obj->getType()) < 0) continue;
        int c = (int)std::floor(obj->getPositionX() / CELL);
        if (c < 0) c = 0;
        maxCol = std::max(maxCol, c);
    }
    g_cols.resize(maxCol + 1);
    for (auto obj : CCArrayExt<GameObject*>(objects)) {
        if (channelOf(obj->getType()) < 0) continue;
        int c = (int)std::floor(obj->getPositionX() / CELL);
        if (c < 0) c = 0;
        g_cols[c].push_back(obj);
    }
    log::info("GDBot Bridge: indexed {} objects into {} columns",
              g_indexedCount, (int)g_cols.size());
}

class $modify(BridgePlayLayer, PlayLayer) {
    // PlayLayer has no own update(); the per-frame method it overrides is
    // postUpdate (runs each frame after physics). GJBaseGameLayer::update is the
    // base loop, but postUpdate keeps `this` a PlayLayer so all fields resolve.
    void postUpdate(float dt) {
        PlayLayer::postUpdate(dt);
        ensureShared();
        if (!g_shared) return;

        // NOTE: fast respawn is deliberately NOT handled here. Calling
        // resetLevel() from inside postUpdate lands in the middle of PlayLayer's
        // own update chain, mid death sequence, and the respawned player dies again
        // on the very next frame — an unrecoverable death/reset loop that spins the
        // attempt counter forever (measured: 4000+ attempts, 0.00%, ~18fps). The
        // reset now happens at the top of the frame in the scheduler hook instead.

        auto p = m_player1;
        if (!p) return;
        const float px = p->getPositionX();
        const float py = p->getPositionY();

        g_shared->in_level    = 1;
        g_shared->dead        = p->m_isDead ? 1 : 0;
        g_shared->on_ground   = p->m_isOnGround ? 1 : 0;
        g_shared->gamemode    = currentMode(p);
        g_shared->upside_down = p->m_isUpsideDown ? 1 : 0;
        g_shared->attempt     = m_attempts;
        g_shared->is_practice = m_isPracticeMode ? 1 : 0;
        g_shared->player_speed  = p->m_playerSpeed;
        g_shared->gravity_mod   = p->m_gravityMod;
        g_shared->vehicle_size  = p->m_vehicleSize;
        g_shared->x           = px;
        g_shared->y           = py;
        g_shared->vy          = static_cast<float>(p->m_yVelocity);
        g_shared->length      = m_levelLength;
        g_shared->percent     = m_levelLength > 0.f ? px / m_levelLength : 0.f;
        // Read the id off the level itself, so it's right whether the agent
        // loaded it or the player picked it by hand.
        if (m_level) g_currentLvl = m_level->m_levelID.value();
        g_shared->level_id    = g_currentLvl;

        // --- occupancy grid -------------------------------------------------
        // Cells are 30 units square. Column 0 sits GRID_BEHIND cells behind the
        // player; row GRID_H/2 is the player's own row.
        if (m_objects && (int)m_objects->count() != g_indexedCount) buildIndex(m_objects);
        std::memset(g_shared->grid, 0, GRID_CELLS);

        // The floor and the ship ceiling are GJGroundLayers, not GameObjects, so
        // nothing in m_objects marks them. Without this the network cannot see
        // the ground it is standing on. Both planes move (ship sections raise
        // the floor), so read them live rather than assuming a constant.
        const float groundY  = m_groundLayer  ? m_groundLayer->getPositionY()  : 0.f;
        const float ceilingY = m_groundLayer2 ? m_groundLayer2->getPositionY() : 0.f;
        g_shared->ground_y  = groundY;
        g_shared->ceiling_y = ceilingY;
        for (int r = 0; r < GRID_H; r++) {
            const float cellY = py + (r - GRID_H / 2) * CELL;
            const bool blocked = (cellY < groundY) ||
                                 (ceilingY > groundY && cellY > ceilingY);
            if (!blocked) continue;
            for (int w = 0; w < GRID_W; w++)
                g_shared->grid[r * GRID_W + w] = 1;   // channel 0 = solid
        }

        const int baseCol = (int)std::floor(px / CELL) - GRID_BEHIND;
        for (int w = 0; w < GRID_W; w++) {
            const int c = baseCol + w;
            if (c < 0 || c >= (int)g_cols.size()) continue;
            for (auto obj : g_cols[c]) {
                const int ch = channelOf(obj->getType());
                if (ch < 0) continue;
                const int r = (int)std::floor((obj->getPositionY() - py) / CELL) + GRID_H / 2;
                if (r < 0 || r >= GRID_H) continue;
                g_shared->grid[ch * GRID_H * GRID_W + r * GRID_W + w] = 1;
            }
        }

        // --- practice checkpoints -------------------------------------------
        // In practice mode GD respawns at the last checkpoint, so dropping one
        // every few blocks lets the agent grind a segment instead of replaying 0%.
        if (g_practiceOn && !p->m_isDead && px > g_lastCpX + 200.f) {
            this->createCheckpoint();
            g_lastCpX = px;
        }
        const int cpn = m_checkpointArray ? m_checkpointArray->count() : 0;
        g_shared->checkpoint_count = std::min(cpn, MAX_CP);
        for (int i = 0; i < std::min(cpn, MAX_CP); i++) {
            auto cp = static_cast<CheckpointObject*>(m_checkpointArray->objectAtIndex(i));
            float cx = cp && cp->m_player1Checkpoint ? cp->m_player1Checkpoint->m_position.x : 0.f;
            g_shared->cp_pct[i] = m_levelLength > 0.f ? cx / m_levelLength : 0.f;
        }

        // --- handshake --------------------------------------------------------
        // Publish the frame, then give the agent a bounded window to answer. The
        // bound is what keeps a stalled or crashed agent from freezing the game.
        g_shared->state_seq += 1;
        if (g_shared->attached && g_evState && g_evAction) {
            // Drop any answer left over from an earlier frame before advertising
            // this one. Without this a single timeout desyncs the pair forever:
            // the stale signal satisfies the wait immediately, the seq check fails,
            // and the mod stops waiting at all from then on — measured as the game
            // free-running at ~240fps while the agent caught only 1 frame in 4.
            ResetEvent(g_evAction);
            SetEvent(g_evState);

            bool answered = false;
            const DWORD deadline = GetTickCount() + ACTION_WAIT_MS;
            for (;;) {
                const DWORD now = GetTickCount();
                const DWORD left = (now >= deadline) ? 0 : (deadline - now);
                if (WaitForSingleObject(g_evAction, left) != WAIT_OBJECT_0) break;
                if (g_shared->action_seq == g_shared->state_seq) { answered = true; break; }
                if (left == 0) break;   // an older frame's answer; out of time
            }

            if (answered) {
                g_unanswered = 0;
            } else if (++g_unanswered > ATTACH_TIMEOUT) {
                log::warn("GDBot Bridge: agent stopped answering — detaching");
                g_shared->attached = 0;
                g_shared->speed = 1;
                g_shared->action = 0;
                g_unanswered = 0;
            }
        }

        // Apply the action as an edge-triggered jump (the same path as real input).
        const int act = g_shared->action;
        if (act && !g_lastAction)       this->handleButton(true, 1, true);
        else if (!act && g_lastAction)  this->handleButton(false, 1, true);
        g_lastAction = act;
    }

    void destroyPlayer(PlayerObject* player, GameObject* object) {
        PlayLayer::destroyPlayer(player, object);
        if (g_shared && g_shared->fast_respawn && player == m_player1 && !m_hasCompletedLevel) {
            g_lastRespawnX = player->getPositionX();   // feeds the loop backstop
            g_needReset = true;
        }
    }

    void resetLevel() {
        PlayLayer::resetLevel();
        if (g_shared) g_shared->dead = 0;
        g_lastAction = 0;
        g_needReset = false;
        // Checkpoints survive a respawn, so resume dropping them past the last one.
        const int cpn = m_checkpointArray ? m_checkpointArray->count() : 0;
        if (cpn == 0) g_lastCpX = -1e9f;
    }

    void onQuit() {
        PlayLayer::onQuit();
        if (g_shared) { g_shared->in_level = 0; g_shared->dead = 0; g_shared->level_id = 0; }
        g_lastAction = 0;
        g_practiceOn = false;
        g_lastCpX = -1e9f;
        clearIndex();
    }

    // --- commands, dispatched from the scheduler tick -------------------------
    void gdbotCommand(int op, int arg) {
        switch (op) {
            case CMD_RESET_START:
                this->removeAllCheckpoints();
                this->resetLevelFromStart();
                g_lastCpX = -1e9f;
                break;
            case CMD_SET_PRACTICE: {
                const bool want = arg != 0;
                if (want != g_practiceOn) {
                    this->togglePracticeMode(want);
                    g_practiceOn = want;
                    g_lastCpX = -1e9f;
                }
                break;
            }
            case CMD_CLEAR_CP:
                this->removeAllCheckpoints();
                g_lastCpX = -1e9f;
                break;
            case CMD_RESPAWN_CP: {
                // Keep the first (arg+1) checkpoints and drop the rest, so GD's
                // respawn lands on checkpoint `arg`.
                if (!m_checkpointArray) break;
                const int keep = std::max(1, arg + 1);
                while ((int)m_checkpointArray->count() > keep)
                    this->removeCheckpoint(false);
                const int cpn = m_checkpointArray->count();
                if (cpn > 0) {
                    auto cp = static_cast<CheckpointObject*>(
                        m_checkpointArray->objectAtIndex(cpn - 1));
                    if (cp && cp->m_player1Checkpoint)
                        g_lastCpX = cp->m_player1Checkpoint->m_position.x;
                }
                this->resetLevel();
                break;
            }
            default: break;
        }
    }
};

// Keep simulating while the window is in the background. Without this GD drops
// to ~23fps the moment it loses focus, which caps training at 0.4x real time
// (measured) — useless, since you are never looking at the game while it trains.
class $modify(BridgeApp, AppDelegate) {
    void applicationDidEnterBackground() {
        if (g_shared && g_shared->attached) return;
        AppDelegate::applicationDidEnterBackground();
    }
};

// Presenting a frame blocks on vsync, which pins the render loop to the monitor
// (~144Hz here). Since one rendered frame = one agent decision, that vsync wait
// is the hard ceiling on training speed. Skipping most presents removes it; the
// window still updates every `speed` frames so you can watch.
static int g_swapCount = 0;
class $modify(BridgeView, CCEGLView) {
    void swapBuffers() {
        if (g_shared && g_shared->attached) {
            const int n = std::clamp(g_shared->speed, 1, 32);
            if (n > 1 && (++g_swapCount % n) != 0) return;
        }
        CCEGLView::swapBuffers();
    }
};

// Global per-frame tick. Fixes the timestep while an agent drives, and handles
// commands even on the menu so the agent can leave a level and pick another.
class $modify(BridgeScheduler, CCScheduler) {
    void update(float dt) {
        ensureShared();

        // While an agent is driving we step with a FIXED dt instead of the real
        // frame time. postUpdate fires once per rendered frame, so this pins the
        // agent to exactly step_hz decisions per second of GAME time no matter
        // how fast the game is actually rendering — otherwise the decision rate
        // would follow the monitor (measured 140Hz here) and a policy trained on
        // one machine would not transfer to another.
        //
        // Wall-clock speedup is therefore just render_fps / step_hz. We do NOT
        // loop the scheduler to go faster: GD gates its own stepping internally,
        // so extra calls are no-ops (measured — 16 calls produced 1 postUpdate).
        // "Driving" means an agent is attached AND there is a level for it to
        // drive. Everything gated on this must stay off on the menu.
        const bool inLevel = PlayLayer::get() != nullptr;
        const bool driving = g_shared && g_shared->attached && inLevel;

        // Watchdog: an agent that dies without detaching (a crash, a kill, an
        // exception before its cleanup runs) would otherwise leave `attached` set
        // forever. postUpdate's own timeout cannot help — it needs a PlayLayer,
        // and the stranded case is precisely the one without one.
        //
        // Deliberately slack. A first attempt at 10s fired during GD's own
        // menu-to-level transition and detached a perfectly healthy agent mid
        // startup. The counter also resets on any command, because an agent still
        // issuing commands is by definition alive — so this only ever fires on a
        // genuinely dead one.
        if (g_shared && g_shared->attached && !inLevel) {
            if (++g_menuAttached > 7200) {          // ~2 minutes at 60fps
                log::warn("GDBot Bridge: attached with no level for 2min — releasing");
                g_shared->attached = 0;
                g_shared->speed = 1;
                g_shared->action = 0;
                g_menuAttached = 0;
            }
        } else {
            g_menuAttached = 0;
        }

        // Fast respawn, at the TOP of the frame and outside PlayLayer's update
        // chain. Doing it from postUpdate re-entered the death sequence and looped
        // forever (see the note there).
        // Let the death sequence actually finish before restarting. Resetting on
        // the very next frame re-enters it and the respawned player dies again;
        // a few frames of slack is still ~50ms against GD's ~1s auto-retry.
        if (g_shared && g_needReset && ++g_resetDelay >= 4) {
            g_needReset = false;
            g_resetDelay = 0;
            if (auto pl = PlayLayer::get()) {
                // Backstop: if respawning never gets the player moving, something is
                // wrong, and spinning resets makes the game unusable. Give up on
                // fast respawn rather than lock the level into a loop.
                g_deadRespawns = (g_lastRespawnX >= 1.f) ? 0 : g_deadRespawns + 1;
                if (g_deadRespawns > 20) {
                    g_shared->fast_respawn = 0;
                    g_deadRespawns = 0;
                    log::warn("GDBot Bridge: fast respawn made no progress 20x — "
                              "disabling it so the level can recover");
                } else {
                    pl->resetLevel();
                }
            }
        }

        if (driving) {
            const int hz = g_shared->step_hz > 0 ? g_shared->step_hz : 60;
            dt = 1.0f / static_cast<float>(hz);
        }
        CCScheduler::update(dt);

        if (!g_shared) return;

        // Collapse the frame-pacing sleep while an agent drives A LEVEL. This is
        // NOT a speedup — it is what keeps the handshake in lockstep. Removing it
        // was tried and measured: the render loop free-ran at ~240fps, the mod
        // outran the agent, and 1650 of every 2250 published frames went
        // unanswered. With it, every speed setting reports zero missed frames.
        //
        // The `pl` term matters as much as the `attached` one. The handshake only
        // exists inside PlayLayer, so collapsing the interval on the menu buys
        // nothing and spins the game into an unresponsive white window — measured,
        // after an agent died holding attached=1 with no level loaded.
        //
        // The 60 steps/s ceiling is not ours to lift: GD gates its own stepping, so
        // postUpdate never fires faster however much render work is skipped (16
        // CCScheduler::update calls were measured to produce exactly one postUpdate).
        // Live training runs at about 1x real time, and that is the honest number.
        //
        // Known cost: while attached the window stops presenting new frames, so the
        // game looks frozen on a stale image even though it is training correctly.
        // Detach (or stop the trainer) to watch it play.
        const int wantSpeed = driving ? std::clamp(g_shared->speed, 1, 32) : 0;
        if (wantSpeed != g_appliedSpeed) {
            CCDirector::get()->setAnimationInterval(driving ? 1.0 / 100000.0 : 1.0 / 60.0);
            g_appliedSpeed = wantSpeed;
        }
        const bool wantMute = g_shared->mute != 0;
        if (wantMute != g_muted) {
            FMODAudioEngine::sharedEngine()->setBackgroundMusicVolume(wantMute ? 0.f : 1.f);
            g_muted = wantMute;
        }

        auto pl = PlayLayer::get();
        if (!pl) {   // accurate on the menu: onQuit is skipped on replaceScene
            g_shared->in_level = 0;
            g_shared->level_id = 0;
        }

        if (g_shared->cmd_epoch != g_lastCmd) {
            g_lastCmd = g_shared->cmd_epoch;
            g_menuAttached = 0;   // an agent issuing commands is alive; see above
            const int op = g_shared->cmd_op, arg = g_shared->cmd_arg;
            if (op == CMD_LOAD_LEVEL) {
                g_practiceOn = false;
                g_lastCpX = -1e9f;
                clearIndex();
                if (arg > 0) {
                    if (auto level = GameLevelManager::get()->getMainLevel(arg, false)) {
                        CCDirector::get()->replaceScene(PlayLayer::scene(level, false, false));
                        g_currentLvl = arg;
                    }
                } else {
                    CCDirector::get()->replaceScene(MenuLayer::scene(false));
                    g_currentLvl = 0;
                }
            } else if (pl) {
                static_cast<BridgePlayLayer*>(pl)->gdbotCommand(op, arg);
            }
            g_shared->cmd_ack = g_lastCmd;
        }
    }
};
