/*
 * GDBot Bridge — a Geode mod that connects Geometry Dash to the gdbot agent.
 *
 * Every physics frame it writes exact PlayLayer/PlayerObject state into a named
 * shared-memory block, and reads back a single "jump" action to drive the game.
 * Geode resolves all class member offsets per GD version, so this works on the
 * 64-bit 2.2 build with NO manual offsets.
 *
 * Shared-memory layout MUST match gdbot/live_shared.py exactly.
 *
 * NOTE: a few member names below (m_isBird / m_isDart / m_isOnGround /
 * m_yVelocity / m_isDead / m_attempts / m_levelLength / handleButton) come from
 * the Geode bindings and may differ slightly by SDK version. If the build errors
 * on one, hover it in your editor (Geode headers) for the exact name and adjust.
 */

#include <Geode/Geode.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <windows.h>
#include <cstdint>

using namespace geode::prelude;

#define GDBOT_LOOKAHEAD 10   // forward cells; 1 cell = 1 block = 30 GD units

#pragma pack(push, 1)
struct GDBotShared {
    int32_t magic;      // 'GDBT' = 0x54444247
    int32_t frame;      // increments every physics frame (freshness)
    int32_t in_level;   // 1 while a PlayLayer is active
    int32_t dead;       // player died this attempt
    int32_t on_ground;  // player is on a surface
    int32_t gamemode;   // matches gdbot.game_state ids (0=cube ...)
    int32_t attempt;    // current attempt number
    float   x, y;       // player position
    float   vy;         // vertical velocity
    float   percent;    // 0..1 progress
    float   length;     // level length (x units)
    int32_t action;     // Python -> mod: 1 = hold jump, 0 = release
    int32_t spike[GDBOT_LOOKAHEAD];   // 1 if a Hazard sits in that forward cell
    float   ground[GDBOT_LOOKAHEAD];  // height of the highest Solid top in that
                                      // cell relative to the player (units; 0 = flat)
    // --- practice-mode curriculum ---
    int32_t practice;        // Python -> mod: 1 = practice mode + auto frontier checkpoints
    int32_t reset_epoch;     // Python -> mod: increment to clear checkpoints & restart from start
    int32_t checkpoint_count;// mod -> Python: number of active checkpoints
};
#pragma pack(pop)

static GDBotShared* g_shared = nullptr;
static HANDLE       g_map = nullptr;
static int          g_lastAction = 0;
static bool         g_practiceOn = false;
static int          g_lastResetEpoch = 0;
static float        g_lastCpX = -1e9f;

static void ensureShared() {
    if (g_shared) return;
    g_map = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
                               0, 4096, "GDBotShared");
    if (!g_map) return;
    g_shared = static_cast<GDBotShared*>(
        MapViewOfFile(g_map, FILE_MAP_ALL_ACCESS, 0, 0, sizeof(GDBotShared)));
    if (g_shared) {
        g_shared->magic = 0x54444247;
        g_shared->action = 0;
        log::info("GDBot Bridge: shared memory ready");
    }
}

// gamemode ids must match gdbot/game_state.py
static int currentMode(PlayerObject* p) {
    if (p->m_isShip)   return 1; // SHIP
    if (p->m_isBird)   return 3; // UFO  (bindings call it "bird")
    if (p->m_isBall)   return 2; // BALL
    if (p->m_isDart)   return 4; // WAVE (bindings call it "dart")
    if (p->m_isRobot)  return 5; // ROBOT
    if (p->m_isSpider) return 6; // SPIDER
    // if (p->m_isSwing) return 7; // SWING (uncomment once confirmed for your SDK)
    return 0;                    // CUBE
}

class $modify(BridgePlayLayer, PlayLayer) {
    // PlayLayer has no own update(); the per-frame method it overrides is
    // postUpdate (runs each frame after physics). GJBaseGameLayer::update is the
    // base loop, but postUpdate keeps `this` a PlayLayer so all fields resolve.
    void postUpdate(float dt) {
        PlayLayer::postUpdate(dt);
        ensureShared();
        if (!g_shared) return;

        auto p = m_player1;
        g_shared->frame     += 1;
        g_shared->in_level   = 1;
        g_shared->dead       = p->m_isDead ? 1 : 0;
        g_shared->on_ground  = p->m_isOnGround ? 1 : 0;
        g_shared->gamemode   = currentMode(p);
        g_shared->attempt    = m_attempts;
        g_shared->x          = p->getPositionX();
        g_shared->y          = p->getPositionY();
        g_shared->vy         = static_cast<float>(p->m_yVelocity);
        g_shared->length     = m_levelLength;
        g_shared->percent    = m_levelLength > 0.f ? p->getPositionX() / m_levelLength : 0.f;

        // --- forward grid (1 block = 30 units): per cell ahead, flag Hazards
        // (spikes to jump over) and record the tallest Solid top relative to the
        // player (blocks to jump onto), so the network reacts to what it "sees".
        for (int i = 0; i < GDBOT_LOOKAHEAD; i++) { g_shared->spike[i] = 0; g_shared->ground[i] = 0.f; }
        if (m_objects) {
            float px = p->getPositionX();
            float py = p->getPositionY();
            for (auto obj : CCArrayExt<GameObject*>(m_objects)) {
                float dx = obj->getPositionX() - px;
                if (dx <= 0.f || dx > GDBOT_LOOKAHEAD * 30.f) continue;
                int i = (int)(dx / 30.f);
                if (i < 0 || i >= GDBOT_LOOKAHEAD) continue;
                auto t = obj->getType();
                if (t == GameObjectType::Hazard) {
                    g_shared->spike[i] = 1;
                } else if (t == GameObjectType::Solid || t == GameObjectType::Slope) {
                    float rel = (obj->getPositionY() + 15.f) - py;  // block top vs cube
                    if (rel > g_shared->ground[i]) g_shared->ground[i] = rel;
                }
            }
        }

        // --- practice-mode curriculum: in practice mode, drop a checkpoint every
        // few blocks of new progress; GD respawns at the last checkpoint on death,
        // so the agent grinds each segment instead of restarting from 0%.
        bool wantPractice = g_shared->practice != 0;
        if (wantPractice != g_practiceOn) {
            this->togglePracticeMode(wantPractice);
            g_practiceOn = wantPractice;
            g_lastCpX = -1e9f;
        }
        if (g_shared->reset_epoch != g_lastResetEpoch) {
            g_lastResetEpoch = g_shared->reset_epoch;
            this->removeAllCheckpoints();
            this->resetLevelFromStart();
            g_lastCpX = -1e9f;
        }
        if (g_practiceOn && !p->m_isDead) {
            float cx = p->getPositionX();
            if (cx > g_lastCpX + 200.f) {   // ~6.5 blocks between checkpoints
                this->createCheckpoint();
                g_lastCpX = cx;
            }
        }
        g_shared->checkpoint_count = m_checkpointArray ? m_checkpointArray->count() : 0;

        // Apply the agent's action as an edge-triggered jump (same path as real input).
        int act = g_shared->action;
        if (act && !g_lastAction) this->handleButton(true, 1, true);
        else if (!act && g_lastAction) this->handleButton(false, 1, true);
        g_lastAction = act;
    }

    void resetLevel() {
        PlayLayer::resetLevel();
        if (g_shared) g_shared->dead = 0;
        g_lastAction = 0;
    }

    void onQuit() {
        PlayLayer::onQuit();
        if (g_shared) { g_shared->in_level = 0; g_shared->dead = 0; }
        g_lastAction = 0;
        g_practiceOn = false;
        g_lastCpX = -1e9f;
    }
};
