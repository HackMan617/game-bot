"""MarI/O-style live visualization of a NEAT network.

Draws the network the way SethBling's MarI/O does: input "vision" on the left,
the JUMP output on the right, connections colored by weight sign (green = +,
red = -) and thickness by magnitude, and nodes brightening as they activate.

A neat-python FeedForwardNetwork exposes everything we need after `activate()`:
  net.input_nodes / net.output_nodes  -> node keys
  net.node_evals                      -> (node, act, agg, bias, response, links)
  net.values[key]                     -> that node's current activation
"""

import pygame

from .observation import LOOKAHEAD

POS = (90, 220, 130)   # positive weight
NEG = (225, 90, 90)    # negative weight


def _positions(net, rect):
    """Assign a screen position to every node key."""
    pos = {}
    cell = 18
    gx = rect.x + 45
    gy = rect.y + 48

    # Inputs laid out as the "track ahead": row 0 = ground height per column,
    # row 1 = spike per column, then velocity + on-ground below. Input key
    # -(idx+1) maps to observation index idx (see observation.build_observation).
    for key in net.input_nodes:
        idx = -key - 1
        if idx < LOOKAHEAD * 2:
            col = idx // 2
            row = idx % 2                       # 0 = height, 1 = spike
            pos[key] = (gx + col * cell, gy + row * cell)
        elif idx == LOOKAHEAD * 2:
            pos[key] = (gx, gy + int(3.2 * cell))          # vy
        else:
            pos[key] = (gx + cell, gy + int(3.2 * cell))   # on_ground

    # Output(s) on the right.
    ox = rect.right - 90
    oy = rect.y + rect.height // 2
    for i, key in enumerate(net.output_nodes):
        pos[key] = (ox, oy + i * 44)

    # Hidden nodes (if evolution added any) stacked in a middle column.
    hidden = [n for (n, *_rest) in net.node_evals if n not in net.output_nodes]
    hx = rect.x + int(rect.width * 0.58)
    step = max(20, (rect.height - 96) // max(1, len(hidden)))
    for i, key in enumerate(hidden):
        pos[key] = (hx, rect.y + 56 + i * step)

    return pos


def draw_network(surface, net, rect):
    pos = _positions(net, rect)
    values = getattr(net, "values", {})

    # Connections first, so nodes draw on top.
    for entry in net.node_evals:
        node, links = entry[0], entry[5]
        if node not in pos:
            continue
        for i, w in links:
            if i not in pos:
                continue
            color = POS if w >= 0 else NEG
            width = max(1, min(4, int(abs(w) + 0.5)))
            pygame.draw.line(surface, color, pos[i], pos[node], width)

    label_font = pygame.font.SysFont("consolas", 12)

    # Nodes.
    for key, p in pos.items():
        v = float(values.get(key, 0.0))
        if key in net.output_nodes:
            active = v > 0.5
            pygame.draw.circle(surface, (90, 220, 130) if active else (70, 70, 92), p, 13)
            pygame.draw.circle(surface, (240, 240, 250), p, 13, 2)
            surface.blit(label_font.render("JUMP", True, (235, 235, 245)),
                         (p[0] + 18, p[1] - 7))
        else:
            inten = min(1.0, abs(v))
            base = int(45 + inten * 190)
            pygame.draw.circle(surface, (base, base, min(255, base + 28)), p, 6)

    title = pygame.font.SysFont("consolas", 16).render(
        "NEURAL NETWORK   green = +weight   red = -weight   node brightness = activation",
        True, (200, 200, 215))
    surface.blit(title, (rect.x + 10, rect.y + 8))
    surface.blit(label_font.render(
        "inputs = 10 cells ahead x (ground height, spike)  +  vertical velocity  +  on-ground",
        True, (150, 150, 165)), (rect.x + 10, rect.bottom - 22))
