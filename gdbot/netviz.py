"""Compact, MarI/O-style live visualization of a NEAT network.

Input tiles in a tight grid on the left (bright = active), hidden nodes in the
middle, the JUMP output on the right, connections colored by weight sign
(green = +, red = -). Works for any input count (sim's 22, live's 12, ...).

A neat-python FeedForwardNetwork exposes what we need after `activate()`:
  net.input_nodes / net.output_nodes, net.node_evals, net.values[key].
"""

import math

import pygame

POS = (90, 220, 130)   # positive weight
NEG = (225, 90, 90)     # negative weight


def _layout(net, rect):
    inputs = list(net.input_nodes)
    outputs = list(net.output_nodes)
    hidden = [n for (n, *_rest) in net.node_evals if n not in outputs]

    pos = {}
    n = len(inputs)
    # Input tile grid: a single row (a "look-ahead radar") when small, else 2 rows.
    rows = 1 if n <= 12 else 2
    cols = math.ceil(n / rows)
    cell = 19
    gx = rect.x + 22
    gy = rect.centery - ((rows - 1) * cell) // 2   # vertically centered, aligns with JUMP
    for k, key in enumerate(inputs):
        pos[key] = (gx + (k % cols) * cell, gy + (k // cols) * cell)

    # Hidden column (only if evolution added any), then JUMP — both pulled left
    # so connections stay short and the whole graph reads compactly.
    hx = rect.x + int(rect.width * 0.46)
    if hidden:
        step = max(15, (rect.height - 90) // max(1, len(hidden)))
        for i, key in enumerate(hidden):
            pos[key] = (hx, rect.y + 52 + i * step)

    ox = rect.x + int(rect.width * 0.66)
    oy = rect.y + rect.height // 2
    for i, key in enumerate(outputs):
        pos[key] = (ox, oy + i * 40)

    return pos, inputs, outputs, hidden


def draw_network(surface, net, rect):
    pos, inputs, outputs, hidden = _layout(net, rect)
    values = getattr(net, "values", {})
    small = pygame.font.SysFont("consolas", 11)

    # connections first
    for entry in net.node_evals:
        node, links = entry[0], entry[5]
        if node not in pos:
            continue
        for i, w in links:
            if i in pos:
                pygame.draw.line(surface, POS if w >= 0 else NEG,
                                 pos[i], pos[node], max(1, min(3, int(abs(w) + 0.4))))

    # input tiles (bright = active)
    for key in inputs:
        inten = min(1.0, abs(float(values.get(key, 0.0))))
        b = int(28 + inten * 212)
        x, y = pos[key]
        pygame.draw.rect(surface, (b, b, b), (x - 7, y - 7, 14, 14))
        pygame.draw.rect(surface, (70, 80, 112), (x - 7, y - 7, 14, 14), 1)

    # hidden nodes
    for key in hidden:
        inten = min(1.0, abs(float(values.get(key, 0.0))))
        b = int(45 + inten * 190)
        pygame.draw.circle(surface, (b, b, min(255, b + 30)), pos[key], 5)

    # output(s)
    for key in outputs:
        active = float(values.get(key, 0.0)) > 0.5
        x, y = pos[key]
        pygame.draw.rect(surface, (90, 220, 130) if active else (58, 58, 82),
                         (x - 11, y - 11, 22, 22))
        pygame.draw.rect(surface, (240, 240, 250), (x - 11, y - 11, 22, 22), 2)
        surface.blit(small.render("JUMP", True, (235, 235, 245)), (x + 16, y - 6))

    title = pygame.font.SysFont("consolas", 13).render(
        "neural net   green +   red -   bright = active", True, (190, 190, 205))
    surface.blit(title, (rect.x + 10, rect.y + 8))
    if inputs:
        ix, iy = pos[inputs[0]]
        surface.blit(small.render("look-ahead cells + motion  →  JUMP",
                                  True, (140, 140, 158)), (ix - 8, iy - 24))
