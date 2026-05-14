#!/usr/bin/env python3
# =============================================================================
# ui/hybrid_dashboard.py — Pure Pygame CityMind Dashboard
# =============================================================================
# Drop-in replacement for Tkinter+Pygame hybrid. Zero Tkinter dependency.
# Theme: Cyberpunk City Intelligence — deep navy, neon cyan, animated grid.
# =============================================================================

import pygame
import pygame.gfxdraw
import sys
import math
import time
from collections import deque

# ── Palette ────────────────────────────────────────────────────────────────────
BG_VOID        = (5,  8,  18)
BG_PANEL       = (10, 15, 30)
BG_CARD        = (16, 24, 46)
BG_CARD_ALT    = (12, 19, 38)

CYAN           = (0,   220, 255)
BLUE           = (30,  110, 255)
GREEN          = (0,   230, 120)
YELLOW         = (255, 220,  30)
ORANGE         = (255, 150,  30)
RED            = (255,  60,  60)
PURPLE         = (170,  60, 255)
PINK           = (255,  80, 160)

TEXT_HI        = (230, 240, 255)
TEXT_MID       = (140, 160, 200)
TEXT_LO        = (70,  90, 130)
BORDER         = (35,  55,  95)
GRID_LINE      = (18,  26,  50)
SEPARATOR      = (28,  40,  72)

# Node type colour map
NODE_COLORS = {
    "Residential":    (35,  85, 185),
    "Hospital":       (200, 45,  70),
    "School":         (185, 160,  0),
    "Industrial":     (190,  95,  0),
    "Power Plant":    (130,  40, 210),
    "Ambulance Depot":(0,  185, 210),
    None:             (22,  32,  58),
}
NODE_ABBREV = {
    "Residential":    "R",
    "Hospital":       "H",
    "School":         "S",
    "Industrial":     "I",
    "Power Plant":    "P",
    "Ambulance Depot":"A",
}

RISK_OVERLAY = {
    "High":   (255,  50,  50, 150),
    "Medium": (255, 180,   0, 110),
    "Low":    (  0, 200, 100,  70),
}

CHALLENGE_COLORS = [CYAN, BLUE, GREEN, YELLOW, ORANGE]
CHALLENGE_NAMES  = [
    "C1   CITY LAYOUT",
    "C2   ROAD NETWORK",
    "C3   AMBULANCES",
    "C4   EMERGENCY ROUTE",
    "C5   CRIME ML",
]

# ── Thin wrapper classes (mimic Tkinter widgets used in run_hybrid.py) ─────────

class ButtonWrapper:
    """Lets run_hybrid.py call .config(command=fn) on a pygame button."""
    def __init__(self):
        self.command = None
    def config(self, command=None, **_):
        if command is not None:
            self.command = command


class LabelWrapper:
    """Lets run_hybrid.py call .config(text=...) on a stat cell."""
    def __init__(self, text="0"):
        self.text = str(text)
    def config(self, text=None, **_):
        if text is not None:
            self.text = str(text)


class BoolVar:
    """Mimics tkinter.BooleanVar."""
    def __init__(self, value=False):
        self._v = bool(value)
    def get(self):
        return self._v
    def set(self, value):
        self._v = bool(value)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _surf(w, h, color, alpha=None):
    """Create a surface, optionally with per-surface alpha."""
    if alpha is not None or (isinstance(color, tuple) and len(color) == 4):
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        if isinstance(color, tuple) and len(color) == 4:
            s.fill(color)
        else:
            s.fill((*color, alpha))
        return s
    s = pygame.Surface((w, h))
    s.fill(color)
    return s


def _card(screen, rect, color=BG_CARD, border=BORDER, radius=6, alpha=200):
    s = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    pygame.draw.rect(s, (*color, alpha), s.get_rect(), border_radius=radius)
    pygame.draw.rect(s, (*border, 255), s.get_rect(), 1, border_radius=radius)
    screen.blit(s, rect.topleft)


def _text_center(screen, font, text, rect, color):
    surf = font.render(text, True, color)
    screen.blit(surf, (rect.centerx - surf.get_width() // 2,
                       rect.centery - surf.get_height() // 2))


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


# ── Main Dashboard Class ───────────────────────────────────────────────────────

class HybridDashboard:
    """
    Pure-Pygame CityMind UI — no Tkinter.

    Public API (identical to the old HybridDashboard):
        log(msg)            — append a line to the event log
        set_graph(graph)    — hot-swap the underlying graph
        run()               — enter the event loop (blocking)

    Attributes wired by run_hybrid.py:
        challenge_buttons   — dict[1..5 → ButtonWrapper]
        stat_labels         — dict['roads'/'ambulances'/'risk_high' → LabelWrapper]
        overlays            — dict[key → bool]
        overlay_vars        — dict[key → BoolVar]
    """

    # ── Window / layout ──────────────────────────────────────────────────────
    W, H        = 1440, 900
    SIDEBAR_W   = 370
    PAD         = 12
    TITLE_H     = 38

    def __init__(self, graph):
        pygame.init()
        pygame.display.set_caption("CityMind  ·  Urban Intelligence System")
        flags = pygame.RESIZABLE
        self.screen = pygame.display.set_mode((self.W, self.H), flags)
        self.clock  = pygame.time.Clock()

        self.graph   = graph
        self.running = True

        # Animation
        self.t       = 0.0        # global time accumulator
        self._scanline_y = 0      # scanline effect

        # Grid camera state
        self._cell   = None       # computed each frame
        self._gox    = 0          # grid origin x
        self._goy    = 0          # grid origin y

        # Interaction
        self.hovered = None
        self.selected = None

        # Log
        self.log_lines  = deque(maxlen=500)
        self.log_scroll = 0       # lines scrolled up from bottom

        # Overlays
        self.overlays     = {"mst": False, "heatmap": False, "ambulance": False}
        self.overlay_vars = {k: BoolVar(v) for k, v in self.overlays.items()}

        # Public widget handles
        self.challenge_buttons = {i: ButtonWrapper() for i in range(1, 6)}
        self.stat_labels = {
            "roads":      LabelWrapper("–"),
            "ambulances": LabelWrapper("–"),
            "risk_high":  LabelWrapper("–"),
        }

        # Internal render state
        self._btn_rects     = {}   # challenge button rects
        self._ov_rects      = {}   # overlay toggle rects
        self._log_rect      = None

        self._init_fonts()

    # ── Font init ─────────────────────────────────────────────────────────────

    def _init_fonts(self):
        def _f(names, size, bold=False):
            for n in names.split(","):
                try:
                    f = pygame.font.SysFont(n.strip(), size, bold=bold)
                    if f:
                        return f
                except Exception:
                    pass
            return pygame.font.SysFont("monospace", size, bold=bold)

        self.fnt_title  = _f("Orbitron,Rajdhani,Eurostile,Consolas", 20, bold=True)
        self.fnt_sub    = _f("Rajdhani,Calibri,Segoe UI", 13)
        self.fnt_hdr    = _f("Rajdhani,Calibri,Segoe UI", 14, bold=True)
        self.fnt_btn    = _f("Rajdhani,Segoe UI,Arial", 13, bold=True)
        self.fnt_body   = _f("Segoe UI,Calibri,Arial", 12)
        self.fnt_mono   = _f("JetBrains Mono,Consolas,Courier New,monospace", 11)
        self.fnt_stat   = _f("Orbitron,Rajdhani,Consolas", 18, bold=True)
        self.fnt_node   = _f("Rajdhani,Segoe UI,Arial", 10, bold=True)

    # ── Public API ────────────────────────────────────────────────────────────

    def log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        self.log_lines.append((ts, str(msg)))
        # Auto-scroll to bottom
        self.log_scroll = 0

    def set_graph(self, graph):
        self.graph = graph

    def run(self):
        while self.running:
            dt = self.clock.tick(60) / 1000.0
            self.t += dt
            self._handle_events()
            self._render()
            pygame.display.flip()
        pygame.quit()
        sys.exit(0)

    # ── Event loop ────────────────────────────────────────────────────────────

    def _handle_events(self):
        mx, my = pygame.mouse.get_pos()
        self.hovered = self._pick_node(mx, my)

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                self.running = False

            elif ev.type == pygame.VIDEORESIZE:
                self.W, self.H = max(800, ev.w), max(600, ev.h)
                self.screen = pygame.display.set_mode(
                    (self.W, self.H), pygame.RESIZABLE)

            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 1:
                    self._handle_click(ev.pos)
                elif ev.button == 4:
                    self.log_scroll = min(
                        max(0, len(self.log_lines) - 1),
                        self.log_scroll + 2)
                elif ev.button == 5:
                    self.log_scroll = max(0, self.log_scroll - 2)

            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    self.running = False
                elif ev.key == pygame.K_HOME:
                    self.log_scroll = max(0, len(self.log_lines) - 1)
                elif ev.key == pygame.K_END:
                    self.log_scroll = 0

    def _handle_click(self, pos):
        x, y = pos
        # Challenge buttons
        for i, rect in self._btn_rects.items():
            if rect.collidepoint(x, y):
                w = self.challenge_buttons.get(i)
                if w and w.command:
                    import threading
                    threading.Thread(target=w.command, daemon=True).start()
                return
        # Overlay toggles
        for key, rect in self._ov_rects.items():
            if rect.collidepoint(x, y):
                nv = not self.overlay_vars[key].get()
                self.overlay_vars[key].set(nv)
                self.overlays[key] = nv
                return
        # Node selection
        if self.hovered:
            self.selected = self.hovered

    # ── Master render ─────────────────────────────────────────────────────────

    def _render(self):
        self.screen.fill(BG_VOID)

        sidebar_x  = self.W - self.SIDEBAR_W
        grid_area  = pygame.Rect(0, self.TITLE_H, sidebar_x - 1, self.H - self.TITLE_H)
        side_area  = pygame.Rect(sidebar_x, 0, self.SIDEBAR_W, self.H)

        self._draw_scanlines()
        self._draw_grid_area(grid_area)
        self._draw_sidebar(side_area)
        self._draw_title_bar(sidebar_x)

        # Vertical separator
        pygame.draw.line(self.screen, BORDER,
                         (sidebar_x - 1, 0), (sidebar_x - 1, self.H), 1)

    # ── Scanlines (subtle) ────────────────────────────────────────────────────

    def _draw_scanlines(self):
        # Very subtle horizontal scanline every 4px
        sl = pygame.Surface((self.W, self.H), pygame.SRCALPHA)
        for y in range(0, self.H, 4):
            pygame.draw.line(sl, (0, 0, 0, 18), (0, y), (self.W, y))
        self.screen.blit(sl, (0, 0))

    # ── Title bar ─────────────────────────────────────────────────────────────

    def _draw_title_bar(self, sidebar_x):
        bar = pygame.Rect(0, 0, self.W, self.TITLE_H)

        # Background
        bg = pygame.Surface((self.W, self.TITLE_H), pygame.SRCALPHA)
        bg.fill((*BG_PANEL, 230))
        self.screen.blit(bg, (0, 0))
        pygame.draw.line(self.screen, CYAN, (0, self.TITLE_H - 1),
                         (self.W, self.TITLE_H - 1), 1)

        # Animated cyan corner accent
        pulse = 0.5 + 0.5 * math.sin(self.t * 3)
        accent_w = int(120 + 30 * pulse)
        accent = pygame.Surface((accent_w, self.TITLE_H), pygame.SRCALPHA)
        for i in range(accent_w):
            alpha = int(60 * (1 - i / accent_w))
            pygame.draw.line(accent, (*CYAN, alpha), (i, 0), (i, self.TITLE_H))
        self.screen.blit(accent, (0, 0))

        # Title text
        title = self.fnt_title.render("CITYMIND", True, CYAN)
        self.screen.blit(title, (14, self.TITLE_H // 2 - title.get_height() // 2))

        sub = self.fnt_sub.render("Urban Intelligence System", True, TEXT_MID)
        self.screen.blit(sub, (14 + title.get_width() + 14,
                               self.TITLE_H // 2 - sub.get_height() // 2 + 1))

        # Grid info
        if self.graph:
            rows = getattr(self.graph, "rows", "?")
            cols = getattr(self.graph, "cols", "?")
            gi = self.fnt_sub.render(f"GRID  {rows}×{cols}", True, TEXT_LO)
            self.screen.blit(gi, (sidebar_x - gi.get_width() - 60,
                                  self.TITLE_H // 2 - gi.get_height() // 2))

        # Clock
        clk = self.fnt_mono.render(time.strftime("%H:%M:%S"), True, TEXT_MID)
        self.screen.blit(clk, (self.W - clk.get_width() - 14,
                               self.TITLE_H // 2 - clk.get_height() // 2))

        # FPS
        fps = self.fnt_mono.render(f"{self.clock.get_fps():.0f} fps", True, TEXT_LO)
        self.screen.blit(fps, (self.W - fps.get_width() - 14,
                               self.TITLE_H - fps.get_height() - 2))

    # ── Grid area ─────────────────────────────────────────────────────────────

    def _draw_grid_area(self, area: pygame.Rect):
        if not self.graph:
            msg = self.fnt_hdr.render("Waiting for graph…", True, TEXT_LO)
            self.screen.blit(msg, (area.centerx - msg.get_width() // 2,
                                   area.centery - msg.get_height() // 2))
            return

        rows = getattr(self.graph, "rows", 10)
        cols = getattr(self.graph, "cols", 10)

        pad = 20
        aw = area.width  - 2 * pad
        ah = area.height - 2 * pad
        cell = max(8, min(aw // cols, ah // rows))
        self._cell = cell

        gw = cell * cols
        gh = cell * rows
        ox = area.x + pad + (aw - gw) // 2
        oy = area.y + pad + (ah - gh) // 2
        self._gox, self._goy = ox, oy

        # ── Grid background
        bg_rect = pygame.Rect(ox - 2, oy - 2, gw + 4, gh + 4)
        _card(self.screen, bg_rect, BG_PANEL, BORDER, radius=4, alpha=180)

        # ── Grid lines
        for r in range(rows + 1):
            yy = oy + r * cell
            pygame.draw.line(self.screen, GRID_LINE, (ox, yy), (ox + gw, yy))
        for c in range(cols + 1):
            xx = ox + c * cell
            pygame.draw.line(self.screen, GRID_LINE, (xx, oy), (xx, oy + gh))

        all_nodes = (list(self.graph.all_nodes())
                     if hasattr(self.graph, "all_nodes") else [])

        # ── Road / MST edges
        if self.overlays.get("mst"):
            self._draw_road_edges(ox, oy, cell)

        # ── Nodes
        for node in all_nodes:
            self._draw_node(node, ox, oy, cell)

        # ── Heatmap overlay
        if self.overlays.get("heatmap"):
            hs = pygame.Surface((gw, gh), pygame.SRCALPHA)
            for node in all_nodes:
                risk  = getattr(node, "predicted_risk", "Low")
                color = RISK_OVERLAY.get(risk)
                if color:
                    r2, c2 = node.row, node.col
                    pygame.draw.rect(hs, color,
                                     (c2 * cell, r2 * cell, cell, cell))
            self.screen.blit(hs, (ox, oy))

        # ── Ambulance coverage rings
        if self.overlays.get("ambulance"):
            for node in all_nodes:
                if getattr(node, "has_ambulance", False):
                    self._draw_amb_ring(node, ox, oy, cell)

        # ── Hover tooltip
        if self.hovered:
            self._draw_tooltip(self.hovered, ox, oy, cell, area)

        # ── Selected node detail
        if self.selected:
            self._draw_selected_highlight(self.selected, ox, oy, cell)

    def _draw_road_edges(self, ox, oy, cell):
        """Draw MST / road edges if graph exposes them."""
        edges = None
        if hasattr(self.graph, "edges"):
            edges = self.graph.edges
        elif hasattr(self.graph, "mst_edges"):
            edges = self.graph.mst_edges

        if not edges:
            return

        edge_surf = pygame.Surface(
            (self._cell * getattr(self.graph, "cols", 10) + 4,
             self._cell * getattr(self.graph, "rows", 10) + 4),
            pygame.SRCALPHA)

        for edge in edges:
            try:
                if hasattr(edge, "__len__") and len(edge) == 4:
                    r1, c1, r2, c2 = edge
                elif hasattr(edge, "src") and hasattr(edge, "dst"):
                    r1, c1 = edge.src.row, edge.src.col
                    r2, c2 = edge.dst.row, edge.dst.col
                else:
                    continue
            except Exception:
                continue

            x1 = ox + c1 * cell + cell // 2
            y1 = oy + r1 * cell + cell // 2
            x2 = ox + c2 * cell + cell // 2
            y2 = oy + r2 * cell + cell // 2

            # Glow pass
            pygame.draw.line(self.screen, (*GREEN, 35), (x1, y1), (x2, y2), 4)
            # Solid pass
            pygame.draw.aaline(self.screen, GREEN, (x1, y1), (x2, y2))

    def _draw_node(self, node, ox, oy, cell):
        r, c   = node.row, node.col
        ltype  = getattr(node, "location_type", None)
        color  = NODE_COLORS.get(ltype, NODE_COLORS[None])
        access = getattr(node, "is_accessible", True)

        if not access:
            color = tuple(max(0, v - 50) for v in color)

        mg    = max(1, cell // 10)
        inner = cell - 2 * mg
        nx    = ox + c * cell + mg
        ny    = oy + r * cell + mg

        is_hov = (self.hovered  is node)
        is_sel = (self.selected is node)

        # Selection glow
        if is_sel:
            gs = pygame.Surface((inner + 10, inner + 10), pygame.SRCALPHA)
            gs.fill((*CYAN, 50))
            self.screen.blit(gs, (nx - 5, ny - 5))
        elif is_hov:
            gs = pygame.Surface((inner + 6, inner + 6), pygame.SRCALPHA)
            gs.fill((*CYAN, 25))
            self.screen.blit(gs, (nx - 3, ny - 3))

        # Cell fill
        nr = max(1, cell // 8)
        pygame.draw.rect(self.screen, color,
                         pygame.Rect(nx, ny, inner, inner),
                         border_radius=nr)

        # Animated border for special nodes
        pulse = 0.5 + 0.5 * math.sin(self.t * 2.5 + r * 0.4 + c * 0.3)
        if ltype == "Hospital":
            bc = (*RED, int(80 + 120 * pulse))
            s  = pygame.Surface((inner, inner), pygame.SRCALPHA)
            pygame.draw.rect(s, bc, s.get_rect(), max(1, cell // 9), border_radius=nr)
            self.screen.blit(s, (nx, ny))
        elif ltype == "Ambulance Depot":
            bc = (*CYAN, int(80 + 120 * pulse))
            s  = pygame.Surface((inner, inner), pygame.SRCALPHA)
            pygame.draw.rect(s, bc, s.get_rect(), max(1, cell // 9), border_radius=nr)
            self.screen.blit(s, (nx, ny))

        # Abbreviation label
        if cell >= 22 and ltype:
            abbr = NODE_ABBREV.get(ltype, "?")
            lbl  = self.fnt_node.render(abbr, True, TEXT_HI)
            self.screen.blit(lbl, (nx + inner // 2 - lbl.get_width() // 2,
                                   ny + inner // 2 - lbl.get_height() // 2))

        # Blocked cross
        if not access and cell >= 14:
            m2 = mg + 2
            pygame.draw.line(self.screen, RED,
                             (ox + c*cell + m2,   oy + r*cell + m2),
                             (ox + c*cell + cell - m2, oy + r*cell + cell - m2), 2)
            pygame.draw.line(self.screen, RED,
                             (ox + c*cell + cell - m2, oy + r*cell + m2),
                             (ox + c*cell + m2,   oy + r*cell + cell - m2), 2)

    def _draw_amb_ring(self, node, ox, oy, cell):
        cx = ox + node.col * cell + cell // 2
        cy = oy + node.row * cell + cell // 2
        rad = cell * 3
        s   = pygame.Surface((rad * 2 + 4, rad * 2 + 4), pygame.SRCALPHA)
        pygame.draw.circle(s, (*CYAN, 18), (rad + 2, rad + 2), rad)
        pygame.draw.circle(s, (*CYAN, 90), (rad + 2, rad + 2), rad, 2)
        self.screen.blit(s, (cx - rad - 2, cy - rad - 2))

    def _draw_tooltip(self, node, ox, oy, cell, area):
        r, c   = node.row, node.col
        ltype  = getattr(node, "location_type", "?") or "?"
        dens   = getattr(node, "population_density", 0)
        risk   = getattr(node, "predicted_risk", "–")
        access = getattr(node, "is_accessible", True)

        lines = [
            (f"({r}, {c})",                CYAN),
            (ltype,                         TEXT_HI),
            (f"Density  {dens:.0f}",       TEXT_MID),
            (f"Risk      {risk}",           RED if risk == "High" else
                                            YELLOW if risk == "Medium" else GREEN),
            (f"Access  {'✓' if access else '✗'}",
                                            GREEN if access else RED),
        ]

        lh  = 15
        tw  = 150
        th  = len(lines) * lh + 12
        tx  = ox + c * cell + cell + 4
        ty  = oy + r * cell

        if tx + tw > area.right:
            tx = ox + c * cell - tw - 4
        ty = _clamp(ty, area.top, area.bottom - th - 4)

        bg = pygame.Surface((tw, th), pygame.SRCALPHA)
        bg.fill((*BG_PANEL, 230))
        pygame.draw.rect(bg, CYAN, bg.get_rect(), 1, border_radius=4)
        self.screen.blit(bg, (tx, ty))

        for i, (txt, col) in enumerate(lines):
            s = self.fnt_body.render(txt, True, col)
            self.screen.blit(s, (tx + 7, ty + 6 + i * lh))

    def _draw_selected_highlight(self, node, ox, oy, cell):
        mg = max(1, cell // 10)
        nx = ox + node.col * cell + mg
        ny = oy + node.row * cell + mg
        inner = cell - 2 * mg
        nr = max(1, cell // 8)
        pulse = abs(math.sin(self.t * 4))
        s = pygame.Surface((inner, inner), pygame.SRCALPHA)
        pygame.draw.rect(s, (*CYAN, int(50 + 80 * pulse)),
                         s.get_rect(), 2, border_radius=nr)
        self.screen.blit(s, (nx, ny))

    # ── Sidebar ───────────────────────────────────────────────────────────────

    def _draw_sidebar(self, area: pygame.Rect):
        # Sidebar background
        bg = pygame.Surface((area.width, area.height), pygame.SRCALPHA)
        bg.fill((*BG_PANEL, 240))
        self.screen.blit(bg, area.topleft)

        p = self.PAD
        y = self.TITLE_H + p
        bw = area.width - p * 2

        y = self._draw_challenge_buttons(area, p, y, bw)
        y = self._draw_separator(area, p, y, bw)
        y = self._draw_stats(area, p, y, bw)
        y = self._draw_separator(area, p, y, bw)
        y = self._draw_overlays(area, p, y, bw)
        y = self._draw_separator(area, p, y, bw)
        y = self._draw_legend(area, p, y, bw)
        y = self._draw_separator(area, p, y, bw)
        self._draw_log(area, p, y, bw)

    def _draw_separator(self, area, p, y, bw):
        pygame.draw.line(self.screen, SEPARATOR,
                         (area.x + p, y + 3),
                         (area.x + p + bw, y + 3), 1)
        return y + 10

    def _draw_section_header(self, area, p, y, text):
        """Draw a small section header label."""
        s = self.fnt_hdr.render(text, True, CYAN)
        self.screen.blit(s, (area.x + p, y))
        return y + s.get_height() + 5

    # Challenge buttons ────────────────────────────────────────────────────────

    def _draw_challenge_buttons(self, area, p, y, bw):
        y = self._draw_section_header(area, p, y, "CHALLENGES")

        btn_h = 40
        mx, my = pygame.mouse.get_pos()

        for i in range(1, 6):
            rect = pygame.Rect(area.x + p, y, bw, btn_h)
            self._btn_rects[i] = rect

            col     = CHALLENGE_COLORS[i - 1]
            hov     = rect.collidepoint(mx, my)
            bg_a    = 200 if hov else 140

            # Card bg
            bgs = pygame.Surface((bw, btn_h), pygame.SRCALPHA)
            bgs.fill((*BG_CARD, bg_a))
            pygame.draw.rect(bgs, (*col, 255) if hov else (*BORDER, 255),
                             bgs.get_rect(), 1, border_radius=5)
            self.screen.blit(bgs, rect.topleft)

            # Left accent strip (animated on hover)
            strip_h = btn_h if not hov else int(btn_h * (0.6 + 0.4 * abs(math.sin(self.t * 6))))
            strip_y = rect.y + (btn_h - strip_h) // 2
            pygame.draw.rect(self.screen, col,
                             pygame.Rect(area.x + p, strip_y, 4, strip_h),
                             border_radius=2)

            # Label
            lbl = self.fnt_btn.render(CHALLENGE_NAMES[i - 1], True,
                                      col if hov else TEXT_HI)
            self.screen.blit(lbl, (rect.x + 12,
                                   rect.centery - lbl.get_height() // 2))

            # Arrow
            arr = self.fnt_btn.render("▶", True, col)
            self.screen.blit(arr, (rect.right - arr.get_width() - 10,
                                   rect.centery - arr.get_height() // 2))

            y += btn_h + 4

        return y + 2

    # Stats ────────────────────────────────────────────────────────────────────

    def _draw_stats(self, area, p, y, bw):
        y = self._draw_section_header(area, p, y, "STATISTICS")

        stats = [
            ("ROADS",      self.stat_labels["roads"].text,      GREEN),
            ("AMBULANCES", self.stat_labels["ambulances"].text,  CYAN),
            ("HIGH RISK",  self.stat_labels["risk_high"].text,   RED),
        ]

        sw  = (bw - 2 * 4) // 3
        sh  = 56
        sx  = area.x + p

        for key, val, col in stats:
            r = pygame.Rect(sx, y, sw, sh)
            _card(self.screen, r, BG_CARD, col, radius=5, alpha=180)

            vs = self.fnt_stat.render(str(val), True, col)
            self.screen.blit(vs, (r.centerx - vs.get_width() // 2, r.y + 8))

            ks = self.fnt_body.render(key, True, TEXT_LO)
            self.screen.blit(ks, (r.centerx - ks.get_width() // 2, r.y + sh - ks.get_height() - 6))

            sx += sw + 4

        return y + sh + 6

    # Overlays ─────────────────────────────────────────────────────────────────

    def _draw_overlays(self, area, p, y, bw):
        y = self._draw_section_header(area, p, y, "OVERLAYS")

        ovs = [
            ("mst",       "Road Network",  GREEN),
            ("heatmap",   "Risk Heatmap",  RED),
            ("ambulance", "AMB Coverage",  CYAN),
        ]

        oh  = 28
        mx, my = pygame.mouse.get_pos()

        for key, label, col in ovs:
            rect   = pygame.Rect(area.x + p, y, bw, oh)
            self._ov_rects[key] = rect
            active = self.overlays.get(key, False)
            hov    = rect.collidepoint(mx, my)

            bg_col = (*col, 40) if active else (*BG_CARD, 140)
            bgs = pygame.Surface((bw, oh), pygame.SRCALPHA)
            bgs.fill(bg_col)
            bd  = col if (active or hov) else BORDER
            pygame.draw.rect(bgs, (*bd, 220), bgs.get_rect(), 1, border_radius=4)
            self.screen.blit(bgs, rect.topleft)

            # Checkbox
            cb = pygame.Rect(rect.x + 7, rect.centery - 6, 13, 13)
            if active:
                pygame.draw.rect(self.screen, col, cb, border_radius=2)
                ck = self.fnt_body.render("✓", True, BG_VOID)
                self.screen.blit(ck, (cb.x + 1, cb.y - 1))
            else:
                pygame.draw.rect(self.screen, TEXT_LO, cb, 1, border_radius=2)

            lbl = self.fnt_btn.render(label, True, col if active else TEXT_MID)
            self.screen.blit(lbl, (rect.x + 28, rect.centery - lbl.get_height() // 2))

            y += oh + 4

        return y + 2

    # Legend ───────────────────────────────────────────────────────────────────

    def _draw_legend(self, area, p, y, bw):
        y = self._draw_section_header(area, p, y, "NODE TYPES")

        items = list(NODE_COLORS.items())
        items = [(k, v) for k, v in items if k is not None]

        cw = bw // 2 - 2
        lh = 18
        for idx, (name, col) in enumerate(items):
            col_x = area.x + p + (idx % 2) * (cw + 4)
            iy    = y + (idx // 2) * lh

            pygame.draw.rect(self.screen, col,
                             pygame.Rect(col_x, iy + 3, 12, 12), border_radius=2)
            ls = self.fnt_body.render(name[:16], True, TEXT_MID)
            self.screen.blit(ls, (col_x + 16, iy + 1))

        total_rows = (len(items) + 1) // 2
        return y + total_rows * lh + 4

    # Event log ────────────────────────────────────────────────────────────────

    def _draw_log(self, area, p, y, bw):
        y = self._draw_section_header(area, p, y, "EVENT LOG")

        log_h = self.H - y - p
        if log_h < 40:
            return

        rect = pygame.Rect(area.x + p, y, bw, log_h)
        self._log_rect = rect
        _card(self.screen, rect, BG_CARD, BORDER, radius=4, alpha=160)

        lh   = 14
        pad  = 4
        vis  = max(1, (log_h - pad * 2) // lh)

        lines = list(self.log_lines)
        total = len(lines)
        end   = max(0, total - self.log_scroll)
        start = max(0, end - vis)

        clip = self.screen.get_clip()
        self.screen.set_clip(rect.inflate(-2, -2))

        for i, (ts, msg) in enumerate(lines[start:end]):
            ly = rect.y + pad + i * lh

            ts_s = self.fnt_mono.render(ts, True, TEXT_LO)
            self.screen.blit(ts_s, (rect.x + pad, ly))

            if   msg.startswith("✓"):   mc = GREEN
            elif msg.startswith("✗"):   mc = RED
            elif msg.startswith("▶"):   mc = YELLOW
            elif msg.startswith("   "): mc = TEXT_LO
            else:                       mc = TEXT_MID

            tx   = rect.x + pad + ts_s.get_width() + 4
            avail = rect.right - tx - pad
            # Truncate to fit
            trimmed = msg
            ms = self.fnt_mono.render(trimmed, True, mc)
            while ms.get_width() > avail and len(trimmed) > 0:
                trimmed = trimmed[:-1]
                ms = self.fnt_mono.render(trimmed + "…", True, mc)
            if len(trimmed) < len(msg):
                ms = self.fnt_mono.render(trimmed + "…", True, mc)
            self.screen.blit(ms, (tx, ly))

        self.screen.set_clip(clip)

        # Scroll indicator
        if total > vis:
            sb_h = max(20, int(log_h * vis / total))
            sb_y = rect.y + int((log_h - sb_h) *
                                 _clamp(1 - self.log_scroll / max(1, total - vis), 0, 1))
            pygame.draw.rect(self.screen, BORDER,
                             pygame.Rect(rect.right - 5, sb_y, 3, sb_h),
                             border_radius=2)

    # ── Node picking ──────────────────────────────────────────────────────────

    def _pick_node(self, mx, my):
        if not self.graph or self._cell is None:
            return None
        ox, oy, cell = self._gox, self._goy, self._cell
        rows = getattr(self.graph, "rows", 10)
        cols = getattr(self.graph, "cols", 10)
        c = (mx - ox) // cell
        r = (my - oy) // cell
        if 0 <= r < rows and 0 <= c < cols:
            if hasattr(self.graph, "get_node"):
                try:
                    return self.graph.get_node(r, c)
                except Exception:
                    pass
        return None