"""Generate three matplotlib figures for aircraft design paper.

All data hardcoded — no file I/O needed.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Ellipse, FancyBboxPatch, Polygon

sns.set_theme(style="whitegrid", context="paper", font="DejaVu Serif")

# ── Data ──────────────────────────────────────────────────────────────────

DEFAULTS = {
    "AR": 11.22, "AREA": 124.6, "SPAN": 37.35,
    "SWEEP": 25.0, "TAPER": 0.278,
    "FUS_LENGTH": 37.79, "FUS_HEIGHT": 4.06, "FUS_WIDTH": 3.76,
    "SF": 1.0
}

COMBINATIONS = {
    "seq_if":    {"AR": 8.719,  "AREA": 160.0,                          "SF": 1.0},
    "seq_sp":    {"AR": 12.5,   "AREA": 160.0,  "SPAN": 44.72,          "SF": 0.8},
    "orch_if":   {"AR": 10.0,                   "FUS_LENGTH": 35.0,     "SF": 1.2},
    "orch_sp":   {"AR": 12.1,                   "SWEEP": 25.0, "TAPER": 0.3},
    "orch_gr":   {                                                        "SF": 0.8},
    "net_if":    {"AR": 14.0,   "AREA": 162.86, "SPAN": 46.0},
    "net_sp":    {"AR": 12.1,   "AREA": 160.0,  "SPAN": 44.0,           "SF": 0.8},
    "net_gr":    {"AR": 11.22,                                           "SF": 0.8},
    "reference": {}
}

LABELS = {
    "seq_if":    "Seq + IterFB",
    "seq_sp":    "Seq + StagedPipe",
    "orch_if":   "Orch + IterFB",
    "orch_sp":   "Orch + StagedPipe",
    "orch_gr":   "Orch + GraphRt",
    "net_if":    "Net + IterFB",
    "net_sp":    "Net + StagedPipe",
    "net_gr":    "Net + GraphRt",
    "reference": "Reference (SLSQP)"
}

HANDLERS = {
    "seq_if": "sequential", "seq_sp": "sequential",
    "orch_if": "orchestrated", "orch_sp": "orchestrated", "orch_gr": "orchestrated",
    "net_if": "networked", "net_sp": "networked", "net_gr": "networked",
    "reference": "baseline"
}

HANDLER_COLORS = {
    "sequential":  "#4e79a7",
    "orchestrated": "#f28e2b",
    "networked":   "#59a14f",
    "baseline":    "#333333",
}

HANDLER_LS = {
    "sequential":  "-",
    "orchestrated": "--",
    "networked":   ":",
    "baseline":    "-",
}

HANDLER_MARKER = {
    "sequential":  "s",
    "orchestrated": "^",
    "networked":   "o",
    "baseline":    "*",
}

FIGSIZE_1 = (10, 7)
FIGSIZE_2 = (8, 6)
FIGSIZE_3 = (12, 9)
DPI = 300


# ── Parameter derivation ─────────────────────────────────────────────────

def derive_params(combo_overrides):
    p = {**DEFAULTS, **combo_overrides}
    has_span = "SPAN" in combo_overrides
    has_area = "AREA" in combo_overrides

    if has_span and not has_area:
        p["AREA"] = p["SPAN"] ** 2 / p["AR"]
    elif has_area and not has_span:
        p["SPAN"] = np.sqrt(p["AR"] * p["AREA"])
    elif has_span and has_area:
        p["AR"] = p["SPAN"] ** 2 / p["AREA"]
    # else: both from defaults, consistent

    semispan = p["SPAN"] / 2
    sweep_rad = np.radians(p["SWEEP"])
    c_root = (2 * p["AREA"]) / (p["SPAN"] * (1 + p["TAPER"]))
    c_tip = p["TAPER"] * c_root

    p["c_root"] = c_root
    p["c_tip"] = c_tip
    p["semispan"] = semispan
    p["sweep_rad"] = sweep_rad
    return p


ALL_PARAMS = {k: derive_params(v) for k, v in COMBINATIONS.items()}


# ── Wing planform geometry ────────────────────────────────────────────────

def wing_polygon(p):
    """Return (right_wing_verts, left_wing_verts) as Nx2 arrays.
    Coordinate system: x = chordwise (aft positive), y = spanwise (right positive).
    """
    c_root = p["c_root"]
    c_tip = p["c_tip"]
    semispan = p["semispan"]
    sweep_rad = p["sweep_rad"]

    # Tip quarter-chord x position
    tip_qc_x = c_root * 0.25 + np.tan(sweep_rad) * semispan
    tip_le_x = tip_qc_x - 0.25 * c_tip
    tip_te_x = tip_qc_x + 0.75 * c_tip

    # Right wing polygon (root to tip)
    right = np.array([
        [0, 0],
        [tip_le_x, semispan],
        [tip_te_x, semispan],
        [c_root, 0],
        [0, 0],
    ])

    # Left wing: mirror y, shift so symmetric about x = c_root/2
    left = right.copy()
    left[:, 1] = -left[:, 1]

    return right, left


# ── FIGURE 1 — 2D Planform Overlay ───────────────────────────────────────

def plot_planform(output_dir):
    fig, ax = plt.subplots(figsize=FIGSIZE_1, dpi=DPI)

    # Draw order: non-reference first, reference last on top
    draw_order = [k for k in COMBINATIONS if k != "reference"] + ["reference"]

    for key in draw_order:
        p = ALL_PARAMS[key]
        handler = HANDLERS[key]
        color = HANDLER_COLORS[handler]
        ls = HANDLER_LS[handler]
        label_text = f"{LABELS[key]}  (span={p['SPAN']:.1f}m, AR={p['AR']:.2f})"

        right, left = wing_polygon(p)

        if key == "reference":
            kwargs_fill = dict(facecolor="#d0d0d0", edgecolor="black",
                               linewidth=1.8, linestyle=ls, alpha=0.5, zorder=10)
            kwargs_fus = dict(facecolor="#e0e0e0", edgecolor="black",
                              linewidth=1.5, zorder=10)
        else:
            kwargs_fill = dict(facecolor=color, edgecolor=color,
                               linewidth=1.0, linestyle=ls,
                               alpha=0.25, zorder=2)
            kwargs_fus = dict(facecolor=color, edgecolor=color,
                              linewidth=0.8, alpha=0.15, zorder=2)

        # Draw wings
        ax.fill(right[:, 1], right[:, 0], label=label_text, **kwargs_fill)
        ax.fill(left[:, 1], left[:, 0], **kwargs_fill)

        # Fuselage stub
        fus_w = p["FUS_WIDTH"]
        fus_l = p["FUS_LENGTH"]
        c_root = p["c_root"]
        fus_rect = plt.Rectangle(
            (-fus_w / 2, -fus_l + c_root * 0.4),
            fus_w, fus_l,
            **kwargs_fus)
        ax.add_patch(fus_rect)

    ax.set_aspect("equal")
    ax.set_xlabel("Spanwise y (m)", fontsize=11)
    ax.set_ylabel("x (m)", fontsize=11)
    ax.set_title("Wing Planform Comparison — All Combinations",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left", framealpha=0.9, ncol=1)

    plt.tight_layout()
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(output_dir, f"fig1_planform.{ext}"),
                    dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved fig1_planform")


# ── FIGURE 2 — Design Space Scatter ──────────────────────────────────────

def plot_design_space(output_dir):
    fig, ax = plt.subplots(figsize=FIGSIZE_2, dpi=DPI)

    # Background fuel surrogate contourf
    area_grid = np.linspace(90, 180, 200)
    ar_grid = np.linspace(6, 16, 200)
    A, R = np.meshgrid(area_grid, ar_grid)
    fuel_proxy = A * np.sqrt(R)
    ax.contourf(A, R, fuel_proxy, levels=15, cmap="YlOrRd", alpha=0.15)

    # Design bounds rectangle
    ax.plot([100, 160, 160, 100, 100], [7, 7, 14, 14, 7],
            color="#888", linestyle="--", linewidth=1.0, alpha=0.6, zorder=3,
            label="Design bounds")

    # Span constraint curves
    for span_val, style, lbl in [(28, "-", "span = 28 m"),
                                  (37.35, "--", "span = 37.35 m (default)"),
                                  (48, "-", "span = 48 m")]:
        ar_curve = span_val ** 2 / area_grid
        clr = "#2ca02c" if span_val == 37.35 else "#7fbfff"
        ax.plot(area_grid, ar_curve, color=clr, linewidth=1.0,
                linestyle=style, alpha=0.6, zorder=3, label=lbl)

    # Scatter points
    for key in COMBINATIONS:
        p = ALL_PARAMS[key]
        handler = HANDLERS[key]
        color = HANDLER_COLORS[handler]
        marker = HANDLER_MARKER[handler]
        s = 200 if key == "reference" else 120
        edgew = 1.5 if key == "reference" else 1.0
        zord = 12 if key == "reference" else 8

        ax.scatter(p["AREA"], p["AR"], s=s, marker=marker, color=color,
                   edgecolors="black", linewidths=edgew, zorder=zord,
                   label=LABELS[key])

        # Annotation offset
        dx, dy = 2.0, 0.25
        if key == "reference":
            dx, dy = -3.0, -0.6
        elif key == "net_if":
            dx, dy = 2.0, -0.5
        elif key == "orch_sp":
            dx, dy = -8.0, 0.4
        elif key == "net_gr":
            dx, dy = 2.0, -0.5
        ax.annotate(LABELS[key], (p["AREA"], p["AR"]),
                    xytext=(p["AREA"] + dx, p["AR"] + dy),
                    fontsize=6.5, ha="left", va="bottom",
                    arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.4),
                    zorder=15)

    # Text box
    textstr = ("Wing area drives structural weight.\n"
               "AR drives induced drag.\n"
               r"Span = $\sqrt{\mathrm{AR} \times \mathrm{Area}}$")
    props = dict(boxstyle="round,pad=0.4", facecolor="white",
                 alpha=0.85, edgecolor="#ccc")
    ax.text(0.97, 0.03, textstr, transform=ax.transAxes, fontsize=7,
            verticalalignment="bottom", horizontalalignment="right",
            bbox=props, zorder=20)

    ax.set_xlim(90, 180)
    ax.set_ylim(6, 16)
    ax.set_xlabel("Wing Area (m$^2$)", fontsize=11)
    ax.set_ylabel("Aspect Ratio", fontsize=11)
    ax.set_title("Design Space — Wing Area vs. Aspect Ratio",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=6.5, loc="upper left", framealpha=0.9, ncol=2)

    plt.tight_layout()
    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(output_dir, f"fig2_design_space.{ext}"),
                    dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved fig2_design_space")


# ── FIGURE 3 — Orthographic Silhouettes ──────────────────────────────────

def draw_top_view(ax, p, color):
    """Top view: planform + fuselage + nacelles."""
    right, left = wing_polygon(p)

    # Wings
    ax.fill(right[:, 1], right[:, 0], facecolor=color, alpha=0.2,
            edgecolor=color, linewidth=1.2)
    ax.fill(left[:, 1], left[:, 0], facecolor=color, alpha=0.2,
            edgecolor=color, linewidth=1.2)

    # Fuselage
    fus_w = p["FUS_WIDTH"]
    fus_l = p["FUS_LENGTH"]
    c_root = p["c_root"]
    fus_rect = FancyBboxPatch(
        (-fus_w / 2, -fus_l + c_root * 0.4), fus_w, fus_l,
        boxstyle="round,pad=0.3",
        facecolor=color, alpha=0.15, edgecolor=color, linewidth=1.0)
    ax.add_patch(fus_rect)

    # Nacelles
    semispan = p["semispan"]
    sweep_rad = p["sweep_rad"]
    nacelle_y = semispan * 0.35
    nacelle_x = p["c_root"] * 0.25 + np.tan(sweep_rad) * nacelle_y
    nacelle_r = 0.95 * p["SF"]
    for sign in [1, -1]:
        circ = plt.Circle((sign * nacelle_y, nacelle_x),
                           nacelle_r, facecolor=color, alpha=0.3,
                           edgecolor=color, linewidth=0.8)
        ax.add_patch(circ)

    ax.set_aspect("equal")


def draw_side_view(ax, p, color):
    """Side view: fuselage, wing stub, engine, vertical tail, horizontal tail."""
    fl = p["FUS_LENGTH"]
    fh = p["FUS_HEIGHT"]
    sf = p["SF"]

    # Fuselage body
    fus = FancyBboxPatch(
        (0, -fh / 2), fl, fh,
        boxstyle="round,pad=0.8",
        facecolor=color, alpha=0.15, edgecolor=color, linewidth=1.2)
    ax.add_patch(fus)

    # Wing stub
    wing_y = -fh * 0.1
    ax.plot([fl * 0.35, fl * 0.60], [wing_y, wing_y],
            color=color, linewidth=2.5, solid_capstyle="round")

    # Engine ellipse
    eng = Ellipse((fl * 0.47, -fh * 0.35),
                  width=fl * 0.14 * sf, height=fh * 0.24 * sf,
                  facecolor=color, alpha=0.25, edgecolor=color, linewidth=0.8)
    ax.add_patch(eng)

    # Vertical tail
    vt_base_x = fl * 0.88
    vt_height = fh * 0.55
    vt = Polygon([
        [vt_base_x, fh * 0.4],
        [fl * 0.98, fh * 0.4 + vt_height],
        [fl * 0.92, fh * 0.4 + vt_height],
        [vt_base_x - fl * 0.03, fh * 0.4],
    ], closed=True, facecolor=color, alpha=0.2, edgecolor=color, linewidth=0.8)
    ax.add_patch(vt)

    # Horizontal tail
    ht_y = fh * 0.5
    ax.plot([fl * 0.87, fl * 0.97], [ht_y, ht_y],
            color=color, linewidth=2.0, solid_capstyle="round")

    ax.set_aspect("equal")


def draw_front_view(ax, p, color):
    """Front view: fuselage ellipse, wing line, engine circles."""
    fh = p["FUS_HEIGHT"]
    fw = p["FUS_WIDTH"]
    span = p["SPAN"]
    sf = p["SF"]

    # Fuselage ellipse
    fus = Ellipse((0, 0), fw, fh,
                  facecolor=color, alpha=0.15, edgecolor=color, linewidth=1.2)
    ax.add_patch(fus)

    # Wing line
    ax.plot([-span / 2, span / 2], [0, 0],
            color=color, linewidth=2.0)

    # Engine circles
    eng_y = span * 0.35
    eng_r = 0.95 * sf
    for sign in [1, -1]:
        circ = plt.Circle((sign * eng_y, -fh * 0.15),
                           eng_r, facecolor=color, alpha=0.25,
                           edgecolor=color, linewidth=0.8)
        ax.add_patch(circ)

    ax.set_aspect("equal")


def add_dim_arrow(ax, start, end, text, offset=(0, 0), fontsize=7, color="#555"):
    """Add a dimension arrow with text."""
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="<->", color=color, lw=0.8))
    mid_x = (start[0] + end[0]) / 2 + offset[0]
    mid_y = (start[1] + end[1]) / 2 + offset[1]
    ax.text(mid_x, mid_y, text, fontsize=fontsize, ha="center", va="center",
            color=color, fontweight="medium",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=1))


def plot_silhouettes(output_dir):
    # Columns: Reference, Best (orch_sp), Most Different (net_if)
    cols = ["reference", "orch_sp", "net_if"]
    col_labels = ["Reference (SLSQP)", "Orch + StagedPipe\n(Best Success)", "Net + IterFB\n(Most Different)"]

    fig, axes = plt.subplots(3, 3, figsize=FIGSIZE_3, dpi=DPI)

    for col_idx, key in enumerate(cols):
        p = ALL_PARAMS[key]
        handler = HANDLERS[key]
        color = HANDLER_COLORS[handler]

        # Top view
        ax_top = axes[0, col_idx]
        draw_top_view(ax_top, p, color)
        ax_top.set_title(col_labels[col_idx], fontsize=9, fontweight="bold", pad=8)
        if col_idx == 0:
            ax_top.set_ylabel("Top View\nx (m)", fontsize=9)

        # Dimension: span arrow
        semispan = p["semispan"]
        c_root = p["c_root"]
        add_dim_arrow(ax_top, (-semispan, c_root + 2), (semispan, c_root + 2),
                      f"span = {p['SPAN']:.1f} m", offset=(0, 1.5))

        # Side view
        ax_side = axes[1, col_idx]
        draw_side_view(ax_side, p, color)
        if col_idx == 0:
            ax_side.set_ylabel("Side View\ny (m)", fontsize=9)

        # Dimension: length
        fl = p["FUS_LENGTH"]
        fh = p["FUS_HEIGHT"]
        add_dim_arrow(ax_side, (0, -fh * 0.7), (fl, -fh * 0.7),
                      f"L = {fl:.1f} m", offset=(0, -fh * 0.2))
        # Dimension: height
        add_dim_arrow(ax_side, (fl + 2, -fh / 2), (fl + 2, fh / 2),
                      f"H = {fh:.1f} m", offset=(2.5, 0))

        # Front view
        ax_front = axes[2, col_idx]
        draw_front_view(ax_front, p, color)
        if col_idx == 0:
            ax_front.set_ylabel("Front View\ny (m)", fontsize=9)

        # Dimension: width
        fw = p["FUS_WIDTH"]
        add_dim_arrow(ax_front, (-fw / 2, -fh / 2 - 1.5), (fw / 2, -fh / 2 - 1.5),
                      f"W = {fw:.1f} m", offset=(0, -1.0))
        # Dimension: span on front view
        span = p["SPAN"]
        add_dim_arrow(ax_front, (-span / 2, fh / 2 + 1.5), (span / 2, fh / 2 + 1.5),
                      f"span = {span:.1f} m", offset=(0, 1.0))

    # Set axis limits per column for consistency
    for col_idx, key in enumerate(cols):
        p = ALL_PARAMS[key]
        semispan = p["semispan"]
        c_root = p["c_root"]
        fl = p["FUS_LENGTH"]
        fh = p["FUS_HEIGHT"]
        span = p["SPAN"]

        # Top view
        margin_top = max(semispan, fl) * 0.15
        axes[0, col_idx].set_xlim(-semispan - margin_top, semispan + margin_top)
        axes[0, col_idx].set_ylim(-fl + c_root * 0.4 - margin_top, c_root + 5)

        # Side view
        margin_side = fl * 0.1
        axes[1, col_idx].set_xlim(-margin_side, fl + 6)
        axes[1, col_idx].set_ylim(-fh * 1.2, fh * 1.5)

        # Front view
        margin_front = span * 0.1
        axes[2, col_idx].set_xlim(-span / 2 - margin_front, span / 2 + margin_front)
        axes[2, col_idx].set_ylim(-fh / 2 - 4, fh / 2 + 4)

    for ax_row in axes:
        for ax in ax_row:
            ax.tick_params(labelsize=7)

    fig.suptitle("Orthographic Silhouettes — Reference vs. Best vs. Most Different",
                 fontsize=13, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    for ext in ("pdf", "png", "svg"):
        fig.savefig(os.path.join(output_dir, f"fig3_silhouettes.{ext}"),
                    dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved fig3_silhouettes")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base, "figures")
    os.makedirs(output_dir, exist_ok=True)

    plot_planform(output_dir)
    plot_design_space(output_dir)
    plot_silhouettes(output_dir)

    print(f"\nAll paper figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
