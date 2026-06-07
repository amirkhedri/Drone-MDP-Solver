"""
visualize.py — Required Visualizations for Drone MDP Challenge
─────────────────────────────────────────────────────────────────────────────
Generates three figures:
  1. value_heatmap.png  — V(s) heatmap per damage level
  2. convergence.png    — max|ΔV| per iteration (log-scale Y)
  3. policy_map.png     — directional arrows per damage level

Usage:
  1. Run the server: python server.py
  2. Click ▶ Run in the browser to compute the policy
     (this triggers compute_policy() which saves convergence_data.json
      and value_data.json next to policy.py / server.py)
  3. Then run: python visualize.py

The script auto-detects the JSON data files in the same directory.
─────────────────────────────────────────────────────────────────────────────
"""

import json
import os
import sys
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend — safe on all systems
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyArrow


# ─────────────────────────────────────────────────────────────────────────────
#  0.  Load data
# ─────────────────────────────────────────────────────────────────────────────

def _find_file(name):
    """Search for a data file next to this script, or in cwd."""
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), name),
        os.path.join(os.getcwd(), name),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def load_data():
    conv_path = _find_file("convergence_data.json")
    val_path  = _find_file("value_data.json")

    if not conv_path or not val_path:
        print("ERROR: Data files not found. Please run the policy first via the browser.")
        print("  Expected: convergence_data.json and value_data.json")
        sys.exit(1)

    with open(conv_path) as f:
        convergence = json.load(f)

    with open(val_path) as f:
        val_data = json.load(f)

    print(f"[visualize.py] Loaded {len(convergence)} convergence iterations")
    print(f"[visualize.py] Grid: {val_data['rows']}×{val_data['cols']}  "
          f"damage levels: 0–{val_data['max_damage']}")

    return convergence, val_data


def build_arrays(val_data):
    """Convert flat list of state dicts into 3-D numpy arrays."""
    rows       = val_data['rows']
    cols       = val_data['cols']
    max_damage = val_data['max_damage']   # = 4

    n_dmg = max_damage + 1   # 0..4

    # Arrays indexed [damage, row, col]
    V_arr      = np.full((n_dmg, rows, cols), np.nan)
    action_arr = np.full((n_dmg, rows, cols), None, dtype=object)
    cell_arr   = np.full((n_dmg, rows, cols), '', dtype=object)
    terminal   = np.zeros((n_dmg, rows, cols), dtype=bool)

    for entry in val_data['states']:
        r, c, d = entry['r'], entry['c'], entry['d']
        V_arr[d, r, c]      = entry['V']
        action_arr[d, r, c] = entry['action']
        cell_arr[d, r, c]   = entry['cell']
        terminal[d, r, c]   = entry['terminal']

    return V_arr, action_arr, cell_arr, terminal


# ─────────────────────────────────────────────────────────────────────────────
#  1.  VALUE HEATMAP   →  value_heatmap.png
# ─────────────────────────────────────────────────────────────────────────────

def plot_value_heatmap(V_arr, cell_arr, terminal, rows, cols, save_path="value_heatmap.png"):
    n_dmg = V_arr.shape[0]
    fig, axes = plt.subplots(1, n_dmg, figsize=(4.5 * n_dmg, 4.8))
    fig.patch.set_facecolor('#0d1117')

    # Colour limits across ALL damage levels (consistent scale)
    all_vals = V_arr[~np.isnan(V_arr)]
    vmin, vmax = float(np.min(all_vals)), float(np.max(all_vals))
    vcenter = 0.0 if vmin < 0 < vmax else (vmin + vmax) / 2

    for d in range(n_dmg):
        ax = axes[d] if n_dmg > 1 else axes
        ax.set_facecolor('#161b22')

        V_d = V_arr[d].copy()

        # Build mask arrays
        obstacle_mask = np.isnan(V_d)
        goal_mask     = terminal[d]

        # Temporarily fill obstacles/goals for display
        V_display = np.where(obstacle_mask, np.nan, V_d)
        V_display = np.where(goal_mask, vmax, V_display)

        norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax) \
               if vmin < vcenter < vmax else None

        im = ax.imshow(
            V_display,
            cmap='RdYlGn',
            norm=norm,
            vmin=vmin if norm is None else None,
            vmax=vmax if norm is None else None,
            aspect='equal',
            origin='upper',
            interpolation='nearest',
        )

        # Overlay obstacle cells (dark grey)
        for r in range(rows):
            for c in range(cols):
                if obstacle_mask[r, c]:
                    ax.add_patch(plt.Rectangle(
                        (c - 0.5, r - 0.5), 1, 1,
                        color='#21262d', zorder=2))
                elif goal_mask[r, c]:
                    ax.add_patch(plt.Rectangle(
                        (c - 0.5, r - 0.5), 1, 1,
                        color='#2ea043', alpha=0.85, zorder=2))
                    ax.text(c, r, '★', ha='center', va='center',
                            fontsize=11, color='white', zorder=3, fontweight='bold')

        # Cell type overlays
        for r in range(rows):
            for c in range(cols):
                ctype = cell_arr[d, r, c]
                if ctype == 'storm':
                    ax.add_patch(plt.Rectangle(
                        (c - 0.5, r - 0.5), 1, 1,
                        color='#ff7b72', alpha=0.25, zorder=2))
                elif ctype == 'medkit':
                    ax.add_patch(plt.Rectangle(
                        (c - 0.5, r - 0.5), 1, 1,
                        color='#3fb950', alpha=0.30, zorder=2))
                elif ctype in ('portal_a', 'portal_b'):
                    ax.add_patch(plt.Rectangle(
                        (c - 0.5, r - 0.5), 1, 1,
                        color='#a371f7', alpha=0.30, zorder=2))

        # Annotate value text in each cell
        for r in range(rows):
            for c in range(cols):
                if not obstacle_mask[r, c]:
                    val = V_d[r, c]
                    if not np.isnan(val):
                        txt = f"{val:.0f}" if abs(val) >= 10 else f"{val:.1f}"
                        ax.text(c, r, txt, ha='center', va='center',
                                fontsize=6.5, color='#e6edf3', zorder=4,
                                fontweight='bold')

        # Grid lines
        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which='minor', color='#30363d', linewidth=0.5)
        ax.tick_params(which='minor', length=0)
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        ax.set_xticklabels(range(cols), color='#8b949e', fontsize=8)
        ax.set_yticklabels(range(rows), color='#8b949e', fontsize=8)
        ax.tick_params(colors='#8b949e', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')

        ax.set_title(f'Damage = {d}', color='#e6edf3', fontsize=12,
                     fontweight='bold', pad=8)
        ax.set_xlabel('Column', color='#8b949e', fontsize=9)
        ax.set_ylabel('Row', color='#8b949e', fontsize=9)

        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(
            color='#8b949e', labelcolor='#8b949e')

    # Legend
    legend_elements = [
        mpatches.Patch(color='#2ea043', alpha=0.85, label='Goal (★)'),
        mpatches.Patch(color='#ff7b72', alpha=0.5,  label='Storm cell'),
        mpatches.Patch(color='#3fb950', alpha=0.5,  label='Medkit cell'),
        mpatches.Patch(color='#a371f7', alpha=0.5,  label='Portal cell'),
        mpatches.Patch(color='#21262d',              label='Obstacle (wall)'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=5,
               facecolor='#161b22', edgecolor='#30363d',
               labelcolor='#e6edf3', fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle('Value Function Heatmap  —  V(s) per Damage Level',
                 color='#e6edf3', fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='#0d1117')
    plt.close()
    print(f"[visualize.py] ✓ Saved  {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  2.  CONVERGENCE CURVE   →  convergence.png
# ─────────────────────────────────────────────────────────────────────────────

THETA_DEFAULT = 1e-6   # must match policy.py


def plot_convergence(convergence_history, theta=THETA_DEFAULT,
                     save_path="convergence.png"):
    if not convergence_history:
        print("[visualize.py] No convergence data — skipping convergence.png")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')

    iters = np.arange(1, len(convergence_history) + 1)
    deltas = np.array(convergence_history, dtype=float)

    # Replace zeros (after full convergence) with tiny value for log plot
    deltas_plot = np.where(deltas == 0, 1e-12, deltas)

    ax.semilogy(iters, deltas_plot, color='#58a6ff', linewidth=2,
                label='max |ΔV| per iteration', zorder=3)

    # Theta threshold line
    ax.axhline(theta, color='#f85149', linewidth=1.5, linestyle='--',
               label=f'Convergence threshold θ = {theta:.0e}', zorder=4)

    # Shade converged region
    conv_iter = None
    for i, d in enumerate(deltas):
        if d < theta:
            conv_iter = i + 1
            break
    if conv_iter:
        ax.axvline(conv_iter, color='#3fb950', linewidth=1.5, linestyle=':',
                   label=f'Converged at iteration {conv_iter}', zorder=4)
        ax.fill_betweenx([theta * 0.01, deltas_plot.max() * 2],
                         conv_iter, len(iters),
                         alpha=0.08, color='#3fb950', zorder=1)
        ax.text(conv_iter + 1, theta * 3,
                f'Converged\n(iter {conv_iter})',
                color='#3fb950', fontsize=9, va='bottom')

    ax.set_xlabel('Iteration', color='#8b949e', fontsize=11)
    ax.set_ylabel('max |V_new(s) − V_old(s)|   [log scale]',
                  color='#8b949e', fontsize=11)
    ax.set_title('Value Iteration — Convergence Curve',
                 color='#e6edf3', fontsize=13, fontweight='bold')

    ax.set_xlim(1, max(len(iters), 10))
    ax.set_ylim(min(deltas_plot) * 0.1, deltas_plot.max() * 5)

    ax.tick_params(colors='#8b949e', labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor('#30363d')
    ax.yaxis.set_tick_params(labelcolor='#8b949e')
    ax.xaxis.set_tick_params(labelcolor='#8b949e')

    ax.grid(True, which='both', color='#21262d', linewidth=0.8, linestyle='-')
    ax.grid(True, which='minor', color='#1c2330', linewidth=0.4, linestyle=':')

    legend = ax.legend(facecolor='#1c2330', edgecolor='#30363d',
                       labelcolor='#e6edf3', fontsize=10, loc='upper right')

    # Summary annotation
    final_delta = deltas[-1]
    n_iters     = len(deltas)
    summary = (f"Total iterations: {n_iters}\n"
               f"Final Δ: {final_delta:.2e}\n"
               f"θ: {theta:.0e}")
    ax.text(0.02, 0.05, summary, transform=ax.transAxes,
            color='#8b949e', fontsize=9, va='bottom',
            bbox=dict(facecolor='#1c2330', edgecolor='#30363d', alpha=0.8,
                      boxstyle='round,pad=0.4'))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='#0d1117')
    plt.close()
    print(f"[visualize.py] ✓ Saved  {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  3.  POLICY MAP   →  policy_map.png
# ─────────────────────────────────────────────────────────────────────────────

ARROW_DIR = {
    'N': ( 0,  -1),   # row decreases → up on plot (origin='upper')
    'S': ( 0,   1),
    'E': ( 1,   0),
    'W': (-1,   0),
}

CELL_COLORS = {
    'goal':     '#2ea043',
    'obstacle': '#21262d',
    'storm':    '#ff7b72',
    'medkit':   '#3fb950',
    'portal_a': '#a371f7',
    'portal_b': '#a371f7',
    'normal':   '#161b22',
}


def plot_policy_map(action_arr, cell_arr, terminal, rows, cols,
                    save_path="policy_map.png"):
    n_dmg = action_arr.shape[0]
    fig, axes = plt.subplots(1, n_dmg, figsize=(4.5 * n_dmg, 4.8))
    fig.patch.set_facecolor('#0d1117')

    for d in range(n_dmg):
        ax = axes[d] if n_dmg > 1 else axes
        ax.set_facecolor('#0d1117')
        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(rows - 0.5, -0.5)   # origin=upper

        # Draw cell backgrounds
        for r in range(rows):
            for c in range(cols):
                ctype = cell_arr[d, r, c]
                if ctype == '':
                    color = CELL_COLORS['obstacle']
                elif terminal[d, r, c]:
                    color = CELL_COLORS['goal']
                else:
                    color = CELL_COLORS.get(ctype, CELL_COLORS['normal'])
                rect = plt.Rectangle(
                    (c - 0.5, r - 0.5), 1, 1,
                    color=color, linewidth=0, zorder=1)
                ax.add_patch(rect)

        # Grid lines
        for x in np.arange(-0.5, cols, 1):
            ax.axvline(x, color='#30363d', linewidth=0.5, zorder=2)
        for y in np.arange(-0.5, rows, 1):
            ax.axhline(y, color='#30363d', linewidth=0.5, zorder=2)

        # Draw arrows & special markers
        arrow_len = 0.32
        for r in range(rows):
            for c in range(cols):
                ctype = cell_arr[d, r, c]
                if ctype == '':
                    # Obstacle: cross mark
                    ax.text(c, r, '✕', ha='center', va='center',
                            fontsize=9, color='#484f58', zorder=3)
                    continue

                if terminal[d, r, c]:
                    # Goal cell
                    ax.text(c, r, '★', ha='center', va='center',
                            fontsize=13, color='white', zorder=3,
                            fontweight='bold')
                    continue

                act = action_arr[d, r, c]
                if act is None:
                    continue

                # Storm / medkit / portal small indicator (ASCII — safe on all systems)
                if ctype == 'storm':
                    ax.text(c, r + 0.38, 'S', ha='center', va='center',
                            fontsize=6, color='#ff7b72', zorder=3,
                            fontweight='bold')
                elif ctype == 'medkit':
                    ax.text(c, r + 0.38, '+', ha='center', va='center',
                            fontsize=8, color='#3fb950', zorder=3,
                            fontweight='bold')
                elif ctype in ('portal_a', 'portal_b'):
                    ax.text(c, r + 0.38, 'P', ha='center', va='center',
                            fontsize=6, color='#a371f7', zorder=3,
                            fontweight='bold')

                # Arrow
                dx, dy = ARROW_DIR[act]
                ax.annotate(
                    '',
                    xy    =(c + dx * arrow_len, r + dy * arrow_len),
                    xytext=(c - dx * arrow_len * 0.7,
                            r - dy * arrow_len * 0.7),
                    arrowprops=dict(
                        arrowstyle='->', color='#58a6ff',
                        lw=1.6, mutation_scale=14),
                    zorder=4,
                )

        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        ax.set_xticklabels(range(cols), color='#8b949e', fontsize=8)
        ax.set_yticklabels(range(rows), color='#8b949e', fontsize=8)
        ax.tick_params(colors='#8b949e', labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')
        ax.set_xlabel('Column', color='#8b949e', fontsize=9)
        ax.set_ylabel('Row',    color='#8b949e', fontsize=9)
        ax.set_title(f'Damage = {d}', color='#e6edf3',
                     fontsize=12, fontweight='bold', pad=8)

    # Legend
    legend_elements = [
        mpatches.Patch(color='#2ea043',  label='Goal  ★'),
        mpatches.Patch(color='#21262d',  label='Obstacle  ✕'),
        mpatches.Patch(color='#ff7b72',  label='Storm cell'),
        mpatches.Patch(color='#3fb950',  label='Medkit cell'),
        mpatches.Patch(color='#a371f7',  label='Portal cell'),
        mpatches.Patch(color='#161b22',  label='Normal cell'),
        mpatches.FancyArrow(0, 0, 1, 0, color='#58a6ff',
                            width=0.3, label='Optimal action'),
    ]
    fig.legend(handles=legend_elements, loc='lower center', ncol=7,
               facecolor='#161b22', edgecolor='#30363d',
               labelcolor='#e6edf3', fontsize=9,
               bbox_to_anchor=(0.5, -0.03))

    fig.suptitle('Optimal Policy Map  —  Directional Arrows per Damage Level',
                 color='#e6edf3', fontsize=14, fontweight='bold', y=1.02)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='#0d1117')
    plt.close()
    print(f"[visualize.py] ✓ Saved  {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("  📊  Drone MDP — Visualization Script")
    print("=" * 60)

    convergence, val_data = load_data()

    rows = val_data['rows']
    cols = val_data['cols']

    V_arr, action_arr, cell_arr, terminal = build_arrays(val_data)

    print("\n[visualize.py] Generating plots …")

    plot_value_heatmap(V_arr, cell_arr, terminal, rows, cols,
                       save_path="value_heatmap.png")

    plot_convergence(convergence, theta=THETA_DEFAULT,
                     save_path="convergence.png")

    plot_policy_map(action_arr, cell_arr, terminal, rows, cols,
                    save_path="policy_map.png")

    print("\n[visualize.py] ✓ All three figures saved:")
    print("    • value_heatmap.png")
    print("    • convergence.png")
    print("    • policy_map.png")
    print("=" * 60)
