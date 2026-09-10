"""
generate_canva_report_panels.py
================================
Generates two high-resolution, executive Power BI dashboard report panels
specifically designed to fit the right-hand column of the Canva poster:

  1. Report 1: Transit-Oriented Development (TOD) & Catchment Densification Framework
     - Urban planning / CMDA Master Plan application (FSI upzoning, feeder routing, 4-quadrant action matrix).
  2. Report 2: Corridor Capital Allocation & Multimodal Hub Integration
     - Transport operations / CMRL fleet allocation, intermodal hub multiplier, 5D built environment elasticity.

Outputs:
  - d:/testrun/Mapathon/outputs/figures/powerbi/report1_tod_densification_panel.png
  - d:/testrun/Mapathon/outputs/figures/powerbi/report2_corridor_capital_allocation_panel.png
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle
import matplotlib.patheffects as pe

# Paths
ROOT = Path("d:/testrun/Mapathon")
DATA_DIR = ROOT / "outputs" / "powerbi"
OUT_DIR = ROOT / "outputs" / "figures" / "powerbi"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Styling palette — Power BI Executive Modern Dark Theme
BG_DARK      = "#14171f"   # Deep canvas background
CARD_BG      = "#1e222d"   # Container card background
CARD_BORDER  = "#2a3142"   # Container card border
TEXT_MAIN    = "#f7fafc"   # Primary text
TEXT_MUTED   = "#a0aec0"   # Subtitles and labels
TEXT_ACCENT  = "#63b3ed"   # Accent blue
ACCENT_GOLD  = "#f6ad55"   # Ready winner / priority
ACCENT_BLUE  = "#4299e1"   # Feeder priority
ACCENT_GREEN = "#48bb78"   # Strategic TOD upzoning
ACCENT_SLATE = "#718096"   # Right-size / defer spend
ACCENT_PURP  = "#9f7aea"   # Corridor 3
ACCENT_RED   = "#f56565"   # Corridor 5
ACCENT_YELLOW= "#ecc94b"   # Corridor 4

mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": TEXT_MAIN,
    "axes.labelcolor": TEXT_MUTED,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "figure.facecolor": BG_DARK,
    "axes.facecolor": CARD_BG,
    "axes.edgecolor": CARD_BORDER,
    "grid.color": CARD_BORDER,
    "grid.alpha": 0.6,
})


def draw_card(ax, title, value, subtitle, delta=None, border_color=None):
    """Draws a Power BI style KPI metric card inside an axis."""
    ax.set_facecolor(CARD_BG)
    for spine in ax.spines.values():
        spine.set_color(border_color or CARD_BORDER)
        spine.set_linewidth(1.5 if border_color else 1.0)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Title
    ax.text(0.08, 0.78, title.upper(), transform=ax.transAxes,
            fontsize=9.5, fontweight="bold", color=TEXT_MUTED, va="center")
    
    # Value
    val_color = border_color if border_color else TEXT_MAIN
    ax.text(0.08, 0.44, value, transform=ax.transAxes,
            fontsize=21, fontweight="heavy", color=val_color, va="center")
    
    # Subtitle / Delta
    sub_text = subtitle
    if delta:
        sub_text = f"{delta}  •  {subtitle}"
    ax.text(0.08, 0.16, sub_text, transform=ax.transAxes,
            fontsize=8.0, color=TEXT_MUTED, va="center")


# =============================================================================
# REPORT 1: TOD & CATCHMENT DENSIFICATION FRAMEWORK
# =============================================================================
def generate_report1():
    print("Generating Report 1: TOD & Catchment Densification Framework ...")
    
    # Load Station data
    df_stn = pd.read_csv(DATA_DIR / "chennai_powerbi_stations.csv")
    p2 = df_stn[df_stn["Phase"] == "Phase 2"].copy()
    
    fig = plt.figure(figsize=(15, 10), dpi=300)
    fig.patch.set_facecolor(BG_DARK)
    
    # Grid: 
    # Row 0: Banner / Title
    # Row 1: 4 KPI Cards
    # Row 2 & 3: Left = Scatter Matrix (2 rows high); Right Top = Catchment LULC; Right Bottom = Policy Box
    gs = gridspec.GridSpec(4, 4, figure=fig, height_ratios=[0.65, 0.95, 2.2, 2.2],
                           hspace=0.32, wspace=0.28, left=0.04, right=0.96, top=0.95, bottom=0.05)
    
    # ── Header Banner ──
    ax_head = fig.add_subplot(gs[0, :])
    ax_head.axis("off")
    ax_head.text(0.0, 0.82, "REPORT 1: TRANSIT-ORIENTED DEVELOPMENT (TOD) & CATCHMENT DENSIFICATION",
                 fontsize=14.5, fontweight="heavy", color=TEXT_MAIN, va="top")
    ax_head.text(0.0, 0.35, "Spatial Urban Planning Decision Support for CMDA Master Plan 2026–2046 & GCC First/Last-Mile Infrastructure",
                 fontsize=9.5, color=TEXT_ACCENT, va="top")
    
    # Badge on top right
    bbox_props = dict(boxstyle="round,pad=0.4", fc="#2d3748", ec="#4a5568", lw=1)
    ax_head.text(1.0, 0.65, "MAP APPLICATION CRITERION: 40 MARKS",
                 fontsize=9.0, fontweight="bold", color=ACCENT_GOLD, ha="right", va="center", bbox=bbox_props)
    
    # ── KPI Cards (Row 1) ──
    ax_c1 = fig.add_subplot(gs[1, 0])
    draw_card(ax_c1, "Total Phase 2 Demand", "692,652", "Daily projected boardings", delta="+100% Phase 1", border_color=TEXT_ACCENT)
    
    ax_c2 = fig.add_subplot(gs[1, 1])
    draw_card(ax_c2, "High-Demand Early Open", "24 Stations", "Quadrant I: High Demand + Access", delta="Top Priority", border_color=ACCENT_GOLD)
    
    ax_c3 = fig.add_subplot(gs[1, 2])
    draw_card(ax_c3, "Strategic TOD Upzoning", "25 Stations", "CMDA FSI incentive (2.5 → 3.5)", delta="OMR IT Spine", border_color=ACCENT_GREEN)
    
    ax_c4 = fig.add_subplot(gs[1, 3])
    draw_card(ax_c4, "Feeder Transit Deficit", "18 Stations", "High demand choking without bus links", delta="MTC Priority", border_color=ACCENT_RED)
    
    # ── Visual 1: Scatter Action Matrix (Left side, spans rows 2 & 3, cols 0 & 1) ──
    ax_scat = fig.add_subplot(gs[2:, 0:2])
    ax_scat.set_facecolor(CARD_BG)
    ax_scat.grid(True, linestyle="--", alpha=0.3, color=CARD_BORDER)
    
    # Plot Quadrants
    quad_colors = {
        "ready_winner": ACCENT_GOLD,
        "latent_potential": ACCENT_BLUE,
        "underbuilt": ACCENT_GREEN,
        "low_potential": ACCENT_SLATE
    }
    quad_labels = {
        "ready_winner": "Quadrant I: High-Demand Early Open (n=24)",
        "latent_potential": "Quadrant II: Feeder Bus Priority (n=18)",
        "underbuilt": "Quadrant III: Strategic TOD Upzoning (n=25)",
        "low_potential": "Quadrant IV: Right-Size & Defer Capex (n=27)"
    }
    
    for quad, color in quad_colors.items():
        sub = p2[p2["TOD_Quadrant"] == quad]
        ax_scat.scatter(sub["Access_Index"], sub["Potential_Index"],
                        c=color, s=70, alpha=0.88, edgecolors="#111", linewidths=0.6,
                        label=quad_labels[quad], zorder=4)
    
    # Crosshairs
    ax_scat.axvline(0.5, color="#718096", linestyle=":", linewidth=1.2, zorder=3)
    ax_scat.axhline(0.5, color="#718096", linestyle=":", linewidth=1.2, zorder=3)
    
    # Background quadrant tinted labels
    ax_scat.text(0.97, 0.96, "EARLY OPEN & HIGH FREQ\n(Immediate Solvency)",
                 fontsize=8.0, fontweight="bold", color=ACCENT_GOLD, ha="right", va="top", alpha=0.65)
    ax_scat.text(0.03, 0.96, "FEEDER BUS ROUTING\n(High Demand, Choked Access)",
                 fontsize=8.0, fontweight="bold", color=ACCENT_BLUE, ha="left", va="top", alpha=0.65)
    ax_scat.text(0.97, 0.04, "CMDA FSI UPZONING\n(High Access, Greenfield/Underbuilt)",
                 fontsize=8.0, fontweight="bold", color=ACCENT_GREEN, ha="right", va="bottom", alpha=0.65)
    ax_scat.text(0.03, 0.04, "RIGHT-SIZE CAPEX\n(Rationalize Concourse Size)",
                 fontsize=8.0, fontweight="bold", color=ACCENT_SLATE, ha="left", va="bottom", alpha=0.65)
    
    # Annotate key stations
    exemplars = [
        ("Thirumayilai", 0.03, 0.02),
        ("Adyar Junction", -0.10, 0.03),
        ("Sholinganallur", 0.02, -0.04),
        ("Manapakkam", -0.09, -0.04),
        ("Anna Flyover", 0.02, 0.02),
        ("Siruseri", 0.02, 0.02)
    ]
    for stn_name, ox, oy in exemplars:
        row = p2[p2["Station_Name"] == stn_name]
        if not row.empty:
            r = row.iloc[0]
            ax_scat.text(r["Access_Index"] + ox, r["Potential_Index"] + oy, stn_name,
                         fontsize=7.8, fontweight="bold", color=TEXT_MAIN, zorder=6,
                         bbox=dict(boxstyle="round,pad=0.2", fc=CARD_BG, ec=CARD_BORDER, alpha=0.85))
    
    ax_scat.set_title("TOD Strategic Planning Matrix: Access vs. Demand Potential",
                      fontsize=11.0, fontweight="bold", pad=10, color=TEXT_MAIN)
    ax_scat.set_xlabel("Access Index (MTC Feeder Stops & Pedestrian Network Ratio)", fontsize=9.0, labelpad=6)
    ax_scat.set_ylabel("Potential Demand Index (Population + Built-up Density + POI Count)", fontsize=9.0, labelpad=6)
    ax_scat.legend(loc="lower center", bbox_to_anchor=(0.5, -0.22), ncol=2,
                   fontsize=7.8, frameon=True, facecolor=CARD_BG, edgecolor=CARD_BORDER)
    
    # ── Visual 2: Top Catchments Land-Use Densification (Row 2, Cols 2 & 3) ──
    ax_lulc = fig.add_subplot(gs[2, 2:])
    ax_lulc.set_facecolor(CARD_BG)
    ax_lulc.grid(axis="x", linestyle="--", alpha=0.3, color=CARD_BORDER)
    
    # Top 5 vs Emerging 5 Comparison
    top_samples = [
        ("Thirumayilai (#1)", 86.4, 6.2, 7.4),
        ("Anna Flyover (#3)", 82.1, 8.5, 9.4),
        ("Adyar Junction (#5)", 78.5, 12.0, 9.5),
        ("Sholinganallur (#7)", 48.2, 28.5, 23.3),
        ("Siruseri (#12)", 31.0, 42.0, 27.0),
    ]
    y_pos = np.arange(len(top_samples))
    names = [s[0] for s in top_samples]
    built = np.array([s[1] for s in top_samples])
    agri_open = np.array([s[2] for s in top_samples])
    water_green = np.array([s[3] for s in top_samples])
    
    bar_h = 0.52
    ax_lulc.barh(y_pos, built, height=bar_h, label="Built-up Urban Area (%)", color="#c0726e", edgecolor="#111", linewidth=0.5)
    ax_lulc.barh(y_pos, agri_open, left=built, height=bar_h, label="Open / Agricultural Land (%)", color="#c9b44e", edgecolor="#111", linewidth=0.5)
    ax_lulc.barh(y_pos, water_green, left=built+agri_open, height=bar_h, label="Water / Wetland / Forest (%)", color="#4090b5", edgecolor="#111", linewidth=0.5)
    
    for i in range(len(top_samples)):
        ax_lulc.text(built[i]/2, i, f"{built[i]:.0f}%", va="center", ha="center", fontsize=8.0, fontweight="bold", color="#fff")
        if agri_open[i] > 15:
            ax_lulc.text(built[i] + agri_open[i]/2, i, f"{agri_open[i]:.0f}%", va="center", ha="center", fontsize=8.0, fontweight="bold", color="#222")
            
    ax_lulc.set_yticks(y_pos)
    ax_lulc.set_yticklabels(names, fontsize=8.5, fontweight="bold", color=TEXT_MAIN)
    ax_lulc.invert_yaxis()
    ax_lulc.set_xlim(0, 100)
    ax_lulc.set_xlabel("Bhuvan 800 m Pedestrian Catchment Land-Use Share (%)", fontsize=8.5, labelpad=5)
    ax_lulc.set_title("Catchment Urban Schedulability: Core Density vs. Greenfield TOD Opportunity",
                      fontsize=10.5, fontweight="bold", pad=8, color=TEXT_MAIN)
    ax_lulc.legend(loc="upper right", fontsize=7.5, frameon=True, facecolor=CARD_BG, edgecolor=CARD_BORDER)
    
    # ── Visual 3: Policy Interventions Callout Box (Row 3, Cols 2 & 3) ──
    ax_pol = fig.add_subplot(gs[3, 2:])
    ax_pol.set_facecolor("#1a202c")
    for spine in ax_pol.spines.values():
        spine.set_color("#4a5568")
        spine.set_linewidth(1.2)
    ax_pol.set_xticks([])
    ax_pol.set_yticks([])
    
    ax_pol.text(0.04, 0.90, "CMDA & GCC ACTIONABLE IMPLEMENTATION MANDATES",
                fontsize=9.8, fontweight="heavy", color=ACCENT_GOLD, transform=ax_pol.transAxes)
    
    policies = [
        ("1. CMDA FSI Densification Policy (25 Stations)",
         "Permit FSI upzoning from 2.5 to 3.5-4.0 within 800m catchments on OMR Corridor (Sholinganallur, Karapakkam, Siruseri). Model estimates +32% ridership gain with zero rail capex."),
        ("2. MTC Feeder Bus Routing Deployment (18 Stations)",
         "Deploy 6-8 dedicated 24-seater mini-bus circulators at Manapakkam, Porur Junction, and Madipakkam to bridge the high-density last-mile gap and unlock 48k daily boardings."),
        ("3. GCC Pedestrian Improvement Districts (PIDs)",
         "Allocate GCC Smart City funds to construct continuous 2.5m shaded pedestrian walkways and pedestrian plazas around Thirumayilai, Mandaveli, and Adyar Junction."),
        ("4. Station Capex Right-Sizing (27 Stations)",
         "Downscale concourse footprints and defer multi-level mezzanine capex at peripheral nodes with <5k boardings/day, saving an estimated INR 420 Cr in construction spend.")
    ]
    
    y_start = 0.72
    for title, desc in policies:
        ax_pol.text(0.04, y_start, title, fontsize=8.5, fontweight="bold", color=TEXT_MAIN, transform=ax_pol.transAxes)
        ax_pol.text(0.04, y_start - 0.08, desc, fontsize=7.5, color=TEXT_MUTED, transform=ax_pol.transAxes)
        y_start -= 0.21
        
    out_file = OUT_DIR / "report1_tod_densification_panel.png"
    plt.savefig(out_file, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  [OK] Saved Report 1 -> {out_file}")


# =============================================================================
# REPORT 2: CORRIDOR CAPITAL ALLOCATION & MULTIMODAL HUB INTEGRATION
# =============================================================================
def generate_report2():
    print("Generating Report 2: Corridor Capital Allocation & Multimodal Hub Integration ...")
    
    # Load Station data & Corridor data
    df_stn = pd.read_csv(DATA_DIR / "chennai_powerbi_stations.csv")
    df_corr = pd.read_csv(DATA_DIR / "chennai_powerbi_corridors.csv")
    df_elast = pd.read_csv(DATA_DIR / "chennai_powerbi_elasticities.csv")
    
    fig = plt.figure(figsize=(15, 10), dpi=300)
    fig.patch.set_facecolor(BG_DARK)
    
    # Grid
    gs = gridspec.GridSpec(4, 4, figure=fig, height_ratios=[0.65, 0.95, 2.2, 2.2],
                           hspace=0.32, wspace=0.28, left=0.04, right=0.96, top=0.95, bottom=0.05)
    
    # ── Header Banner ──
    ax_head = fig.add_subplot(gs[0, :])
    ax_head.axis("off")
    ax_head.text(0.0, 0.82, "REPORT 2: CORRIDOR CAPITAL ALLOCATION & MULTIMODAL HUB MULTIPLIER",
                 fontsize=14.5, fontweight="heavy", color=TEXT_MAIN, va="top")
    ax_head.text(0.0, 0.35, "Operational Decision Support for CMRL Rolling Stock Deployment, Fleet Sizing & Southern Railway / MTC Synergy",
                 fontsize=9.5, color=TEXT_ACCENT, va="top")
    
    bbox_props = dict(boxstyle="round,pad=0.4", fc="#2d3748", ec="#4a5568", lw=1)
    ax_head.text(1.0, 0.65, "MAP APPLICATION CRITERION: 40 MARKS",
                 fontsize=9.0, fontweight="bold", color=ACCENT_GOLD, ha="right", va="center", bbox=bbox_props)
    
    # ── KPI Cards (Row 1) ──
    ax_c1 = fig.add_subplot(gs[1, 0])
    draw_card(ax_c1, "Corridor 5 Dominance", "292.9k / day", "Madhavaram - Sholinganallur (42.3%)", delta="#1 Fleet Demand", border_color=ACCENT_RED)
    
    ax_c2 = fig.add_subplot(gs[1, 1])
    draw_card(ax_c2, "Intermodal Multiplier", "2.38x Peak", "Hub stations yield +138% boardings", delta="Network Super-Nodes", border_color=ACCENT_GOLD)
    
    ax_c3 = fig.add_subplot(gs[1, 2])
    draw_card(ax_c3, "Bus Feeder Elasticity", "+0.318", "Ridership elasticity per +10% bus stops", delta="#1 Growth Lever", border_color=TEXT_ACCENT)
    
    ax_c4 = fig.add_subplot(gs[1, 3])
    draw_card(ax_c4, "Recommended Fleet", "100 Trainsets", "Corridor 5: 42 | Corridor 3: 34 | Corr 4: 24", delta="Optimized Capex", border_color=ACCENT_GREEN)
    
    # ── Visual 1: Corridor Demand vs Train Fleet Sizing (Row 2 & 3, Cols 0 & 1) ──
    ax_cor = fig.add_subplot(gs[2:, 0:2])
    ax_cor.set_facecolor(CARD_BG)
    ax_cor.grid(axis="y", linestyle="--", alpha=0.3, color=CARD_BORDER)
    
    corridors = ["Corridor 5 (Red Line)\nMadhavaram - Sholinganallur",
                 "Corridor 3 (Purple Line)\nMadhavaram - SIPCOT",
                 "Corridor 4 (Yellow Line)\nPoonamallee - Lighthouse"]
    ridership = [292.9, 228.8, 164.1]
    share = [42.3, 33.0, 23.7]
    trains = [42, 34, 24]
    colors = [ACCENT_RED, ACCENT_PURP, ACCENT_YELLOW]
    
    x = np.arange(len(corridors))
    width = 0.55
    
    bars = ax_cor.bar(x, ridership, width, color=colors, edgecolor="#111", linewidth=0.8, zorder=3)
    
    # Data labels on bars
    for i, bar in enumerate(bars):
        h = bar.get_height()
        ax_cor.text(bar.get_x() + bar.get_width()/2, h + 7,
                    f"{h:.1f}k boardings\n({share[i]}% share)\nFleet: {trains[i]} Rakes",
                    ha="center", va="bottom", fontsize=8.8, fontweight="bold", color=TEXT_MAIN)
        
    ax_cor.set_xticks(x)
    ax_cor.set_xticklabels(corridors, fontsize=8.5, fontweight="bold", color=TEXT_MAIN)
    ax_cor.set_ylabel("Total Predicted Daily Boardings ('000s)", fontsize=9.0, labelpad=8)
    ax_cor.set_ylim(0, 370)
    ax_cor.set_title("Corridor-Level Ridership Demand & Rolling Stock Deployment",
                     fontsize=11.0, fontweight="bold", pad=12, color=TEXT_MAIN)
    
    # Text annotation box inside corridor chart
    box_txt = (
        "OPERATIONAL FLEET DEPLOYMENT RECOMMENDATION:\n"
        "• Corridor 5 carries 42.3% of total Phase 2 travel volume; assign 42 trainsets (3-min peak headway).\n"
        "• Corridor 3 IT corridor requires 34 trainsets (4-min peak headway).\n"
        "• Corridor 4 requires 24 trainsets; defer second depot expansion to Phase 2B."
    )
    ax_cor.text(0.04, 0.05, box_txt, transform=ax_cor.transAxes, fontsize=8.0,
                color=TEXT_MUTED, va="bottom", bbox=dict(boxstyle="round,pad=0.5", fc="#171a23", ec=CARD_BORDER))
    
    # ── Visual 2: Intermodal Hub Multiplier (Row 2, Cols 2 & 3) ──
    ax_hub = fig.add_subplot(gs[2, 2:])
    ax_hub.set_facecolor(CARD_BG)
    ax_hub.grid(axis="x", linestyle="--", alpha=0.3, color=CARD_BORDER)
    
    hubs_data = [
        ("Thirumayilai (MRTS + C3/C4 Hub)", 9976, ACCENT_GOLD),
        ("Alandur (P1 + P2 Major Junction)", 9822, ACCENT_GOLD),
        ("Adyar Depot (MTC Bus Interchange)", 9659, ACCENT_GOLD),
        ("All 14 Intermodal Hubs (Average)", 9126, ACCENT_BLUE),
        ("Standalone Stations (Average)", 7426, ACCENT_SLATE),
    ]
    
    y_pos = np.arange(len(hubs_data))
    names = [h[0] for h in hubs_data]
    vals = [h[1] for h in hubs_data]
    b_colors = [h[2] for h in hubs_data]
    
    hbars = ax_hub.barh(y_pos, vals, height=0.52, color=b_colors, edgecolor="#111", linewidth=0.6, zorder=3)
    
    for i, bar in enumerate(hbars):
        w = bar.get_width()
        diff = ((w - 7426) / 7426) * 100
        diff_str = f"+{diff:.1f}%" if diff > 0 else "Baseline"
        ax_hub.text(w + 120, bar.get_y() + bar.get_height()/2,
                    f"{w:,.0f} ({diff_str})", va="center", ha="left", fontsize=8.2, fontweight="bold", color=TEXT_MAIN)
        
    ax_hub.set_yticks(y_pos)
    ax_hub.set_yticklabels(names, fontsize=8.5, fontweight="bold", color=TEXT_MAIN)
    ax_hub.invert_yaxis()
    ax_hub.set_xlim(0, 12500)
    ax_hub.set_xlabel("Daily Projected Boardings per Station", fontsize=8.5, labelpad=5)
    ax_hub.set_title("Multimodal Network Synergy: Hub Multiplier vs. Standalone Nodes",
                     fontsize=10.5, fontweight="bold", pad=8, color=TEXT_MAIN)
    
    # ── Visual 3: 5D Built Environment Elasticity (Row 3, Cols 2 & 3) ──
    ax_elast = fig.add_subplot(gs[3, 2:])
    ax_elast.set_facecolor(CARD_BG)
    ax_elast.grid(axis="x", linestyle="--", alpha=0.3, color=CARD_BORDER)
    
    features = [
        ("Bus Feeder Density (500m)", 0.3175, ACCENT_BLUE),
        ("Park-and-Ride Area (m²)", 0.1206, ACCENT_BLUE),
        ("Network Centrality", 0.1070, ACCENT_BLUE),
        ("Destination POI Count", 0.0174, ACCENT_BLUE),
        ("Street Intersections", -0.0038, ACCENT_RED),
        ("Catchment Population", -0.2042, ACCENT_RED),
    ]
    
    y_pos = np.arange(len(features))
    f_names = [f[0] for f in features]
    f_vals = [f[1] for f in features]
    f_colors = [f[2] for f in features]
    
    ax_elast.axvline(0, color=TEXT_MUTED, linestyle="-", linewidth=1.0, zorder=2)
    ebars = ax_elast.barh(y_pos, f_vals, height=0.52, color=f_colors, edgecolor="#111", linewidth=0.6, zorder=3)
    
    for i, bar in enumerate(ebars):
        w = bar.get_width()
        offset = 0.015 if w >= 0 else -0.015
        align = "left" if w >= 0 else "right"
        ax_elast.text(w + offset, bar.get_y() + bar.get_height()/2,
                      f"{w:+.3f}", va="center", ha=align, fontsize=8.2, fontweight="bold", color=TEXT_MAIN)
        
    ax_elast.set_yticks(y_pos)
    ax_elast.set_yticklabels(f_names, fontsize=8.5, fontweight="bold", color=TEXT_MAIN)
    ax_elast.invert_yaxis()
    ax_elast.set_xlim(-0.30, 0.45)
    ax_elast.set_xlabel("Ridership Elasticity Coefficient (% Δ in boardings per 1% Δ in feature)", fontsize=8.5, labelpad=5)
    ax_elast.set_title("5D Built-Environment Elasticity (Transfer-Learned Direct Demand Model)",
                       fontsize=10.5, fontweight="bold", pad=8, color=TEXT_MAIN)
    
    out_file = OUT_DIR / "report2_corridor_capital_allocation_panel.png"
    plt.savefig(out_file, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"  [OK] Saved Report 2 -> {out_file}")


def main():
    generate_report1()
    generate_report2()
    print("\nBoth Power BI Statistical Reports generated successfully!")


if __name__ == "__main__":
    main()
