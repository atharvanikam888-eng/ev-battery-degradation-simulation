"""
EV Battery Degradation Simulator
---------------------------------
Simulates how different charging habits affect battery capacity loss
over time, across multiple "virtual owners" with different usage
patterns.

Core idea: each charge cycle causes a small amount of capacity loss.
How much loss depends on three well-documented stress factors:
    1. Depth of Discharge (DoD)   - draining lower before recharging = more stress
    2. State of Charge (SOC) held - charging to 100% and sitting there = more stress
    3. Charge rate                - fast charging generates more heat/stress

This is a simplified empirical model (not a full electrochemical
model), but it captures the real, well-known trends manufacturers
warn about.

Requirements (install once with pip):
    pip install numpy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


# ---------------- SIMULATION SETTINGS ----------------
NUM_CYCLES = 1000          # number of charge/discharge cycles to simulate
                            # (roughly 3-4 years of daily use for many EV owners)
INITIAL_CAPACITY = 100.0   # start at 100% capacity

BASE_DEGRADATION_PER_CYCLE = 0.005  # base % capacity lost per cycle, before adjustments


# ---------------- DEFINE OWNER PROFILES ----------------
# Each owner has:
#   name         - label for the plot/summary
#   soc_max      - the charge level they typically charge to (%)
#   soc_min      - the charge level they typically drain to (%)
#   charge_rate  - 1 = slow/normal charging, 2 = fast charging habit

owners = [
    {
        "name": "Owner A: 0-100%, frequent fast charging",
        "soc_max": 100,
        "soc_min": 0,
        "charge_rate": 2,
    },
    {
        "name": "Owner B: 20-80%, slow charging",
        "soc_max": 80,
        "soc_min": 20,
        "charge_rate": 1,
    },
    {
        "name": "Owner C: 20-90%, frequent fast charging",
        "soc_max": 90,
        "soc_min": 20,
        "charge_rate": 2,
    },
    {
        "name": "Owner D: 30-80%, slow overnight charging",
        "soc_max": 80,
        "soc_min": 30,
        "charge_rate": 1,
    },
]


def simulate_owner(owner, num_cycles):
    """Run the degradation simulation for a single owner profile.

    Returns a numpy array of capacity (%) at each cycle.
    """
    capacity = INITIAL_CAPACITY
    dod = owner["soc_max"] - owner["soc_min"]  # depth of discharge (%)

    history = np.zeros(num_cycles)

    for cycle in range(num_cycles):

        # --- Depth of Discharge stress ---
        # Deeper discharge cycles cause more mechanical/chemical stress.
        # Normalised so a 100% DoD cycle roughly doubles base degradation.
        dod_factor = 1 + (dod / 100) * 1.0

        # --- High SOC stress ---
        # Sitting at/near 100% charge accelerates degradation.
        # No penalty below 80%, scaling penalty above it.
        if owner["soc_max"] > 80:
            high_soc_factor = 1 + ((owner["soc_max"] - 80) / 20) * 0.8
        else:
            high_soc_factor = 1

        # --- Charge rate stress ---
        # Fast charging generates more heat -> more degradation.
        rate_factor = 1.5 if owner["charge_rate"] == 2 else 1.0

        # --- Combine factors ---
        cycle_degradation = (
            BASE_DEGRADATION_PER_CYCLE * dod_factor * high_soc_factor * rate_factor
        )

        capacity -= cycle_degradation
        capacity = max(capacity, 0)  # capacity can't go below 0

        history[cycle] = capacity

    return history


def main():
    # ---------------- RUN SIMULATION ----------------
    capacity_history = np.zeros((len(owners), NUM_CYCLES))

    for i, owner in enumerate(owners):
        capacity_history[i] = simulate_owner(owner, NUM_CYCLES)

    

    # ---------------- PLOT RESULTS ----------------
    fig, ax = plt.subplots(figsize=(11, 6.5))

    cycles = np.arange(1, NUM_CYCLES + 1)

    colors = ["#D1495B", "#0077B6", "#E9A23B", "#2A9D8F"]
    labels = [
        "A | 0-100% SOC, fast",
        "B | 20-80% SOC, slow",
        "C | 20-90% SOC, fast",
        "D | 30-80% SOC, slow"
    ]

    for i, owner in enumerate(owners):
        ax.plot(
            cycles,
            capacity_history[i],
            color=colors[i],
            linewidth=2.6,
            linestyle="--" if i == 3 else "-",
            label=labels[i]
        )

        # Show final capacity beside each curve
        ax.scatter(
            NUM_CYCLES,
            capacity_history[i, -1],
            color=colors[i],
            s=35,
            zorder=5
        )

    # 80% battery health reference
    ax.axhline(
        80,
        color="#666666",
        linestyle="--",
        linewidth=1.2,
        alpha=0.8
    )

    ax.text(
        20, 81,
        "80% reference",
        color="#555555",
        fontsize=9
    )

    # Labels and title
    ax.set_title(
        "EV Battery Capacity Retention",
        fontsize=17,
        fontweight="bold",
        loc="left",
        pad=18
    )

    ax.set_xlabel("Charge / Discharge Events", fontsize=11)
    ax.set_ylabel("Remaining Capacity (%)", fontsize=11)

    # Focus on the region containing the results
    ax.set_xlim(0, NUM_CYCLES + 25)
    ax.set_ylim(65, 102)

    # Professional grid
    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.25
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend above the plotting area
    ax.legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.13),
    ncol=2,
    frameon=False,
    fontsize=9
)

    # Methodology note
    fig.text(
    0.10, 0.025,
        "Simplified linear degradation model | "
        "Illustrative coefficients | Not experimentally validated",
        fontsize=9,
        color="#666666"
    )

    fig.subplots_adjust(
    left=0.10,
    right=0.96,
    bottom=0.28,
    top=0.88
)

    fig.savefig(
        "battery_degradation_professional.png",
        dpi=300,
        facecolor="white"
    )

    plt.show()

    # ---------------- PRINT SUMMARY ----------------
    print(f"\n--- Battery Capacity After {NUM_CYCLES} Cycles ---")

    for i, owner in enumerate(owners):
        print(
            f"{owner['name']:<45}: "
            f"{capacity_history[i, -1]:.1f}% remaining capacity"
        )

    end_capacities = capacity_history[:, -1]

    best_idx = int(np.argmax(end_capacities))
    worst_idx = int(np.argmin(end_capacities))

    print(f"\nHighest capacity retention: {owners[best_idx]['name']}")
    print(f"Lowest capacity retention: {owners[worst_idx]['name']}")


if __name__ == "__main__":
    main()
