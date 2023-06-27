import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rcParams['font.family'] = 'Times New Roman'

# Define the N_sta values, STS values, and the algorithm names
n_sta_values = [200, 300, 400, 500]
sts_values = [4, 8, 16, 32]
algorithms = ['Actor-Critic', 'Q-learning', 'Original']

# Bar width and colors
bar_width = 0.15
colors = ['#26456e', '#3a87b7', '#b4d4da']

# Assume the energy per slot
energy_per_slot = 0.001  # e.g., 0.001 Joules

# Additional energy cost per STA
additional_energy_per_sta = 0.005  # e.g., 0.0005 Joules

# Create a figure and axes
fig, axs = plt.subplots(1, len(sts_values), figsize=(4 * len(sts_values), 6), sharey=True)

for ax_index, sts in enumerate(sts_values):
    # Position of the left bar boundaries
    bar_l = np.arange(len(n_sta_values))

    # Generate positions of the x-axis ticks
    tick_pos = [i+(bar_width/2) for i in bar_l]

    for algo_index, algo in enumerate(algorithms):
        # Read the STS data files for each STS and N_sta value
        energy_values = []
        for n_sta in n_sta_values:
            total_sts = np.genfromtxt(f'total_STS_{algo}_{sts}_{n_sta}.txt')  # replace with the actual file path
            total_energy = total_sts * energy_per_slot + n_sta * additional_energy_per_sta
            avg_energy = np.mean(total_energy)
            energy_values.append(avg_energy)

        # Plot a bar for the current algorithm in each STS scenario
        axs[ax_index].bar([p + bar_width*algo_index for p in bar_l], energy_values, 
                           width=bar_width, color=colors[algo_index])

    # Set the x-axis ticks and labels
    axs[ax_index].set_xticks(tick_pos)
    axs[ax_index].set_xticklabels(n_sta_values)

    # Set axes title and labels
    axs[ax_index].set_title(f'$STS_{{init}}$ = {sts}', fontsize=14)  # Use LaTeX syntax for subscript
    if ax_index == 0:
        axs[ax_index].set_ylabel('Average Energy', fontsize=12)

    # Enable grid lines
    axs[ax_index].grid(True)

    # Add the legend in the last subplot only
    if ax_index == len(sts_values) - 1:
        axs[ax_index].legend(algorithms, loc='upper right')

# Add the common x-label for all subplots
fig.text(0.5, 0.01, 'Number of STAs', ha='center', va='center', fontsize=12)

# Save the figure
plt.tight_layout()
plt.savefig('Average_Energy_Algorithms.png', dpi=300)

# Show the figure
plt.show()
