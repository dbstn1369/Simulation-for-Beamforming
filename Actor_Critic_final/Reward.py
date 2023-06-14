import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

# Your existing code...
episode_times = np.genfromtxt('Reward_AC.txt')
episode_numbers = np.arange(1, len(episode_times) + 1)

bin_size = 1000
episode_times_avg = np.mean(episode_times.reshape(-1, bin_size), axis=1)
episode_numbers_avg = np.arange(1, len(episode_times_avg) + 1) * bin_size

# Calculate the smoothed rewards with a rolling window (for example, window size = 10)
smoothed_rewards = pd.Series(episode_times_avg).rolling(window=10).mean()

plt.style.use('classic')
fig, ax = plt.subplots()
ax.plot(episode_numbers_avg, episode_times_avg, color = 'royalblue')
ax.plot(episode_numbers_avg, smoothed_rewards, color='orange')  # Plot the smoothed rewards in yellow



ax.set_xlabel('Time steps', fontsize=12)
ax.set_ylabel('Reward', fontsize=12)
#ax.set_title('Reward per Episode', fontsize=14)
ax.grid(True)
ax.tick_params(axis='both', which='major', labelsize=10)
fig.savefig('Reward_AC_per_episode.png', dpi=300)
plt.show()
