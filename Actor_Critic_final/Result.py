import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set the font family to a specific font, e.g., 'Arial'
mpl.rcParams['font.family'] = 'Arial'

# 텍스트 파일에서 데이터 읽기
episode_times = np.genfromtxt('total_time.txt')
# 에피소드 번호 생성 (1부터 시작)
episode_numbers = np.arange(1, len(episode_times) + 1)


bin_size = 200
episode_times_avg = np.mean(episode_times.reshape(-1, bin_size), axis=1)
episode_numbers_avg = np.arange(1, len(episode_times_avg) + 1) * bin_size

# Set the plot style to 'classic' for a more suitable style for IEEE papers
plt.style.use('classic')

# Create the plot
fig, ax = plt.subplots()

# Plot the data
ax.plot(episode_numbers_avg, episode_times_avg)

# Set the labels and title
ax.set_xlabel('Episode Number', fontsize=12)
ax.set_ylabel('Time (ms)', fontsize=12)
ax.set_title('Time per Episode', fontsize=14)

# Set the grid
ax.grid(True)

# Customize the tick labels size
ax.tick_params(axis='both', which='major', labelsize=10)

# Save the plot as a high-quality image
fig.savefig('time_per_episode_averaged.png', dpi=300)

# Show the plot
plt.show()