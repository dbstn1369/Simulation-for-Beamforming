import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set the font family to a specific font, e.g., 'Times New Roman'
mpl.rcParams['font.family'] = 'Times New Roman'

# 텍스트 파일에서 데이터 읽기
total_times = np.genfromtxt('total_time_Q.txt')
total_sts = np.genfromtxt('total_STS_Q.txt')

# 에피소드 번호 생성 (1부터 시작)
episode_numbers = np.arange(1, len(total_times) + 1)

# 20개씩 묶어 평균을 구함
bin_size = 2000
total_times_avg = np.mean(total_times.reshape(-1, bin_size), axis=1)
total_sts_avg = np.mean(total_sts.reshape(-1, bin_size), axis=1)
episode_numbers_avg = np.arange(1, len(total_times_avg) + 1) * bin_size

# Set the plot style to 'classic' for a more suitable style for IEEE papers
plt.style.use('classic')

# Create the plot
fig, ax1 = plt.subplots()

# Plot the time data on the first y-axis
#ax1.plot(episode_numbers_avg, total_times_avg, label='Time', color='b')
# Plot the time data on the first y-axis with circles as markers
ax1.plot(episode_numbers_avg, total_times_avg, label='Time', color='b', marker='s')


ax1.set_xlabel('Number of Episode', fontsize=12)
ax1.set_ylabel('Time (ms)', fontsize=12)

# Set the grid
ax1.grid(True)

# Customize the tick labels size
ax1.tick_params(axis='both', which='major', labelsize=10)

# Create the second y-axis
ax2 = ax1.twinx()

# Plot the STS data on the second y-axis
#ax2.plot(episode_numbers_avg, total_sts_avg, label='STS Count', color='r')
# Plot the STS data on the second y-axis with triangles as markers
ax2.plot(episode_numbers_avg, total_sts_avg, label='Total STS', color='r', marker='^')
ax2.set_ylabel('Total STS', fontsize=12)

# Customize the tick labels size
ax2.tick_params(axis='both', which='major', labelsize=10)

# Combine the legends of both y-axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10)

# Set the title
plt.title('Q-learning', fontsize=14)

# Save the plot as a high-quality image
fig.savefig('BF_Q.png', dpi=300)

# Show the plot
plt.show()