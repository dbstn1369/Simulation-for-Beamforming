import numpy as np
import matplotlib.pyplot as plt

# 텍스트 파일에서 데이터 읽기
total_times_algo1 = np.genfromtxt('total_time.txt')
total_sts_algo1 = np.genfromtxt('total_STS.txt')
total_times_algo2 = np.genfromtxt('total_time_Q.txt')
total_sts_algo2 = np.genfromtxt('total_STS_Q.txt')

# 에피소드 번호 생성 (1부터 시작)
episode_numbers = np.arange(1, len(total_times_algo1) + 1)

# 20개씩 묶어 평균을 구함
bin_size = 500
total_times_avg_algo1 = np.mean(total_times_algo1.reshape(-1, bin_size), axis=1)
total_sts_avg_algo1 = np.mean(total_sts_algo1.reshape(-1, bin_size), axis=1)
total_times_avg_algo2 = np.mean(total_times_algo2.reshape(-1, bin_size), axis=1)
total_sts_avg_algo2 = np.mean(total_sts_algo2.reshape(-1, bin_size), axis=1)
episode_numbers_avg = np.arange(1, len(total_times_avg_algo1) + 1) * bin_size

# Set the plot style to 'classic' for a more suitable style for IEEE papers
plt.style.use('classic')

# Create the plot
fig, ax1 = plt.subplots()

# Plot the time data on the first y-axis
ax1.plot(episode_numbers_avg, total_times_avg_algo1, label='Time A_C', color='b')
ax1.plot(episode_numbers_avg, total_times_avg_algo2, label='Time Q', color='g')
ax1.set_xlabel('Episode Number', fontsize=12)
ax1.set_ylabel('Time (ms)', fontsize=12)

# Set the grid
ax1.grid(True)

# Customize the tick labels size
ax1.tick_params(axis='both', which='major', labelsize=10)

# Create the second y-axis
ax2 = ax1.twinx()

# Plot the STS data on the second y-axis
ax2.plot(episode_numbers_avg, total_sts_avg_algo1, label='STS Count A_C', color='r')
ax2.plot(episode_numbers_avg, total_sts_avg_algo2, label='STS Count Q', color='m')
ax2.set_ylabel('STS Count', fontsize=12)

# Customize the tick labels size
ax2.tick_params(axis='both', which='major', labelsize=10)

# Combine the legends of both y-axes
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10)

# Set the title
plt.title('Time and STS Count per Episode for Two Algorithms', fontsize=14)

# Save the
