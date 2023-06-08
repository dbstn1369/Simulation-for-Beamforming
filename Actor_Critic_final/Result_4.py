import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams['font.family'] = 'Times New Roman'

# 텍스트 파일에서 데이터 읽기
total_sts_algo1 = np.genfromtxt('total_STS.txt')
total_sts_algo2 = np.genfromtxt('total_STS_Q.txt')
total_sts_algo3 = np.genfromtxt('total_STS_O.txt')  # New data

# 에피소드 번호 생성 (1부터 시작)
episode_numbers = np.arange(1, len(total_sts_algo1) + 1)

bin_size = 100

total_sts_avg_algo1 = np.mean(total_sts_algo1.reshape(-1, bin_size), axis=1)
total_sts_avg_algo2 = np.mean(total_sts_algo2.reshape(-1, bin_size), axis=1)
total_sts_avg_algo3 = np.mean(total_sts_algo3.reshape(-1, bin_size), axis=1)  # New average

episode_numbers_avg = np.arange(1, len(total_sts_avg_algo1) + 1) * bin_size

# Set the plot style to 'classic' for a more suitable style for IEEE papers
plt.style.use('classic')

# Create the plot
fig, ax1 = plt.subplots()

# Plot the STS data
ax1.plot(episode_numbers_avg, total_sts_avg_algo1, label='STS A-C', color='r', marker='s')
ax1.plot(episode_numbers_avg, total_sts_avg_algo2, label='STS Q', color='m', marker='^')
ax1.plot(episode_numbers_avg, total_sts_avg_algo3, label='STS O', color='b', marker='o')  # New plot

ax1.set_xlabel('Episode Number', fontsize=12)
ax1.set_ylabel('Total STS', fontsize=12)

# Set the grid
ax1.grid(True)

# Customize the tick labels size
ax1.tick_params(axis='both', which='major', labelsize=10)

# Combine the legends
lines1, labels1 = ax1.get_legend_handles_labels()
ax1.legend(lines1, labels1, fontsize=10)

# Set the title
plt.title('Total STS per Episode for Three Algorithms', fontsize=14)

# Save the plot as a high-quality image
fig.savefig('Total STS per Episode for Three Algorithms(A-CvsQvsO)', dpi=300)

plt.show()
