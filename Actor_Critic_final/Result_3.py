import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# Set the font family to a specific font, e.g., 'Arial'
mpl.rcParams['font.family'] = 'Times New Roman'

# 텍스트 파일에서 데이터 읽기
total_times_algo1 = np.genfromtxt('total_time_AC.txt')
total_times_algo2 = np.genfromtxt('total_time_Q.txt')
total_times_algo3 = np.genfromtxt('total_time_O.txt')

# convert microseconds to milliseconds
total_times_algo1 = total_times_algo1 / 1000
total_times_algo2 = total_times_algo2 / 1000
total_times_algo3 = total_times_algo3 / 1000

# 에피소드 번호 생성 (1부터 시작)
episode_numbers = np.arange(1, len(total_times_algo1) + 1)

bin_size = 100
total_times_avg_algo1 = np.mean(total_times_algo1.reshape(-1, bin_size), axis=1)
total_times_avg_algo2 = np.mean(total_times_algo2.reshape(-1, bin_size), axis=1)
total_times_avg_algo3 = np.mean(total_times_algo3.reshape(-1, bin_size), axis=1)

episode_numbers_avg = np.arange(1, len(total_times_avg_algo1) + 1) * bin_size

# Set the plot style to 'classic' for a more suitable style for IEEE papers
plt.style.use('classic')

# Create the plot
fig, ax1 = plt.subplots()

# Plot the time data on the first y-axis
ax1.plot(episode_numbers_avg, total_times_avg_algo1, label='Time A-C', color='#ff5733', marker='s')
ax1.plot(episode_numbers_avg, total_times_avg_algo2, label='Time Q', color='#33ff57', marker='^')
ax1.plot(episode_numbers_avg, total_times_avg_algo3, label='Time O', color='#3357ff', marker='o')
ax1.set_xlabel('Episode Number', fontsize=12)
ax1.set_ylabel('Time (ms)', fontsize=12)



# Set the grid
ax1.grid(True)

# Customize the tick labels size
ax1.tick_params(axis='both', which='major', labelsize=10)

# Combine the legends of both y-axes
lines1, labels1 = ax1.get_legend_handles_labels()
ax1.legend(lines1, labels1, fontsize=10)

# Set the title
#plt.title('Time per Episode for Three Algorithms', fontsize=14)

# Save the plot as a high-quality image
fig.savefig('Time per Episode for Three Algorithms(A-CvsQvsO)', dpi=300)

plt.show()
