import numpy as np

# 파일에서 데이터 읽기
total_sts = np.genfromtxt('total_STS_O.txt')

# STS를 시간으로 변환 (1 STS = 10 마이크로 초)
total_times = total_sts * 10

# 변환된 시간 데이터를 파일에 저장
np.savetxt('total_time_O.txt', total_times)