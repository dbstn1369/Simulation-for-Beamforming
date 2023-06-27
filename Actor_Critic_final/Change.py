import numpy as np

# 파일에서 데이터 읽기
total_sts = np.genfromtxt('total_STS_AC.txt')


total_times = total_sts * 100

# 변환된 시간 데이터를 파일에 저장
np.savetxt('total_time_AC.txt', total_times)