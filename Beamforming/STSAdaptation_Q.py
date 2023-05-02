import random
import numpy as np
import time


num_states = 50  # 연결된 STA 수 (0-10)
num_actions = 3  # 증가, 감소, 그대로 유지
q_table = np.zeros((num_states, num_actions))

learning_rate = 0.1
discount_factor = 0.99
exploration_rate = 1.0
exploration_rate_decay = 0.001


def choose_action(state):
    if np.random.uniform(0, 1) < exploration_rate:
        return np.random.randint(num_actions)
    else:
        return np.argmax(q_table[state])

def update_q_table(state, action, reward, next_state):
    q_table[state, action] = q_table[state, action] + learning_rate * (
            reward + discount_factor * np.max(q_table[next_state]) - q_table[state, action])

def get_reward(time_difference):
    return -time_difference


def SNR():
    # 신호 레벨 범위 (dBm 단위)
    min_signal_level = -80
    max_signal_level = -40

    # 무작위 신호 레벨 개수
    num_signal_levels = 5

    # 무작위 신호 레벨 생성
    random_signal_levels = np.random.uniform(min_signal_level, max_signal_level, num_signal_levels)
    print("Random signal levels (dBm):", random_signal_levels)
    return random_signal_levels

STS = 15


class AccessPoint:
    def __init__(self, num_stations):
        self.num_stations = num_stations
        self.num_sector = 6
        self.stations = [Station(i) for i in range(num_stations)]

    def start_beamforming_training(self):

        beacon_frame = self.create_beacon_frame_with_trn_r()
        for station in self.stations:
            station.receive_bti(beacon_frame)
            station.receive_trn_r(beacon_frame)

    def create_beacon_frame_with_trn_r(self):
        return {'SNR': SNR(), 'trn_r': 'TRN-R data'}

    def recieve(self, sector):
        print(f"sector{sector} of AP")
        for i in range(STS):
            for station in self.stations:
                if not station.pair and sector == station.tx_sector_AP:  # 이미 연결된 STA들이 참여하지 않도록 조건 추가
                    print(f"STS: {i}")
                    station.send_ssw(i, sector)

    def broadcast_ack(self):
        ack_frame = "ACK_DATA"
        for station in self.stations:
            if station.pair == False:
                station.receive_ack_frame(ack_frame)

    def next_bi(self):
        self.start_beamforming_training()
        for i in range(self.num_sector):
            self.recieve(i)
        self.broadcast_ack()

    def all_stations_paired(self):
        return all(station.pair for station in self.stations)

class Station:
    def __init__(self, id):
        self.id = id
        self.pair = False
        self.tx_sector_AP = None
        self.rx_sector = None
        self.collisions = 0
        self.data_success = False
        self.sectors = [i for i in range(1, 4)]
        self.backoff_count = random.randint(1, STS)

    def receive_bti(self, beacon_frame):
        self.tx_sector_AP = self.get_best_sectors(beacon_frame)
        print(f"Station {self.id}: Best TX sector of AP - {self.tx_sector_AP}")

    def get_best_sectors(self, beacon_frame):
        snr_values = beacon_frame['SNR']
        best_sector = np.argmax(snr_values) + 1
        return best_sector

    def receive_trn_r(self, beacon_frame):
        best_rx_sector = self.get_best_rx_sector(beacon_frame)
        print(f"Station {self.id}: Best RX sector of STA after TRN-R - {best_rx_sector}")

    def get_best_rx_sector(self, beacon_frame):
        snr_values = SNR()
        best_rx_sector = np.argmax(snr_values) % len(self.sectors) + 1
        return best_rx_sector

    def send_ssw(self, STS, sector):
        if not self.pair and STS == self.backoff_count:  # 이미 연결된 STA들이 참여하지 않도록 조건 추가
            self.rx_sector = None
            self.data_success = True
            print("Station " + str(self.id) + " transmitted SSW frame successfully")
        return

    def receive_ack_frame(self, ack_frame):
        if self.pair == False:
            if self.data_success == True:
                self.pair = True
                print("Station " + str(self.id) + " received ACK successfully")
        else:
            print("Station " + str(self.id) + " did not receive ACK, will retry in the next BI")

total_STS_used = 0  # 누적된 STS 수를 저장할 변수 추가

for episode in range(1000):
    AP = AccessPoint(num_stations=10)
    connected_stations = 0
    total_time = 0
    total_STS_used = 0  # 에피소드가 시작시 누적된 STS 값을 초기화
    start_time = time.time()
    s_time = time.time()
    AP.start_beamforming_training()

    while not AP.all_stations_paired():
        state = connected_stations
        total_STS_used += STS  # 누적 STS 값 업데이트

        action = choose_action(state)
        if action == 0:
            STS = min(32, STS + 1)  # STS 개수를 최대 32개로 제한
        elif action == 1:
            STS = max(1, STS - 1)

        for i in range(AP.num_sector):
            AP.recieve(i)

        AP.broadcast_ack()

        if not AP.all_stations_paired():
            print("Not all stations are paired. Starting next BI process.")

            f_time = time.time()  # 시간을 할당하는 부분 추가
            time_difference = f_time - s_time
            s_time += time_difference
            reward = get_reward(time_difference)
            connected_stations = sum(station.pair for station in AP.stations)
            next_state = connected_stations

            update_q_table(state, action, reward, next_state)

            AP.next_bi()

    end_time = time.time()  # 시뮬레이션 종료 시간 측정
    total_time = end_time - start_time
    print("EPISODE: " + str(episode) + " All stations are paired. Simulation complete.")
    print(f"Total simulation time: {total_time:.3f} seconds")
    # 결과를 파일에 저장
    with open('beamforming_simulation_results_with_Q.txt', 'a') as f:
        f.write(f"{total_time:.3f}, {total_STS_used}\n")  # 누적된 STS 값을 함께 저장

    exploration_rate = max(0.01, exploration_rate - exploration_rate_decay)
