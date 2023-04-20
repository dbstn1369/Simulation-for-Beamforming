import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ac_models import Actor, Critic
from network_elements import AccessPoint, calculate_state_variables


STS = 32
AP = AccessPoint(num_stations=5, STS=STS)

num_states = 3
num_actions = 3
actor_lr = 0.001
critic_lr = 0.001
discount_factor = 0.99

actor = Actor(num_states, num_actions)
critic = Critic(num_states)
actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)


def choose_action(state):
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        action_probs = actor(state_tensor)
    print(f"Action probabilities: {action_probs}")  # Add this line to check action probabilities
    action = torch.multinomial(action_probs + 1e-8, 1).item()  # Add a small constant value to action_probs
    return action

def update_actor_critic(state, action, reward, next_state):
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
    action_tensor = torch.LongTensor([action])

    value = critic(state_tensor)
    next_value = critic(next_state_tensor).detach()
    target_value = reward + discount_factor * next_value

    critic_loss = nn.MSELoss()(value, target_value)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    action_prob = actor(state_tensor)
    selected_action_prob = action_prob.gather(1, action_tensor.unsqueeze(1)).squeeze()
    advantage = (target_value - value).detach()

    actor_loss = -torch.log(selected_action_prob) * advantage
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
    
def get_reward(AP, successful_ssw_count, STS, prev_STS):
    c1, c2 = 0.5, 0.5  # c1, c2는 각각 0.5로 설정
    U = successful_ssw_count / STS
    if prev_STS != 0:
        delta_U_norm = (STS - prev_STS) / prev_STS
    else:
        delta_U_norm = 0
    C = np.mean([ssw.snr for station in AP.stations for ssw in AP.ssw_list])  
    if prev_STS != 0:  # Add this condition to handle the case when prev_STS is 0
        C_norm = (C - np.mean([AP.ssw.snr for sts in AP.ssw_list])) / np.std([AP.ssw.snr for sts in AP.ssw_list])
    else:
        C_norm = 0  # Set C_norm to a default value when prev_STS is 0
    C_k = 1 - C_norm
    reward = (c1 * U + c2 * delta_U_norm) / (1 + C_k)
    return reward

def get_new_state(AP, STS):

    sts_count, Cog, delta_u_norm = calculate_state_variables(AP.STS, AP.BI, STS, AP)  # calculate_state_variables 함수 호출시 인자값 추가

    return [sts_count, Cog, delta_u_norm]





total_STS_used = 0  # 누적된 STS 수를 저장할 변수 추가
prev_STS = 0
for episode in range(100):
    connected_stations = []
    total_time = 0
    total_STS_used = 0  # 에피소드가 시작시 누적된 STS 값을 초기화
    start_time = time.time()
    s_time = time.time()
    AP.start_beamforming_training()


    while not AP.all_stations_paired():

        sinr_values = []
        connected_stations = [station for station in AP.stations if station.pair]
        state = get_new_state(AP, STS)
        total_STS_used += STS  # 누적 STS 값 업데이트

        action = choose_action(state)
        if action == 0:
            STS = min(32, STS + 1)  # STS 개수를 최대 32개로 제한
        elif action == 1:
            STS = max(1, STS - 1)

        successful_ssw_count = 0
        sinr_values = []
        for i in range(AP.num_sector):
            sinr_values_sector, successful_ssw_count_sector = AP.recieve(i)
            if sinr_values_sector:  # Check if the list is not empty
                sinr_values.extend(sinr_values_sector)
            successful_ssw_count += successful_ssw_count_sector

        AP.broadcast_ack()

        if not AP.all_stations_paired():
            print("Not all stations are paired. Starting next BI process.")

            f_time = time.time()  # 시간을 할당하는 부분 추가
            time_difference = f_time - s_time
            s_time += time_difference
            reward = get_reward(AP,successful_ssw_count, STS, prev_STS)  # Pass the prev_STS variable
            next_state = get_new_state(AP, STS)

            update_actor_critic(state, action, reward, next_state)  # 변경된 부분

            AP.next_bi()

    end_time = time.time()  # 시뮬레이션 종료 시간 측정
    total_time = end_time - start_time
    print("EPISODE: " + str(episode) + " All stations are paired. Simulation complete.")
    print(f"Total simulation time: {total_time:.3f} seconds")
    # 결과를 파일에 저장
    with open('beamforming_simulation_results_with_AC.txt', 'a') as f:
        f.write(f"{total_time:.3f}, {total_STS_used}\n")  # 누적된 STS 값을 함께 저장


import random
import numpy as np
import random
import numpy as np
import torch
import torch.nn as nn



def calculate_state_variables(STS, BI, s_sts, AP):
    # Calculate current STS count
    if isinstance(STS, int):
        s_sts = STS
    else:
        s_sts = int(STS)

    # Calculate propagation delay
    propagation_delay = []
    for sts in range(s_sts):
        STA_positions = np.linspace(AP.min_distance, AP.max_distance)
        delay = np.linalg.norm(STA_positions - 0) / 3e8  # Speed of light
        propagation_delay.append(delay)
    s_pd = np.mean(propagation_delay)
    S_min = AP.min_distance / 3e8
    S_max = AP.max_distance / 3e8
    S_norm = (s_pd - S_min) / (S_max - S_min)
    W = 1 - S_norm

    # Calculate congestion
    weighted_snr_sum = 0
    weight_sum = 0
    for ssw in AP.ssw_list:
        weighted_snr_sum += W * ssw.snr
        weight_sum += W
    if weight_sum != 0:
        s_c = weighted_snr_sum / weight_sum
        C_norm = (s_c - np.mean([ssw.snr for ssw in AP.ssw_list])) / np.std([ssw.snr for ssw in AP.ssw_list])
        C_k = 1 - C_norm

    # Calculate STS usage
    N_ack = sum([1 for station in AP.stations if station.data_success])
    U_current = N_ack / STS
    U_previous = s_sts
    delta_U = U_current - U_previous
    delta_U_min = 0
    delta_U_max = 32
    delta_U_norm = (delta_U - delta_U_min) / (delta_U_max - delta_U_min)

    return s_sts, W, C_k, delta_U_norm


def SINR(received_signal, interfering_signals, noise_power=1e-9):
    interference = sum(interfering_signals)
    return received_signal / (interference + noise_power)

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

class AccessPoint:
    def __init__(self, num_stations, STS, min_distance=0, max_distance=100):  # min_distance와 max_distance 매개변수 추가
        self.num_stations = num_stations
        self.STS = STS
        self.num_sector = 6
        self.ssw_list = []
        self.min_distance = min_distance  # 최소 거리
        self.max_distance = max_distance  # 최대 거리
        station_positions = np.linspace(self.min_distance, self.max_distance, num_stations)  # Assign positions to stations
        self.BI = Station(random.randint(1, num_stations), STS, position=random.choice(station_positions))  # Initialize BI with a random station instance
        self.stations = [Station(i, STS, position=station_positions[i]) for i in range(num_stations)]  # Assign positions to stations
        

    def start_beamforming_training(self):
        self.sinr_values = []

        beacon_frame = self.create_beacon_frame_with_trn_r()
        for station in self.stations:
            station.receive_bti(beacon_frame)
            station.receive_trn_r(beacon_frame)

    def create_beacon_frame_with_trn_r(self):
        return {'SNR': SNR(), 'trn_r': 'TRN-R data'}

    def recieve(self, sector):
        successful_ssw_count = 0
        received_signals = []
        if sector >= len(self.sinr_values):
            return [], successful_ssw_count
        for i in range(self.STS):
            for station in self.stations:
                if not station.pair and sector == station.tx_sector_AP:
                    signal = station.send_ssw(i, sector)
                    if signal is not None:
                        received_signals.append(signal)
                    if station.data_success:
                        successful_ssw_count += 1
            if len(received_signals) > 1:
                sinr_values = [SINR(signal, [s for s in received_signals if s != signal]) for signal in received_signals]
                self.sinr_values[sector].extend(sinr_values)
        if not self.sinr_values[sector]:
            return [], successful_ssw_count
        else:
            return self.sinr_values[sector], successful_ssw_count

    def broadcast_ack(self):
        ack_frame = "ACK_DATA"
        for station in self.stations:
            if station.pair == False:
                station.receive_ack_frame(ack_frame)

    def next_bi(self):
        self.start_beamforming_training()
        for i in range(self.num_sector):
            successful_ssw_count, sinr_values = self.recieve(i)  # sinr_values 반환값 추가
        self.broadcast_ack()

    def all_stations_paired(self):
        return all(station.pair for station in self.stations)

import random
class Station:
    def __init__(self, id, STS, position=None):  # Add position parameter with default value None
        self.id = id
        self.pair = False
        self.tx_sector_AP = None
        self.rx_sector = None
        self.collisions = 0
        self.data_success = False
        self.sectors = [i for i in range(1, 5)]
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
        self.rx_sector = best_rx_sector  # rx_sector에 할당하는 부분 추가
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
            return random.uniform(0.0001, 0.001)  # 임의의 수신 신호 전송
        return None

    def receive_ack_frame(self, ack_frame):
        if self.pair == False:
            if self.data_success == True:
                self.pair = True
                print("Station " + str(self.id) + " received ACK successfully")
        else:
            print("Station " + str(self.id) + " did not receive ACK, will retry in the next BI")
