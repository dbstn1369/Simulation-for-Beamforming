import random
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
from ac_models import Actor, Critic
from network_elements import AccessPoint

STS = 32

num_states = 2
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
    action = torch.multinomial(action_probs, 1).item()
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

def get_reward(successful_ssw_count, total_ssw_count):
    return successful_ssw_count / total_ssw_count

# 충돌이 발생한 프레임의 SINR 값을 상태에 포함시키기 위한 새로운 상태 함수를 정의합니다.
def get_new_state(connected_stations, sinr_values):
    sinr_values_filtered = [x for x in sinr_values if not np.isnan(x)]  # NaN 값을 제외한 리스트 생성
    sinr_mean = np.mean(sinr_values_filtered) if sinr_values_filtered else 0  # 리스트가 비어있을 경우 0을 할당
    state = np.array([connected_stations, sinr_mean], dtype=np.int32)
    return state


def SINR(received_signal, interfering_signals, noise_power=1e-9):
    interference = sum(interfering_signals)
    return received_signal / (interference + noise_power)



total_STS_used = 0  # 누적된 STS 수를 저장할 변수 추가

for episode in range(100):
    AP = AccessPoint(num_stations=5, STS=STS)
    connected_stations = 0
    total_time = 0
    total_STS_used = 0  # 에피소드가 시작시 누적된 STS 값을 초기화
    start_time = time.time()
    s_time = time.time()
    AP.start_beamforming_training()

    while not AP.all_stations_paired():

        sinr_values = []
        connected_stations = sum(station.pair for station in AP.stations)
        state = get_new_state(connected_stations, sinr_values)
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
            reward = get_reward(successful_ssw_count, STS)
            connected_stations = sum(station.pair for station in AP.stations)
            next_state = get_new_state(connected_stations, sinr_values)

            update_actor_critic(state, action, reward, next_state)  # 변경된 부분

            AP.next_bi()

    end_time = time.time()  # 시뮬레이션 종료 시간 측정
    total_time = end_time - start_time
    print("EPISODE: " + str(episode) + " All stations are paired. Simulation complete.")
    print(f"Total simulation time: {total_time:.3f} seconds")
    # 결과를 파일에 저장
    with open('beamforming_simulation_results_with_AC.txt', 'a') as f:
        f.write(f"{total_time:.3f}, {total_STS_used}\n")  # 누적된 STS 값을 함께 저장


