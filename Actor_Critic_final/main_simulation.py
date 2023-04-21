import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ac_models import Actor, Critic, MemoryBuffer
from network_elements import AccessPoint, calculate_state_variables


STS = 5
AP = AccessPoint(num_stations=3, STS=STS)

num_states = 3
num_actions = 3
actor_lr = 0.001
critic_lr = 0.001
discount_factor = 0.99



actor = Actor(num_states, num_actions)
critic = Critic(num_states)
actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)

memory_buffer_capacity = 10000
batch_size = 32
memory_buffer = MemoryBuffer(memory_buffer_capacity)


def choose_action(state):
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        action_probs = actor(state_tensor)
    print(f"Action probabilities: {action_probs}")  # Add this line to check action probabilities
    action = torch.multinomial(action_probs + 1e-8, 1).item()  # Add a small constant value to action_probs
    return action


def update_actor_critic_batch(batch):
    states, actions, rewards, next_states = zip(*batch)

    states_tensor = torch.FloatTensor(states)
    next_states_tensor = torch.FloatTensor(next_states)
    actions_tensor = torch.LongTensor(actions)
    rewards_tensor = torch.FloatTensor(rewards)

    values = critic(states_tensor)
    next_values = critic(next_states_tensor).detach()
    target_values = rewards_tensor + discount_factor * next_values

    critic_loss = nn.MSELoss()(values, target_values)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    action_probs = actor(states_tensor)
    selected_action_probs = action_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze()
    advantages = (target_values - values).detach()

    actor_loss = -torch.log(selected_action_probs) * advantages
    actor_loss = torch.mean(actor_loss)  # Add this line to calculate the mean of the actor loss
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

    sts_count, Cog, delta_u_norm = calculate_state_variables(AP.STS, STS, AP)  # calculate_state_variables 함수 호출시 인자값 추가

    return [sts_count, Cog, delta_u_norm]



total_STS_used = 0  # 누적된 STS 수를 저장할 변수 추가
prev_STS = 0
for episode in range(1):
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
            sinr_values_sector, successful_ssw_count_sector = AP.recieve(i+1)
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
            memory_buffer.push(state, action, reward, next_state)
           

            if len(memory_buffer) >= batch_size:
                batch = memory_buffer.sample(batch_size)
                update_actor_critic_batch(batch)

            AP.next_bi()

    end_time = time.time()  # 시뮬레이션 종료 시간 측정
    total_time = end_time - start_time
    print("EPISODE: " + str(episode) + " All stations are paired. Simulation complete.")
    print(f"Total simulation time: {total_time:.3f} seconds")
    # 결과를 파일에 저장
    with open('beamforming_simulation_results_with_AC.txt', 'a') as f:
        f.write(f"{total_time:.3f}, {total_STS_used}\n")  # 누적된 STS 값을 함께 저장


