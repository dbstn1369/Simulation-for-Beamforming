
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ac_models import Actor, Critic, MemoryBuffer
from network_elements import AccessPoint, calculate_state_variables
import math


STS = 15
AP = AccessPoint(num_stations=100, STS=STS)

num_states = 3
num_actions = 3
actor_lr = 0.001
critic_lr = 0.001
discount_factor = 0.99


actor = Actor(num_states, num_actions)
critic = Critic(num_states)
actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)

memory_buffer_capacity = 1000
batch_size = 32
memory_buffer = MemoryBuffer(memory_buffer_capacity)


def choose_action(state):
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        action_probs = actor(state_tensor)
    action_probs = action_probs.squeeze().numpy()  # Flatten action_probs to 1D numpy array
    action_probs = np.nan_to_num(action_probs, nan=1/len(action_probs))  # Replace NaN values with a small value
    action_probs /= action_probs.sum()  # Normalize the probabilities
    action = np.random.choice(len(action_probs), p=action_probs)
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

    action_probs = actor(states_tensor).squeeze(1)
    selected_action_probs = action_probs.gather(1, actions_tensor.unsqueeze(1)).squeeze()
    advantages = (target_values - values).detach()

    actor_loss = -torch.log(selected_action_probs) * advantages
    actor_loss = torch.mean(actor_loss)  # Add this line to calculate the mean of the actor loss
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
    
def get_reward(AP, successful_ssw_count, STS, training_time):
    c1, c2, c3, c4 = 0.1, 0.1, 0.7, 0.1  
    U = successful_ssw_count / (STS*AP.num_sector) 

    STS, C_k, delta_u_norm = calculate_state_variables(AP.STS, STS, AP)  # calculate_state_variables 함수 호출시 인자값 추가

    reward = 1 / (1 + math.exp(-((c1 * U + c2 * delta_u_norm) / c3*(C_k) - c4 * training_time)))
    print(f"reward: {reward}")
    
    return reward

def get_new_state(AP, STS):

    sts_count, Cog, delta_u_norm = calculate_state_variables(AP.STS, STS, AP)  # calculate_state_variables 함수 호출시 인자값 추가

    return [sts_count, Cog, delta_u_norm]



total_STS_used = 0  # 누적된 STS 수를 저장할 변수 추가
for episode in range(1000):
    connected_stations = []
    total_time = 0
    successful_ssw_count = 0
    total_STS_used = 0  # 에피소드가 시작시 누적된 STS 값을 초기화
    start_time = time.time()
    s_time = time.time()

    AP.reset_all_stations()
    AP.start_beamforming_training()


    while not AP.all_stations_paired():

        connected_stations = [station for station in AP.stations if station.pair]
        state = get_new_state(AP, STS)
        

        action = choose_action(state)
        if action == 0:
            STS = min(32, STS + 1)  # STS 개수를 최대 32개로 제한
            print("STS: "+ str(STS))
        elif action == 1:
            STS = max(1, STS - 1)
            print("STS: "+ str(STS))
    
        for i in range(AP.num_sector):
            successful_ssw_count_sector = AP.receive(i)
            successful_ssw_count += successful_ssw_count_sector

        AP.broadcast_ack()

        if not AP.all_stations_paired():
            print("Not all stations are paired. Starting next BI process.")
            
            
            f_time = time.time()  # 시간을 할당하는 부분 추가
            time_difference = f_time - s_time
            reward = get_reward(AP,successful_ssw_count, STS, time_difference)  # Pass the prev_STS variable
            next_state = get_new_state(AP, STS)
            memory_buffer.push(state, action, reward, next_state)
            successful_ssw_count = 0
            total_STS_used += STS*AP.num_sector  # 누적 STS 값 업데이트
            total_time += time_difference
            s_time = time.time()

            if len(memory_buffer) >= batch_size:
                batch = memory_buffer.sample(batch_size)
                update_actor_critic_batch(batch)
            
            print(f"Current_STS_: {STS-1}")
            print(f"Total_STS_used: {total_STS_used}")
            print(f"Episode: {episode}")
            AP.next_bi()


    end_time = time.time()  # 시뮬레이션 종료 시간 측정
    total_time = end_time - start_time    

    with open('total_time.txt', 'a') as f:
        f.write(f"{total_time:.3f}\n")  # 누적된 STS 값을 함께 저장
    with open('total_STS.txt', 'a') as f:
        f.write(str(total_STS_used)+"\n")  # 누적된 STS 값을 함께 저장

    print("EPISODE: " + str(episode) + " All stations are paired. Simulation complete.")
    print(f"Total simulation time: {total_time:.3f} seconds")
    # Reset all station pairs before starting the next episode






