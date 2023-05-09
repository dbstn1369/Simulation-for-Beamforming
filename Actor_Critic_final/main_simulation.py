
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ac_models import Actor, Critic, MemoryBuffer
from network_elements import AccessPoint, calculate_state_variables
import math
import random


STS = 32
AP = AccessPoint(num_stations=150, STS=STS)

num_states = 3
num_actions = 3
actor_lr = 0.001
critic_lr = 0.005
discount_factor = 0.95


actor = Actor(num_states, num_actions)
critic = Critic(num_states)
actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)

memory_buffer_capacity = 1000
batch_size = 32
memory_buffer = MemoryBuffer(memory_buffer_capacity)




def choose_action(state, episode, epsilon_start=0.1, epsilon_end=0.01, epsilon_decay=1000):
#def choose_action(state):
    epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1.0 * episode / epsilon_decay)
    #epsilon = 0.1
    state_tensor = torch.FloatTensor(state).unsqueeze(0)
    with torch.no_grad():
        action_probs = actor(state_tensor)
    action_probs = action_probs.squeeze().numpy()  # Flatten action_probs to 1D numpy array
    action_probs = np.nan_to_num(action_probs, nan=1/len(action_probs))  # Replace NaN values with a small value
    action_probs /= action_probs.sum()  # Normalize the probabilities

    if random.random() < epsilon:
        action = random.choice(range(len(action_probs)))  # Choose a random action with probability epsilon
    else:
        action = np.argmax(action_probs)  # Choose the action with the highest probability

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
    c1, c2, c3, c4, c5 = 0.33, 0.33, 0.33, 0.4, 0.6
    U = successful_ssw_count / (STS * AP.num_sector) 
    T_m = 1 / (1+ math.exp(-(training_time)))

    STS, C_k, delta_u_norm, E = calculate_state_variables(AP.STS, STS, AP)  # calculate_state_variables 함수 호출시 인자값 추가
    
    reward = 1 / (1 + math.exp(-((c1 * U + c2 * delta_u_norm + c3 * E) - (c4 * C_k + c5 * T_m))))
    
    print(f"reward: {reward}")
    
    return reward



def get_new_state(AP, STS):

    sts_count, Cog, delta_u_norm, E = calculate_state_variables(AP.STS, STS, AP)  # calculate_state_variables 함수 호출시 인자값 추가

    return [sts_count, Cog, delta_u_norm]




total_STS_used = 0  # 누적된 STS 수를 저장할 변수 추가
with open('total_time.txt', 'a') as time_file, open('total_STS.txt', 'a') as sts_file, open('Reward.txt', 'a') as reward_file:
    for episode in range(1000):
        connected_stations = []
        total_time = 0
        training_time = 0
        successful_ssw_count = 0
        total_STS_used = 0  # 에피소드가 시작시 누적된 STS 값을 초기화
        start_time = time.time()
        s_time = start_time

        AP.reset_all_stations()
        AP.start_beamforming_training()


        while not AP.all_stations_paired():

            connected_stations = [station for station in AP.stations if station.pair]
            state = get_new_state(AP, STS)
            
            action = choose_action(state, episode)
            
            if action == 0:
                #STS = STS Original
                STS = max(1, STS - 1)
                print("STS: "+ str(STS))
            elif action == 1:
                #STS = STS Original
                STS = min(32, STS + 1)
                print("STS: "+ str(STS))
            elif action == 2:
                STS = STS
                print("STS: "+ str(STS))
        
            for i in range(AP.num_sector):
                AP.receive(i)
                 
            successful_ssw_count = len(AP.ssw_list)
            AP.broadcast_ack()

            if not AP.all_stations_paired():
                print("Not all stations are paired. Starting next BI process.")
                
                
                f_time = time.time()  # 시간을 할당하는 부분 추가
                time_difference = f_time - s_time
                s_time += time_difference
                
                print(f"Time spent in this BI: {time_difference:.3f} seconds")  # Add this line to print the time for each BI
                reward = get_reward(AP,successful_ssw_count, STS, time_difference)  # Pass the prev_STS variable
                reward_file.write(f"{reward:.3f}\n")
                next_state = get_new_state(AP, STS)
                memory_buffer.push(state, action, reward, next_state)
                successful_ssw_count = 0
                total_STS_used += STS*AP.num_sector  # 누적 STS 값 업데이트
            

                if len(memory_buffer) >= batch_size:
                    batch = memory_buffer.sample(batch_size)
                    update_actor_critic_batch(batch)
                
                print(f"Current_STS_: {STS-1}")
                AP.next_bi()

        end_time = time.time()
        total_time = end_time - start_time
        print(f"Episode: {episode}")
        print(f"Total_STS_used: {total_STS_used}")
            
        time_file.write(f"{total_time:.3f}\n")
        sts_file.write(str(total_STS_used) + "\n")







