
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ac_models import Actor, Critic, MemoryBuffer
from network_elements import AccessPoint, calculate_state_variables
import math
import random
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


num_states = 3
num_actions = 31
actor_lr = 0.001
critic_lr = 0.005
discount_factor = 0.95


actor = Actor(num_states, num_actions).to(device)
critic = Critic(num_states).to(device)
actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)

memory_buffer_capacity = 100
batch_size = 32
memory_buffer = MemoryBuffer(memory_buffer_capacity)



def choose_action(state, successful_ssw_count, bi, i, episode, epsilon_start=0.1, epsilon_end=0.01, epsilon_decay=1000):
    epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1.0 * episode / epsilon_decay)
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

    with torch.no_grad():
        action_probs = actor(state_tensor)

    action_probs = action_probs.cpu().squeeze().numpy()  
    action_probs = np.nan_to_num(action_probs, nan=1/len(action_probs))  
    action_probs /= action_probs.sum()

    reward_list = [get_reward(AP, successful_ssw_count, action+2, bi, i) for action in range(num_actions)]
    high_reward_actions = [idx for idx, reward in enumerate(reward_list) if reward >= 0.5]

    if not high_reward_actions or random.random() < epsilon:
        action = random.choice(range(2, 33))  
    else:
        action = random.choices(high_reward_actions, weights=np.array(reward_list)[high_reward_actions])[0] + 2

    return action


def update_actor_critic_batch(batch, critic_optimizer, actor_optimizer):
    states, actions, rewards, next_states = zip(*batch)

    states_tensor = torch.FloatTensor(states).to(device)
    next_states_tensor = torch.FloatTensor(next_states).to(device)
    actions_tensor = torch.LongTensor(actions).unsqueeze(-1).to(device)

    actions_tensor = actions_tensor.unsqueeze(1).expand(-1, 1, num_actions)  # 크기를 (batch_size, 1, num_actions)로 확장
    rewards_tensor = torch.FloatTensor(rewards).unsqueeze(-1).to(device)
    

    values = critic(states_tensor)
    next_values = critic(next_states_tensor).detach()
    
    rewards_tensor_expanded = rewards_tensor.unsqueeze(1).expand(-1, 1, next_values.size(-1))

    #print('rewards_tensor_expanded shape:', rewards_tensor_expanded.shape)

    target_values = rewards_tensor_expanded + discount_factor * next_values

    critic_loss = nn.MSELoss()(values, target_values)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    action_probs = actor(states_tensor)

    #print('action_probs shape:', action_probs.shape)
    #print('actions_tensor shape:', actions_tensor.shape)


    selected_action_probs = action_probs.gather(2, actions_tensor).squeeze(2)


    advantages = (target_values - values).detach()

    actor_loss = -torch.log(selected_action_probs) * advantages
    actor_loss = torch.mean(actor_loss)
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()


# def update_actor_critic_batch(batch, critic_optimizer, actor_optimizer):
#     return


def get_reward(AP, successful_ssw_count, STS, bi, i):
    c1, c2, c3, c4 = 1, 1, 0.1, 1

    U = successful_ssw_count / (STS) 
    T_max = 32
    T_min = 1
    T_m = ((STS) - T_min) / (T_max - T_min) 
 
    STS, C_k, delta_u_norm, E = calculate_state_variables(AP.STS, AP, i)  
    #print(f"-> {U, E, bi, T_m}")

    reward = 1 / (1 + math.exp(-((c1 * U + c2 * E) - (c3 * bi + c4 * T_m))))
    #print(f"reward: {reward}")
    
    return reward



def get_new_state(AP):

    for i in range(len(AP.num_sector)):
        sts_count, Cog, delta_u_norm, E = calculate_state_variables(AP.STS, AP, i)  # calculate_state_variables 함수 호출시 인자값 추가
        AP.sector_states[i] = [sts_count, Cog, delta_u_norm]
    return AP.sector_states



with open('total_time_O.txt', 'a') as time_file, open('total_STS_O.txt', 'a') as sts_file, open('Reward_O.txt', 'a') as reward_file:
    for episode in range(10000):
       
        STS = [32] * 16
        AP = AccessPoint(num_stations=500, STS=STS)
        
        connected_stations = []
        total_time = 0
        learning_time = 0
        successful_ssw_count = 0
        bi = 0
        start_time = time.time()
        s_time = start_time

        AP.start_beamforming_training()
        

        while not AP.all_stations_paired():

            connected_stations = [station for station in AP.stations if station.pair]
            

            for i in range(len(AP.num_sector)):
                AP.receive(i)
                 
                successful_ssw_count = len(AP.ssw_list)
                AP.broadcast_ack()

                learning_start = time.time()

                states = get_new_state(AP)
                
                  
                new_STS = choose_action(states[i], successful_ssw_count, bi, i, episode)


                #new_STS = 32 

                #print(f"Sector: {i}")  
                #print("STS: "+ str(new_STS)) 

                AP.update_STS(i, new_STS)

                

                AP.total_STS_used += new_STS

                
                reward = get_reward(AP,successful_ssw_count, new_STS, bi, i)  # Pass the prev_STS variable
                reward_file.write(f"{reward:.3f}\n")
                successful_ssw_count = 0
                next_state = get_new_state(AP)
                memory_buffer.push(states, new_STS-2, reward, next_state)
                
                if len(memory_buffer) >= batch_size:
                    batch = memory_buffer.sample(batch_size)
                    update_actor_critic_batch(batch, critic_optimizer, actor_optimizer)
                    
                learning_end = time.time()  # End timing learning
                learning_time += learning_end - learning_start  # Increment total learning time 

            if not AP.all_stations_paired():
                #print("Not all stations are paired. Starting next BI process.")
                AP.next_bi()
                bi += 1
                #print("BI: " + str(bi) )
            
            
        end_time = time.time()
        total_time = end_time - start_time - learning_time  # Subtract the learning time from the total time
        print(f"Episode: {episode}")
        print(f"Total_STS_used: {AP.total_STS_used}")
            
        time_file.write(f"{total_time:.3f}\n")
        sts_file.write(str(AP.total_STS_used) + "\n")


    print("Number of GPUs: ", torch.cuda.device_count())
    print("GPU name: ", torch.cuda.get_device_name(0))
    print("Memory Used: ", torch.cuda.memory_allocated())
    print("Memory Reserved: ", torch.cuda.memory_reserved())
    torch.cuda.empty_cache()







