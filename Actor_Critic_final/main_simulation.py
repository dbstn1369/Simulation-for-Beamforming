
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from ac_models import Actor, Critic, MemoryBuffer
from network_elements import AccessPoint, calculate_state_variables
import math
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


num_states = 3
num_actions = 2
actor_lr = 0.0001
critic_lr = 0.0005
discount_factor = 0.95


actor = Actor(num_states, num_actions).to(device)
critic = Critic(num_states).to(device)
actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)

memory_buffer_capacity = 100
batch_size = 32
memory_buffer = MemoryBuffer(memory_buffer_capacity)



#def choose_action(state, episode, epsilon_start=0.1, epsilon_end=0.01, epsilon_decay=2000):
def choose_action(state):
    #epsilon = epsilon_end + (epsilon_start - epsilon_end) * math.exp(-1.0 * episode / epsilon_decay)
    epsilon = 0.1
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    with torch.no_grad():
        action_probs = actor(state_tensor)
    action_probs = action_probs.cpu().squeeze().numpy()  # Flatten action_probs to 1D numpy array
    action_probs = np.nan_to_num(action_probs, nan=1/len(action_probs))  # Replace NaN values with a small value
    action_probs /= action_probs.sum()  # Normalize the probabilities

    if random.random() < epsilon:
        action = random.choice(range(len(action_probs)))  # Choose a random action with probability epsilon
    else:
        action = np.argmax(action_probs)  # Choose the action with the highest probability

    return action

def update_actor_critic_batch(batch, critic_optimizer, actor_optimizer):
    states, actions, rewards, next_states = zip(*batch)

    states_tensor = torch.FloatTensor(states).to(device)
    next_states_tensor = torch.FloatTensor(next_states).to(device)
    actions_tensor = torch.LongTensor(actions).to(device)
    rewards_tensor = torch.FloatTensor(rewards).to(device)

    values = critic(states_tensor)
    next_values = critic(next_states_tensor).detach()
    target_values = rewards_tensor + discount_factor * next_values

    critic_loss = nn.MSELoss()(values, target_values)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    action_probs = actor(states_tensor)
    selected_action_probs = action_probs.gather(1, actions_tensor.unsqueeze(1).unsqueeze(2)).squeeze(1).squeeze(1)
    advantages = (target_values - values).detach()

    actor_loss = -torch.log(selected_action_probs) * advantages
    actor_loss = torch.mean(actor_loss)
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
    print('action_probs shape:', action_probs.shape)
    print('actions_tensor shape:', actions_tensor.shape)


def get_reward(AP, successful_ssw_count, STS, bi, i):
    c1, c2, c3, c4 = 0.5, 0.5, 0.01, 0.01
    
    U = successful_ssw_count / (STS * len(AP.num_sector)) 
    T_max = 32*len(AP.num_sector)
    T_min = 1*len(AP.num_sector)
    T_m = ((STS*len(AP.num_sector)) - T_min) / (T_max - T_min) 
 
    STS, C_k, delta_u_norm, E = calculate_state_variables(AP.STS, AP, i)  # calculate_state_variables 함수 호출시 인자값 추가
    #print(f"-> {U, E, bi, T_m}")
     
    reward = 1 / (1 + math.exp(-((c1 * U + c2 * E) - (c3 * bi + c4 * T_m))))

    
    print(f"reward: {reward}")
    
    return reward



def get_new_state(AP):

    for i in range(len(AP.num_sector)):
        sts_count, Cog, delta_u_norm, E = calculate_state_variables(AP.STS, AP, i)  # calculate_state_variables 함수 호출시 인자값 추가
        AP.sector_states[i] = [sts_count, Cog, delta_u_norm]
    return AP.sector_states



with open('total_time.txt', 'a') as time_file, open('total_STS.txt', 'a') as sts_file, open('Reward.txt', 'a') as reward_file:
    for episode in range(2000):
        AP = AccessPoint(num_stations=300, STS=[32, 32, 32, 32, 32, 32])
        
        connected_stations = []
        total_time = 0
        successful_ssw_count = 0
        bi = 0
        previous_STS = 0
        start_time = time.time()
        s_time = start_time

        AP.start_beamforming_training()
        

        while not AP.all_stations_paired():

            connected_stations = [station for station in AP.stations if station.pair]

            for i in range(len(AP.num_sector)):
                AP.receive(i)
                 
            successful_ssw_count = len(AP.ssw_list)
            AP.broadcast_ack()

           
            states = get_new_state(AP)
            actions = []
            
            
            action = choose_action(states[i])
            actions.append(action)

            
            if action == 1:
                new_STS = max(16, AP.STS[i] - 1) 
                #new_STS = STS   
                print("STS: "+ str(new_STS)) 
            else:
                new_STS = min(32, AP.STS[i] + 1)  
                #new_STS = STS   
                print("STS: "+ str(new_STS)) 
   
            AP.update_STS(i, new_STS)

            AP.total_STS_used += new_STS * len(AP.num_sector)
            
            bi += 1
            print("BI: " + str(bi) )

            #if(previous_STS != STS):
            reward = get_reward(AP,successful_ssw_count, new_STS, bi, i)  # Pass the prev_STS variable
            reward_file.write(f"{reward:.3f}\n")
            next_state = get_new_state(AP)
            memory_buffer.push(states, action, reward, next_state)
            if len(memory_buffer) >= batch_size:
                batch = memory_buffer.sample(batch_size)
                update_actor_critic_batch(batch, critic_optimizer, actor_optimizer) 

            if not AP.all_stations_paired():
                print("Not all stations are paired. Starting next BI process.")
                successful_ssw_count = 0
                AP.next_bi()
               
                
            
        end_time = time.time()
        total_time = end_time - start_time
        print(f"Episode: {episode}")
        print(f"Total_STS_used: {AP.total_STS_used}")
            
        time_file.write(f"{total_time:.3f}\n")
        sts_file.write(str(AP.total_STS_used) + "\n")


    print("Number of GPUs: ", torch.cuda.device_count())
    print("GPU name: ", torch.cuda.get_device_name(0))
    print("Memory Used: ", torch.cuda.memory_allocated())
    print("Memory Reserved: ", torch.cuda.memory_reserved())
    torch.cuda.empty_cache()







