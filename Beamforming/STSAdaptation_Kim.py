import random
import numpy as np
import time
import torch
import os

os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# STS와 SINR 범주 정의
STS_thresholds = [4, 8, 12, 16, 20, 24, 28, 32]  # 8 categories
SINR_thresholds = [-5, 0, 5, 30]  # 4 categories

# 매핑 테이블 정의
mapping_table = {}
counter = 0
for sts_state in range(len(STS_thresholds)):
    for sinr_state in range(len(SINR_thresholds)):
        mapping_table[(sts_state, sinr_state)] = counter
        counter += 1

num_actions = 3  # Increase, decrease, maintain
num_sectors = 16  # Number of sectors

# Update the q_table tensor to have 2 dimensions
q_table = torch.zeros((32, num_actions), dtype=torch.float32, device=device)

learning_rate = 0.1
discount_factor = 0.95
epsilon = 0.1




def choose_action(state, AP_STS):
    if np.random.uniform(0, 1) < epsilon:
        action = np.random.choice([0, 1, 2])
    else:
        q_values = q_table[state, :].cpu().numpy()  # Convert CUDA tensor to CPU tensor and then to NumPy array
        action = np.argmax(q_values)
    return action



def update_q_table(state, action, reward, next_state):
    state = torch.as_tensor(state, dtype=torch.int64, device=device)
    next_state = torch.as_tensor(next_state, dtype=torch.int64, device=device)

    reward = torch.as_tensor(reward, dtype=torch.float32, device=device)
    action = torch.as_tensor(action, dtype=torch.int64, device=device)

    q_value = q_table[state, action]
    max_next_q_value = torch.max(q_table[next_state, :])
    target_q_value = reward + discount_factor * max_next_q_value
    q_table[state, action] = q_value + learning_rate * (target_q_value - q_value)


def get_reward(successful_ssw_count, total_ssw_count):
    if total_ssw_count == 0:
        return 0
    return successful_ssw_count / total_ssw_count


def get_state(STS, SINR):
    STS_state = sum(STS > np.array(STS_thresholds)) - 1
    if STS_state < 0:
        STS_state = 0  # Assign a default state value if STS_state is less than 0
    SINR_state = sum(SINR > np.array(SINR_thresholds)) - 1

    # Retrieve the scalar state value from the mapping table
    state = mapping_table.get((STS_state, SINR_state))
    if state is None:
        state = 0  # Assign a default state value if the mapping is not found
    return state



def get_new_state(connected_stations, sinr_values):
    if not sinr_values:
        sinr_mean = -1
    else:
        sinr_values_filtered = [x for x in sinr_values if not np.isnan(x)]
        sinr_mean = np.mean(sinr_values_filtered) if sinr_values_filtered else 0

    # Adjust the number of connected stations to be within the valid range
    connected_stations = min(connected_stations, STS_thresholds[-1] - 1)

    state = get_state(connected_stations, sinr_mean)
    return state



def SINR(received_signal, interfering_signals, noise_power=1e-9):
    received_signal = 10 ** (received_signal / 10)
    interfering_signals = [10 ** (i / 10) for i in interfering_signals]
    interference = sum(interfering_signals)
    return received_signal / (interference + noise_power)


def SNR():
    min_signal_level = -80
    max_signal_level = -40
    num_signal_levels = 16
    random_signal_levels = np.random.uniform(min_signal_level, max_signal_level, num_signal_levels)
    return random_signal_levels


class AccessPoint:
    def __init__(self, num_stations, num_sector, STS):
        self.num_stations = num_stations
        self.num_sector = num_sectors
        self.STS = [STS[sector] for sector in range(num_sectors)]
        self.stations = [Station(i, num_sector, self) for i in range(num_stations)]
        self.sinr_values = [[] for _ in range(num_sector)]
        self.collisions = 0

    def start_beamforming_training(self):
        self.sinr_values = [[] for _ in range(self.num_sector)]

        beacon_frame = self.create_beacon_frame_with_trn_r()
        for station in self.stations:
            station.receive_bti(beacon_frame)
            station.receive_trn_r(beacon_frame)

    def create_beacon_frame_with_trn_r(self):
        return {'SNR': SNR(), 'trn_r': 'TRN-R data'}

    def receive(self, sector):
        received_signals = []
        sts_counter = [0] * self.STS[sector]
        sent_stations = set()

        for station in self.stations:
            for i in range(self.STS[sector]):
                if not station.pair and sector == station.tx_sector_AP and station not in sent_stations:
                    signal = station.send_ssw(i, station.rx_sector) #sector error - > STA(o) ,  AP (X)
                    if signal is not None:
                        received_signals.append((i, signal, station))
                        sts_counter[i] += 1
                        sent_stations.add(station)

        successful_ssw_count = self.handle_received_signals(received_signals, sts_counter, sector)
        return self.sinr_values[sector], successful_ssw_count

    def handle_received_signals(self, received_signals, sts_counter, sector):
        collisions = [count > 1 for count in sts_counter]
        num_collisions = sum(collisions)
        self.collisions += num_collisions

        successful_ssw_count = 0
        collision_signals = []

        for sts, signal, _ in received_signals:
            if not collisions[sts]:
                self.sinr_values[sector].append(signal)
                successful_ssw_count += 1
            else:
                collision_signals.append(signal)

        if collision_signals:
            sinr_values = [
                SINR(signal, [s for s in collision_signals if s != signal])
                for signal in collision_signals
            ]
            self.sinr_values[sector].extend(sinr_values)

        return successful_ssw_count

    def broadcast_ack(self):
        ack_frame = "ACK_DATA"
        for station in self.stations:
            if not station.pair:
                station.receive_ack_frame(ack_frame)

    def next_bi(self):
        self.start_beamforming_training()
        for station in self.stations:
            station.reset_backoff_counts(self)
        

    def all_stations_paired(self):
        return all(station.pair for station in self.stations)


class Station:
    def __init__(self, id, num_sector, AP):
        self.id = id
        self.pair = False
        self.tx_sector_AP = None
        self.rx_sector = None
        self.collisions = 0
        self.data_success = False
        self.sectors = [i for i in range(num_sector)]
        self.backoff_counts = [random.randint(0, AP.STS[i] - 1) for i in range(num_sector)]

    def reset_backoff_counts(self, AP):
        self.backoff_counts = [random.randint(0, AP.STS[i] - 1) for i in range(AP.num_sector)]

    def __str__(self):
        return f"Station {self.id}: paired={self.pair}, tx_sector_AP={self.tx_sector_AP}, rx_sector={self.rx_sector}"

    def receive_bti(self, beacon_frame):
        self.tx_sector_AP = self.get_best_sectors(beacon_frame)

    def get_best_sectors(self, beacon_frame):
        snr_values = beacon_frame['SNR']
        top_sector_indices = np.argsort(snr_values)[-16:]
        best_sector = random.choice(top_sector_indices)
        return best_sector

    def receive_trn_r(self, beacon_frame):
        best_rx_sector = self.get_best_rx_sector()
        self.rx_sector = best_rx_sector

    def get_best_rx_sector(self):
        snr_values = SNR()
        best_rx_sector = np.argmax(snr_values) % len(self.sectors)
        return best_rx_sector

    def send_ssw(self, STS, sector):
        if not self.pair and STS == self.backoff_counts[sector]:
            self.rx_sector = None
            self.data_success = True
            #print(f"Station {self.id} transmitted SSW frame successfully")
            return random.uniform(0.0001, 0.001)
        return None

    def receive_ack_frame(self, ack_frame):
        if not self.pair:
            if self.data_success:
                self.pair = True
                #print(f"Station {self.id} received ACK successfully")
        #else:
            #print(f"Station {self.id} did not receive ACK, will retry in the next BI")




with open('total_STS_Q.txt', 'a') as sts_file, open('Reward_Q.txt','a') as reward_file:
    for episode in range(1000):
        total_reward = 0
        step_count = 0
        STS = [8] * 16
        AP = AccessPoint(num_stations=200, num_sector=4, STS=STS)
        connected_stations = 0
        total_STS_used = 0

        AP.start_beamforming_training()

        while not AP.all_stations_paired(): 
            for i in range(AP.num_sector):
                step_count += 1
                sinr_values = []
                connected_stations = sum(station.pair for station in AP.stations)
                state = get_new_state(connected_stations, sinr_values)

                sector_STS = AP.STS[i]

                action = choose_action(state, AP.STS[i])
                if action == 0:
                    AP.STS[i] = min(32, AP.STS[i] + 1)
                elif action == 1:
                    AP.STS[i] = max(2, AP.STS[i] - 1)
                elif action == 2:
                    pass

                sinr_values_sector, successful_ssw_count = AP.receive(i)
                sinr_values.extend(sinr_values_sector)
                AP.broadcast_ack()

                # print("Paired stations:")
                # for station in AP.stations:
                #     if station.pair:
                #         print(station)

                total_STS_used += AP.STS[i]

                if not AP.all_stations_paired():
                    reward = get_reward(successful_ssw_count, AP.STS[i])
                    total_reward += reward
                    connected_stations = sum(station.pair for station in AP.stations)
                    next_state = get_new_state(connected_stations, sinr_values)

                    next_sector_STS = AP.STS[i]
                    update_q_table(state, action, reward, next_state)
                    AP.next_bi()
                   

        #print(f"EPISODE: {episode} All stations are paired. Simulation complete.")



        print("Episode: {}, Total Reward: {}, Step Count: {}".format(episode, total_reward, step_count))


        sts_file.write(f"{total_STS_used}\n")
        reward_file.write(f"{total_reward}\n")
        
