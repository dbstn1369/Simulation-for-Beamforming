import random
import numpy as np
import time
import torch
import os

os.environ["TORCH_USE_CUDA_DSA"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

num_states = 32  # Maximum number of connectable stations
num_actions = 3  # Increase, decrease, maintain
num_sectors = 16  # Number of sectors

# Update the q_table tensor to have 4 dimensions
q_table = torch.zeros((num_states, num_states, num_actions, num_sectors), dtype=torch.float32, device=device)

learning_rate = 0.1
discount_factor = 0.95
exploration_rate = 1.0
exploration_rate_decay = 0.001


def choose_action(state, sector_STS):
    state = torch.as_tensor(state, dtype=torch.int64, device=device)
    sector_STS = torch.as_tensor(sector_STS, dtype=torch.int64, device=device)

    if torch.rand(1).item() < exploration_rate:
        return torch.randint(num_actions, (1,), device=device, dtype=torch.int64).item()
    else:
        q_values = q_table[state[0], state[1], :, sector_STS]

        q_values_numpy = q_values.cpu().numpy()
        max_indices = np.where(q_values_numpy == q_values_numpy.max())[0]
        selected_index = np.random.choice(max_indices)
        return selected_index.item()






def update_q_table(state, action, reward, next_state, sector_STS, next_sector_STS):
    state = torch.as_tensor(state, dtype=torch.int64, device=device)
    next_state = torch.as_tensor(next_state, dtype=torch.int64, device=device)

    reward = torch.as_tensor(reward, dtype=torch.float32, device=device)
    action = torch.as_tensor(action, dtype=torch.int64, device=device)
    sector_STS = torch.as_tensor(sector_STS, dtype=torch.int64, device=device)
    next_sector_STS = torch.as_tensor(next_sector_STS, dtype=torch.int64, device=device)

    if torch.all(next_state < num_states) and torch.all(next_sector_STS < num_sectors):
        q_value = q_table[state[0], state[1], action, sector_STS]
        max_next_q_value = torch.max(q_table[next_state[0], next_state[1], :, next_sector_STS])
        target_q_value = reward + discount_factor * max_next_q_value
        q_table[state[0], state[1], action, sector_STS] = q_value + learning_rate * (target_q_value - q_value)
    else:
        print(f"Ignoring invalid next_state: {next_state} or next_sector_STS: {next_sector_STS}")




 
def get_reward(successful_ssw_count, total_ssw_count):
    if total_ssw_count == 0:
        return 0
    return successful_ssw_count / total_ssw_count


def get_new_state(connected_stations, sinr_values):
    if not sinr_values:
        sinr_mean = -1
    else:
        sinr_values_filtered = [x for x in sinr_values if not np.isnan(x)]
        sinr_mean = np.mean(sinr_values_filtered) if sinr_values_filtered else 0

    # Adjust the number of connected stations to be within the valid range
    connected_stations = min(connected_stations, num_states - 1)

    state = np.array([connected_stations, sinr_mean], dtype=np.int32)
    return state, sinr_values


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
        self.num_sector = num_sector
        self.STS = [STS[sector] for sector in range(num_sector)]
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
                    signal = station.send_ssw(i, sector)
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
        for i in range(self.num_sector):
            successful_ssw_count, sinr_values = self.receive(i)
        self.broadcast_ack()

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
        best_rx_sector = self.get_best_rx_sector(beacon_frame)
        self.rx_sector = best_rx_sector

    def get_best_rx_sector(self, beacon_frame):
        snr_values = SNR()
        best_rx_sector = np.argmax(snr_values) % len(self.sectors)
        return best_rx_sector

    def send_ssw(self, STS, sector):
        if not self.pair and STS == self.backoff_counts[sector]:
            self.rx_sector = None
            self.data_success = True
            print(f"Station {self.id} transmitted SSW frame successfully")
            return random.uniform(0.0001, 0.001)
        return None

    def receive_ack_frame(self, ack_frame):
        if not self.pair:
            if self.data_success:
                self.pair = True
                print(f"Station {self.id} received ACK successfully")
        else:
            print(f"Station {self.id} did not receive ACK, will retry in the next BI")


total_STS_used = 0
with open('total_time_Q.txt', 'a') as time_file, open('total_STS_Q.txt', 'a') as sts_file:
    for episode in range(1000):
        STS = [32] * 16
        AP = AccessPoint(num_stations=500, num_sector=num_sectors, STS=STS)
        connected_stations = 0
        total_time = 0
        total_STS_used = 0
        start_time = time.time()

        AP.start_beamforming_training()

        while not AP.all_stations_paired():
            sinr_values = []
            connected_stations = sum(station.pair for station in AP.stations)
            state, sinr_values = get_new_state(connected_stations, sinr_values)

            sector_STS = AP.STS[:]

            action = choose_action(state, sector_STS)
            if action == 0:
                AP.STS = [min(32, sts + 1) for sts in AP.STS]
            elif action == 1:
                AP.STS = [max(5, sts - 1) for sts in AP.STS]
            elif action == 2:
                AP.STS = AP.STS

            successful_ssw_count = 0
            sinr_values = []
            for i in range(AP.num_sector):
                sinr_values_sector, successful_ssw_count_sector = AP.receive(i)
                if sinr_values_sector:
                    sinr_values.extend(sinr_values_sector)
                successful_ssw_count += successful_ssw_count_sector

            AP.broadcast_ack()

            print("Paired stations:")
            for station in AP.stations:
                if station.pair:
                    print(station)

            if not AP.all_stations_paired():
                total_STS_used += sum(AP.STS)

                reward = get_reward(successful_ssw_count, sum(AP.STS))
                connected_stations = sum(station.pair for station in AP.stations)
                next_state, _ = get_new_state(connected_stations, sinr_values)

                next_sector_STS = AP.STS[:]
                update_q_table(state, action, reward, next_state, sector_STS, next_sector_STS)

                AP.next_bi()

        end_time = time.time()
        total_time = end_time - start_time
        print(f"EPISODE: {episode} All stations are paired. Simulation complete.")
        print(f"Total simulation time: {total_time:.3f} seconds")
        time_file.write(f"{total_time:.3f}\n")
        sts_file.write(f"{total_STS_used}\n")

        exploration_rate = max(0.01, exploration_rate - exploration_rate_decay)
