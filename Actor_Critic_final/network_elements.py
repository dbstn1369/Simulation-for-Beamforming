import random
import numpy as np
import random
import numpy as np
import math


class AccessPoint:
    def __init__(self, num_stations, STS, min_distance=10, max_distance=100):
        self.num_stations = num_stations
        self.STS = STS
        self.num_sector = 6

        self.ssw_list = []
        self.previous_u = 0
        self.collisions = 0
        self.successful_ssw_count = 0

        self.min_distance = min_distance
        self.max_distance = max_distance
        station_positions = np.linspace(self.min_distance, self.max_distance, num_stations)
        self.stations = [Station(i, STS, position=station_positions[i], AP=self) for i in range(self.num_stations)]
        self.BI = self.stations[random.randint(0, num_stations-1)]
        self.set_random_position()


    def reset_all_stations(self):
        self.ssw_list = []
        self.previous_u = 0
        self.collisions = 0
        self.successful_ssw_count = 0

        for station in self.stations:
            station.reset_station()


    def set_random_position(self):
        for station in self.stations:
            station.position = np.random.uniform(self.min_distance, self.max_distance)


    def start_beamforming_training(self):
    
        beacon_frame = self.create_beacon_frame_with_trn_r()
        unconnected_stations = [station for station in self.stations if not station.pair]
        
        for station in unconnected_stations:
            station.receive_bti(beacon_frame)
            station.receive_trn_r(beacon_frame)
            station.backoff_count = random.randint(0, self.STS)  # backoff_count 재설정            

    def create_beacon_frame_with_trn_r(self):
        return {'SNR': SNR(), 'trn_r': 'TRN-R data'}


    def receive(self, sector):
        received_signals = []
        sts_counter = [0] * self.STS

        for i in range(self.STS):
            for station in self.stations:
                if not station.pair and sector == station.tx_sector_AP:
                    signal = station.send_ssw(i, sector)
                    if signal is not None:
                        received_signals.append((i, signal, station))
                        sts_counter[i] += 1

        collisions = [count > 1 for count in sts_counter]
        num_collisions = sum(collisions)
        self.collisions += num_collisions

        for sts, signal, station in received_signals:
            if not collisions[sts]:
                self.ssw_list.append((station.id, signal))
                print(f"Station {station.id} transmitted SSW frame successfully")    
            else:
                print(f"Station {station.id} transmitted SSW frame, but it collided")

        successful_ssw_count = len(self.ssw_list)
        return successful_ssw_count
    

    
    def broadcast_ack(self):
        for station_id, _ in self.ssw_list:  # Use index 0 to get the station ID from the tuple
            ack_frame = f"ACK_DATA_{station_id}"
            self.stations[int(station_id)].receive_ack_frame(ack_frame)


    def next_bi(self):
        self.start_beamforming_training()
        for i in range(self.num_sector):
            self.successful_ssw_count = self.receive(i)
        self.broadcast_ack()
          
  

    def all_stations_paired(self):
        unpaired_stations = [station for station in self.stations if not station.pair]
        if unpaired_stations:
            print(f"Unpaired stations: {[station.id for station in unpaired_stations]}")
            return False
        else:
            return True
        

    def collision_probability(self):
            return self.collisions / self.STS
    
class Station:
    def __init__(self, id, STS, position=None, AP=None):
        self.id = id
        self.sectors = [i for i in range(0, 3)]
        self.position = position
        self.AP = AP

        self.pair = False
        self.tx_sector_AP = None
        self.rx_sector = None
        self.data_success = False
        self.backoff_count = random.randint(0, STS-1)
        self.attempts = 0

    def reset_station(self):
        self.pair = False
        self.tx_sector_AP = None
        self.rx_sector = None
        self.data_success = False
        self.backoff_count = random.randint(0, self.AP.STS-1)
        self.attempts = 0

    def receive_bti(self, beacon_frame):
        self.tx_sector_AP = self.get_best_sectors(beacon_frame)
        print(f"Station {self.id}: Best TX sector of AP - {self.tx_sector_AP}")

    def get_best_sectors(self, beacon_frame):
        snr_values = beacon_frame['SNR']
        best_sector = np.argmax(snr_values)
        return best_sector

    def receive_trn_r(self, beacon_frame):
        best_rx_sector = self.get_best_rx_sector(beacon_frame)
        self.rx_sector = best_rx_sector  # rx_sector에 할당하는 부분 추가
        print(f"Station {self.id}: Best RX sector of STA after TRN-R - {best_rx_sector}")

    def get_best_rx_sector(self, beacon_frame):
        snr_values = SNR()
        best_rx_sector = np.argmax(snr_values) % len(self.sectors)
        return best_rx_sector

    
    def send_ssw(self, STS, sector):
        if not self.pair and STS == self.backoff_count and self.tx_sector_AP == sector:
            self.rx_sector = None
            self.data_success = True
            self.attempts += 1
            return random.uniform(0.0001, 0.001)  # 임의의 수신 신호 전송
        return None


    def receive_ack_frame(self, ack_frame):
        expected_ack_frame = f"ACK_DATA_{self.id}"
        if not self.pair:
            if self.data_success and ack_frame == expected_ack_frame:
                self.pair = True
                print(f"Station {self.id} received ACK successfully")
            else:
                print(f"Station {self.id} did not receive ACK, will retry in the next BI")
        else:
            print(f"Station {self.id} is already paired")


def calculate_state_variables(STS, s_sts, AP):
    C_k = 0
    C_norm = 0
    alpha = 0.9
    
    collision_prob = AP.collision_probability()
    AP.collisions = 0
    print(f"Collision probability: {collision_prob}")
    # Calculate current STS count
    if isinstance(STS, int):
        s_sts = STS
    else:
        s_sts = int(STS)

    propagation_delay = []
    for station in AP.stations:
        delay = np.linalg.norm(station.position - 0) / 3e8
        propagation_delay.append(delay)

    min_propagation_delay = np.linalg.norm(AP.min_distance) / 3e8
    max_propagation_delay = np.linalg.norm(AP.max_distance) / 3e8

    W = []
    for pd in propagation_delay:
        S_norm = (pd - min_propagation_delay) / (max_propagation_delay - min_propagation_delay)
        W.append(1 - S_norm)

     # Calculate congestion
    weighted_snr_sum = 0
    weight_sum = 0
    for (station_id, ssw), wi in zip(AP.ssw_list, W):
        station = AP.stations[int(station_id)]
        delay = np.linalg.norm(station.position - 0) / 3e8
        S_norm = (delay - np.linalg.norm(AP.min_distance) / 3e8) / (np.linalg.norm(AP.max_distance - AP.min_distance) / 3e8)
        
        weight = wi
        weighted_snr_sum += weight * ssw
        weight_sum += weight

        C = weighted_snr_sum / weight_sum
        min_ssw = np.min([ssw for _, ssw in AP.ssw_list])
        max_ssw = np.max([ssw for _, ssw in AP.ssw_list])

        if max_ssw != min_ssw:
            C_norm = (C - min_ssw) / (max_ssw - min_ssw)
        else:
            C_norm = 0

    C_k = 1 / (1 + np.exp(-alpha * (collision_prob-C_norm)))

    print(f"C_k: {C_k}")
    # Calculate STS usage
    N_ack = sum([1 for station in AP.stations if station.data_success])
    U_current = N_ack / (STS*AP.num_sector)
    U_previous = AP.previous_u
    delta_U = U_current - U_previous

    # Calculate minimum and maximum possible values of delta_U
    delta_U_min = -U_previous
    delta_U_max = (AP.num_stations - N_ack) / STS * AP.num_sector - U_previous

    if delta_U_max != delta_U_min:
        delta_U_norm = (delta_U - delta_U_min) / (delta_U_max - delta_U_min)
            # Make sure delta_U_norm is between 0 and 1
        delta_U_norm = max(0, min(1, delta_U_norm))
    else:
        delta_U_norm = 0

    AP.previous_u = U_current


       # Calculate energy consumption
    P_rx = 10
    T_rx = max_propagation_delay
    P_idle = 1
    T_idle = min_propagation_delay

    N_rx = sum([1 for station in AP.stations if station.data_success])  # 수신된 STS 개수 계산
    N_idle = (STS*AP.num_sector) - N_rx  # 대기 중인 STS 개수 계산

    E_rx = P_rx * T_rx
    E_idle = P_idle * T_idle
    E_t = E_rx * N_rx + E_idle * N_idle

    E = 1 / (1 + math.exp(-(E_t)))

    return s_sts, C_k, delta_U_norm, E
    #return s_sts, C_k, delta_U_norm

# def SINR(received_signal, interfering_signals, noise_power=1e-9):
#     interference = sum(interfering_signals)
#     return received_signal / (interference + noise_power)

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
