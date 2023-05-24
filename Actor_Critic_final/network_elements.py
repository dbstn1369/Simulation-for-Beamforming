import random
import numpy as np
import random
import numpy as np
import math


class AccessPoint:
    def __init__(self, num_stations, STS, min_distance=10, max_distance=100):
        self.num_stations = num_stations
        
        self.num_sector = [i for i in range(0, 16)]
        self.STS = [STS[sector] for sector in self.num_sector]
        self.total_STS_used = 0
        self.sector_states = [[0] * 3 for _ in range(len(self.num_sector))]

        self.ssw_list = [[] for _ in range(len(self.num_sector))] 
        self.previous_u = [0] * len(self.num_sector)
        self.collisions = 0
        self.successful_ssw_count = 0

        self.min_distance = min_distance
        self.max_distance = max_distance
        station_positions = np.linspace(self.min_distance, self.max_distance, num_stations)
        self.stations = [Station(i, STS, position=station_positions[i], AP=self) for i in range(self.num_stations)]

       

    def update_STS(self, sector_index, new_STS):

            self.STS[sector_index] = new_STS


    def start_beamforming_training(self):
        unconnected_stations = [station for station in self.stations if not station.pair]

        for station_index, station in enumerate(unconnected_stations):
            beacon_frame = self.create_beacon_frame_with_trn_r(station)
            station.receive_bti(beacon_frame)
            station.receive_trn_r(beacon_frame)
            station.backoff_count = random.randint(0, self.STS[station_index % len(self.STS)])  # backoff_count 재설정

       


    def create_beacon_frame_with_trn_r(self, station):
        return {'SNR': SNR_AP(station.id, station.position), 'trn_r': 'TRN-R data'}


    def receive(self, sector):
        received_signals = []
        sts_counter = [0] * len(self.STS)
        speed_of_light = 3e8
        sent_stations = set()

        for station in self.stations:
            for i in range(len(self.STS)):
                if not station.pair and sector == station.tx_sector_AP and station not in sent_stations:
                    signal = station.send_ssw(i, sector)
                    if signal is not None:
                        distance = station.position
                        time = distance / speed_of_light
                        received_signals.append((i, signal, station, time))
                        sts_counter[i] += 1
                        sent_stations.add(station)

        self.handle_received_signals(received_signals, sts_counter, sector)  # Pass the sector information


    def handle_received_signals(self, received_signals, sts_counter, sector):
        collisions = [count > 1 for count in sts_counter]
        num_collisions = sum(collisions)
        self.collisions += num_collisions

        #print(f"Sector: {sector}")
        #print(f"Length of self.ssw_list: {len(self.ssw_list)}")

        for sts, signal, station, time in received_signals:
            if not collisions[sts]:
                self.ssw_list[sector].append((station.id, signal, time))  # Store the signal in the sector-specific list
                #print(f"Station {station.id} transmitted SSW frame successfully")
            #else:
                #print(f"Station {station.id} transmitted SSW frame, but it collided")


    def broadcast_ack(self):
        for sector_list in self.ssw_list:
            for station_id, _, time in sector_list:
                ack_frame = f"ACK_DATA_{station_id}"
                self.stations[int(station_id)].receive_ack_frame(ack_frame)


    def next_bi(self):
        for sector_list in self.ssw_list:
            sector_list.clear()
        self.start_beamforming_training()
        

          
    def all_stations_paired(self):
        unpaired_stations = [station for station in self.stations if not station.pair]
        if unpaired_stations:
            #print(f"Unpaired stations: {[station.id for station in unpaired_stations]}")
            return False
        else:
            return True
        

    
class Station:
    def __init__(self, id, STS, position, AP=None):
        self.id = id
        self.sectors = [i for i in range(0, 4)]
        self.position = position
        self.AP = AP

        self.snr_values = None

        self.pair = False
        self.tx_sector_AP = None
        self.rx_sector = None
        self.data_success = False
        self.backoff_count = None
        self.attempts = 0


    def receive_bti(self, beacon_frame):
        self.snr_values = np.max(beacon_frame['SNR'])
        self.tx_sector_AP = self.get_best_sectors(beacon_frame)
        self.backoff_count = random.randint(0, self.AP.STS[self.tx_sector_AP] - 1)
        #print(f"backoff of {self.id}: {self.backoff_count}")
        #print(f"Station {self.id}: Best TX sector of AP - {self.tx_sector_AP}")

    def get_best_sectors(self, beacon_frame):
        snr_values = beacon_frame['SNR']
        best_sector = np.argmax(snr_values)
        return best_sector

    def receive_trn_r(self,beacon_frame):
        best_rx_sector = self.get_best_rx_sector()
        self.rx_sector = best_rx_sector  # rx_sector에 할당하는 부분 추가
        #print(f"Station {self.id}: Best RX sector of STA after TRN-R - {best_rx_sector}")

    def get_best_rx_sector(self):
        snr_values = SNR_STA()
        best_rx_sector = np.argmax(snr_values) % len(self.sectors)
        return best_rx_sector

    
    def send_ssw(self, STS, sector):
        if not self.pair and  STS == self.backoff_count and self.tx_sector_AP == sector:
            self.rx_sector = None
            self.data_success = True
            self.attempts += 1
            return self.snr_values if self.snr_values is not None else None
        return None
             

    def receive_ack_frame(self, ack_frame):
        expected_ack_frame = f"ACK_DATA_{self.id}"
        if not self.pair:
            if self.data_success and ack_frame == expected_ack_frame:
                self.pair = True
                #print(f"Station {self.id} received ACK successfully")
            #else:
                #print(f"Station {self.id} did not receive ACK, will retry in the next BI")
        #else:
           #print(f"Station {self.id} is already paired")


def calculate_state_variables(STS, AP, sector_index):
    received_times = [time for _, _, time in AP.ssw_list[sector_index]]  # Use the sector-specific ssw_list

    min_delay = min(received_times) if len(received_times) > 0 else 0
    max_delay = max(received_times) if len(received_times) > 0 else 0

    th_max = 0
    th = 0
    for _, ssw, time in AP.ssw_list[sector_index]:
        snr = ssw  # Assuming ssw should be replaced with snr
        if max_delay == min_delay:
            delay_norm = 0
        else:
            delay_norm = (time - min_delay) / (max_delay - min_delay)
        weight = 1 - delay_norm
        th += weight * snr
        w_max = 1  # Assuming no path loss
        th_max += w_max * snr

    if th_max == 0:
        C_k = 1
    else:
        C_k = (th_max - th) / th_max

    N_ack = sum([station.data_success for station in AP.stations if station.tx_sector_AP == sector_index])

    U_current = N_ack / (STS[sector_index] * len(AP.num_sector))
    U_previous = AP.previous_u[sector_index]
    delta_U = U_current - U_previous

    delta_U_min = -U_previous
    delta_U_max = (len(AP.stations) - N_ack) / (STS[sector_index] * len(AP.num_sector)) - U_previous

    if delta_U_max != delta_U_min:
        delta_U_norm = (delta_U - delta_U_min) / (delta_U_max - delta_U_min)
        delta_U_norm = max(0, min(1, delta_U_norm))
    else:
        delta_U_norm = 0

    AP.previous_u[sector_index] = U_current

    P_rx = 10
    T_rx = max_delay
    P_idle = 1
    T_idle = min_delay

    N_rx = sum([station.data_success for station in AP.stations if station.tx_sector_AP == sector_index])
    N_idle = (STS[sector_index] * len(AP.num_sector)) - N_rx

    E_rx = P_rx * T_rx
    E_idle = P_idle * T_idle
    E_t = E_rx * N_rx + E_idle * N_idle

    E = 1 / (1 + np.exp(-(E_t)))

    return STS[sector_index], C_k, delta_U_norm, E


def SNR_STA():
    # 신호 레벨 범위 (dBm 단위)
    max_signal_level = 80
    min_signal_level = 30

    # 무작위 신호 레벨 개수
    num_signal_levels = 4

    # 무작위 신호 레벨 생성
    random_signal_levels = np.random.uniform(min_signal_level, max_signal_level, num_signal_levels)
    random_signal_levels = np.around(random_signal_levels, 4)
    #print("Random signal levels (dBm):", random_signal_levels)
    return random_signal_levels



def SNR_AP(station_id, distance):
    # # 거리에 따른 path loss 값 계산
    d0 = 1  # 레퍼런스 거리 (1m로 가정)
    PL_d0 = 2  # d0에서의 손실 값 (2dB로 가정)
    n = 2  # path loss exponent (거리에 따라 다르게 설정됨)
    PL_d = PL_d0 + 10 * n * np.log10(distance/d0)
    # # 신호 레벨 범위 (dBm 단위)
    max_signal_level = 100 
    min_signal_level = 40 
    # 무작위 신호 레벨 개수
    num_signal_levels = 16
    # 무작위 신호 레벨 생성
    random_signal_levels = np.random.uniform(min_signal_level - PL_d, max_signal_level - PL_d, num_signal_levels)
    random_signal_levels = np.around(random_signal_levels, 4)
    #print(f"Random signal levels (dBm) for station {station_id}:", random_signal_levels)
    return random_signal_levels


