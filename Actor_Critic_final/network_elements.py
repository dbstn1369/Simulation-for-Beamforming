import random
import numpy as np
import random
import numpy as np
import math


class AccessPoint:
    def __init__(self, num_stations, STS, min_distance=10, max_distance=100):
        self.num_stations = num_stations
        self.STS = STS
        self.num_sector = [i for i in range(0, 6)]
        self.total_STS_used = 0

        self.ssw_list = []
        self.previous_u = 0
        self.collisions = 0
        self.successful_ssw_count = 0

    
        self.min_distance = min_distance
        self.max_distance = max_distance
        station_positions = np.linspace(self.min_distance, self.max_distance, num_stations)
        self.stations = [Station(i, STS, position=station_positions[i], AP=self) for i in range(self.num_stations)]

       

    def update_STS(self, new_STS):
            self.STS = new_STS

    def start_beamforming_training(self):
    
        unconnected_stations = [station for station in self.stations if not station.pair]
        
        for station in unconnected_stations:
            beacon_frame = self.create_beacon_frame_with_trn_r(station)
            station.receive_bti(beacon_frame)
            station.receive_trn_r(beacon_frame)
            station.backoff_count = random.randint(0, self.STS)  # backoff_count 재설정            

    def create_beacon_frame_with_trn_r(self, station):
        return {'SNR': SNR_AP(station.id, station.position), 'trn_r': 'TRN-R data'}


    def receive(self, sector):
        received_signals = []
        sts_counter = [0] * self.STS
        speed_of_light = 3e8

        for i in range(self.STS):
            for station in self.stations:
                if not station.pair and sector == station.tx_sector_AP:
                    signal = station.send_ssw(i, sector)
                    if signal is not None:
                        distance = station.position
                        time = distance / speed_of_light
                        received_signals.append((i, signal, station, time))
                        sts_counter[i] += 1

        
        self.handle_received_signals(received_signals, sts_counter)
        
    def handle_received_signals(self, received_signals, sts_counter):
        collisions = [count > 1 for count in sts_counter]
        num_collisions = sum(collisions)
        self.collisions += num_collisions

        for sts, signal, station, time in received_signals:
            if not collisions[sts]:
                self.ssw_list.append((station.id, signal, time))
                #print(f"Station {station.id} transmitted SSW frame successfully")    
            #else:
                #print(f"Station {station.id} transmitted SSW frame, but it collided")

    

    def broadcast_ack(self):
        for station_id, _, time in self.ssw_list:  # Use index 0 to get the station ID from the tuple
            ack_frame = f"ACK_DATA_{station_id}"
            self.stations[int(station_id)].receive_ack_frame(ack_frame)


    def next_bi(self):
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
        self.backoff_count = random.randint(0, STS-1)
        self.attempts = 0


    def receive_bti(self, beacon_frame):
        self.snr_values = np.max(beacon_frame['SNR'])
        self.tx_sector_AP = self.get_best_sectors(beacon_frame)
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
        if not self.pair and STS == self.backoff_count and self.tx_sector_AP == sector:
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


def calculate_state_variables(STS, AP):
    # Calculate delay
    received_times = [time for _, _, time in AP.ssw_list]
    
    min_delay = min(received_times) if len(received_times) > 0 else 0  # or any other default value

    max_delay = max(received_times) if len(received_times) > 0 else 0  # or any other default value



    th_max = 0
    th = 0
    for (station_id, ssw, time) in AP.ssw_list:
        station = AP.stations[int(station_id)]
        snr = station.snr_values
      
        if max_delay == min_delay:
                delay_norm = 0
            
        else:
            delay_norm = (time - min_delay) / (max_delay - min_delay)
       
    
        weight = 1 - delay_norm
        th += weight * snr
      
        w_max = 1  #no path loss 
        th_max += w_max * snr  # Assuming that ssw should be replaced with snr

    if th_max == 0:
        #print("Warning: th_max is zero, defaulting C_k to 0")
        C_k = 1
    else:
        C_k = (th_max - th) / th_max  # Replacing sigmoid function with original equation

    # Calculate STS usage
    N_ack = sum([station.data_success for station in AP.stations])
    U_current = N_ack / (STS*len(AP.num_sector))
    U_previous = AP.previous_u
    delta_U = U_current - U_previous

    # Calculate minimum and maximum possible values of delta_U
    delta_U_min = -U_previous
    delta_U_max = (AP.num_stations - N_ack) / STS * len(AP.num_sector) - U_previous

    if delta_U_max != delta_U_min:
        delta_U_norm = (delta_U - delta_U_min) / (delta_U_max - delta_U_min)
        # Make sure delta_U_norm is between 0 and 1
        delta_U_norm = max(0, min(1, delta_U_norm))
    else:
        delta_U_norm = 0

    AP.previous_u = U_current

    # Calculate energy consumption
    P_rx = 10
    T_rx = max_delay
    P_idle = 1
    T_idle = min_delay

    N_rx = sum([station.data_success for station in AP.stations])
    N_idle = (STS*len(AP.num_sector)) - N_rx

    E_rx = P_rx * T_rx
    E_idle = P_idle * T_idle
    E_t = E_rx * N_rx + E_idle * N_idle

    E = 1 / (1 + np.exp(-(E_t)))

    return STS, C_k, delta_U_norm, E



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
    num_signal_levels = 6
    # 무작위 신호 레벨 생성
    random_signal_levels = np.random.uniform(min_signal_level - PL_d, max_signal_level - PL_d, num_signal_levels)
    random_signal_levels = np.around(random_signal_levels, 4)
    #print(f"Random signal levels (dBm) for station {station_id}:", random_signal_levels)
    return random_signal_levels


