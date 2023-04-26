import random
import numpy as np
import random
import numpy as np




class AccessPoint:
    def __init__(self, num_stations, STS, min_distance=10, max_distance=100):
        self.num_stations = num_stations
        self.STS = STS
        self.num_sector = 4
        self.ssw_list = []
        self.previous_u = 0
        self.min_distance = min_distance
        self.max_distance = max_distance
        station_positions = np.linspace(self.min_distance, self.max_distance, num_stations)
        self.stations = [Station(i, STS, position=station_positions[i], AP=self) for i in range(self.num_stations)]
        self.BI = self.stations[random.randint(0, num_stations-1)]
        self.set_random_position()


    def reset_all_stations(self):
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
            station.backoff_count = random.randint(1, self.STS)  # backoff_count 재설정            

    def create_beacon_frame_with_trn_r(self):
        return {'SNR': SNR(), 'trn_r': 'TRN-R data'}



    
    def receive(self, sector):
        successful_ssw_count = 0
        received_signals = []
        sts_counter = [0] * self.STS  # STS별로 사용한 횟수를 저장할 리스트 생성

        for i in range(self.STS):
            for station in self.stations:
                if not station.pair and sector == station.tx_sector_AP:
                    signal = station.send_ssw(i, sector)
                    if signal is not None:
                        received_signals.append(signal)
                        sts_counter[i] += 1  # 해당 STS에 대한 카운터를 증가시킵니다.
                    if station.data_success:
                        successful_ssw_count += 1

        # 충돌 처리: 한 STS에서 두 개 이상의 신호가 수신되면 충돌로 간주하고 성공적인 SSW 개수를 줄입니다.
        collisions = sum([1 for count in sts_counter if count > 1])
        successful_ssw_count -= collisions

        return successful_ssw_count

    def broadcast_ack(self):
        ack_frame = "ACK_DATA"
        for station in self.stations:
            if station.pair == False:
                station.receive_ack_frame(ack_frame)



    def next_bi(self):
        self.start_beamforming_training()
        for i in range(self.num_sector):
            successful_ssw_count = self.receive(i)
        self.broadcast_ack()    

  

    def all_stations_paired(self):
        unpaired_stations = [station for station in self.stations if not station.pair]
        if unpaired_stations:
            print(f"Unpaired stations: {[station.id for station in unpaired_stations]}")
            return False
        else:
            return True

class Station:
    def __init__(self, id, STS, position=None, AP=None):
        self.id = id
        self.pair = False
        self.tx_sector_AP = None
        self.rx_sector = None
        self.collisions = 0
        self.data_success = False
        self.sectors = [i for i in range(1, 4)]
        self.backoff_count = random.randint(1, STS)
        self.AP = AP

    def reset_station(self):
        self.pair = False
        self.tx_sector_AP = None
        self.rx_sector = None
        self.data_success = False
        self.backoff_count = random.randint(1, self.AP.STS)


    def receive_bti(self, beacon_frame):
        self.tx_sector_AP = self.get_best_sectors(beacon_frame)
        print(f"Station {self.id}: Best TX sector of AP - {self.tx_sector_AP}")

    def get_best_sectors(self, beacon_frame):
        snr_values = beacon_frame['SNR']
        best_sector = np.argmax(snr_values) + 1
        return best_sector

    def receive_trn_r(self, beacon_frame):
        best_rx_sector = self.get_best_rx_sector(beacon_frame)
        self.rx_sector = best_rx_sector  # rx_sector에 할당하는 부분 추가
        print(f"Station {self.id}: Best RX sector of STA after TRN-R - {best_rx_sector}")

    def get_best_rx_sector(self, beacon_frame):
        snr_values = SNR()
        best_rx_sector = np.argmax(snr_values) % len(self.sectors) + 1
        return best_rx_sector

    
    def send_ssw(self, STS, sector):
        if not self.pair and STS == self.backoff_count:  # 이미 연결된 STA들이 참여하지 않도록 조건 추가
            self.rx_sector = None
            self.data_success = True
            print("Station " + str(self.id) + " transmitted SSW frame successfully")
            return random.uniform(0.0001, 0.001)  # 임의의 수신 신호 전송
        return None

    def receive_ack_frame(self, ack_frame):
        if self.pair == False:
            if self.data_success == True:
                self.pair = True
                print("Station " + str(self.id) + " received ACK successfully")
        else:
            print("Station " + str(self.id) + " did not receive ACK, will retry in the next BI")


def calculate_state_variables(STS, s_sts, AP):
    C_k = 0
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

    s_pd = np.mean(propagation_delay)
    S_norm = (s_pd - min_propagation_delay) / (max_propagation_delay - min_propagation_delay)
    W = 1 - S_norm
    print(f"S_norm: ", S_norm)
    # Calculate congestion
    weighted_snr_sum = 0
    weight_sum = 0
    for ssw in AP.ssw_list:
        weighted_snr_sum += W * ssw.snr
        weight_sum += W
    if weight_sum != 0:
        s_c = weighted_snr_sum / weight_sum
        C_norm = (s_c - np.mean([ssw.snr for ssw in AP.ssw_list])) / np.std([ssw.snr for ssw in AP.ssw_list])
        C_k = 1 - C_norm

    # Calculate STS usage
    N_ack = sum([1 for station in AP.stations if station.data_success])
    U_current = N_ack / STS
    U_previous = AP.previous_u
    delta_U = U_current - U_previous
    delta_U_min = -1
    delta_U_max = 1
    delta_U_norm = (delta_U - delta_U_min) / (delta_U_max - delta_U_min)

    AP.previous_u = U_current 

    return s_sts, C_k, delta_U_norm

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
