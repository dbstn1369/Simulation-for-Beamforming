import random
import numpy as np
import time

def SNR():
    # 신호 레벨 범위 (dBm 단위)
    min_signal_level = -80
    max_signal_level = -40

    # 무작위 신호 레벨 개수
    num_signal_levels = 6

    # 무작위 신호 레벨 생성
    random_signal_levels = np.random.uniform(min_signal_level, max_signal_level, num_signal_levels)
    #print("Random signal levels (dBm):", random_signal_levels)
    return random_signal_levels

STS = 32

class AccessPoint:
    def __init__(self, num_stations):
        self.num_stations = num_stations
        self.num_sector = 6
        self.stations = [Station(i) for i in range(num_stations)]

    def start_beamforming_training(self):
        beacon_frame = self.create_beacon_frame_with_trn_r()
        for station in self.stations:
            station.receive_bti(beacon_frame)
            station.receive_trn_r(beacon_frame)

    def create_beacon_frame_with_trn_r(self):
        return {'SNR': SNR(), 'trn_r': 'TRN-R data'}

    def recieve(self, sector):
        #print(f"sector{sector} of AP")
        for i in range(STS):
            for station in self.stations:
                if not station.pair and sector == station.tx_sector_AP:
                    #print(f"STS: {i}")
                    station.send_ssw(i, sector)

    def broadcast_ack(self):
        ack_frame = "ACK_DATA"
        for station in self.stations:
            station.receive_ack_frame(ack_frame)

    def next_bi(self):
        self.start_beamforming_training()


    def all_stations_paired(self):
        return all(station.pair for station in self.stations)

class Station:
    def __init__(self, id):
        self.id = id
        self.pair = False
        self.tx_sector_AP = None
        self.rx_sector = None
        self.collisions = 0
        self.data_success = False
        self.sectors = [i for i in range(1, 4)]
        self.backoff_count = random.randint(1, STS)

    def receive_bti(self, beacon_frame):
        self.tx_sector_AP = self.get_best_sectors(beacon_frame)
        #print(f"Station {self.id}: Best TX sector of AP - {self.tx_sector_AP}")

    def get_best_sectors(self, beacon_frame):
        snr_values = beacon_frame['SNR']
        best_sector = np.argmax(snr_values) + 1
        return best_sector

    def receive_trn_r(self, beacon_frame):
        best_rx_sector = self.get_best_rx_sector(beacon_frame)
        #print(f"Station {self.id}: Best RX sector of STA after TRN-R - {best_rx_sector}")

    def get_best_rx_sector(self, beacon_frame):
        snr_values = SNR()
        best_rx_sector = np.argmax(snr_values) % len(self.sectors) + 1
        return best_rx_sector

    def send_ssw(self, STS, sector):
        if STS == self.backoff_count:
            self.rx_sector = None
            self.data_success = True
            print("Station " + str(self.id) + " transmitted SSW frame successfully")
        return

    def receive_ack_frame(self, ack_frame):
        if self.pair == False:
            if self.data_success == True:
                self.pair = True
                print("Station " + str(self.id) + " received ACK successfully")
        else:
            print("Station " + str(self.id) + " did not receive ACK, will retry in the next BI")



episode = 1000
with open('total_time_original.txt', 'a') as time_file, open('total_STS_original.txt', 'a') as sts_file:
    for i in range(episode):
        # Create an AP and stations
        AP = AccessPoint(num_stations=150)
        start_time = time.time()  # 시뮬레이션 시작 시간 측정
        total_STS = 0
        # Start Beamforming Training
        AP.start_beamforming_training()

        while not AP.all_stations_paired():
            for i in range(AP.num_sector):
                AP.recieve(i)

            AP.broadcast_ack()

            if not AP.all_stations_paired():
                total_STS += STS*AP.num_sector
                print("Not all stations are paired. Starting next BI process.")
                AP.next_bi()

        end_time = time.time()  # 시뮬레이션 종료 시간 측정
        total_time = end_time - start_time
        print(f"Episode: {i}")
        print(f"Total_STS_used: {total_STS}")
        #print("All stations are paired. Simulation complete.")
        #print(f"Total simulation time: {total_time:.3f} seconds")
        # 결과를 파일에 저장
       
        time_file.write(f"{total_time:.3f}\n")
        sts_file.write(str(total_STS) + "\n")
        
         
