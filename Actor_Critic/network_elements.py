import random
import numpy as np
import random
import numpy as np
import torch
import torch.nn as nn

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

class AccessPoint:
    def __init__(self, num_stations, STS):  # STS 매개변수 추가
        self.num_stations = num_stations
        self.num_sector = 6
        self.stations = [Station(i, STS) for i in range(num_stations)]  # STS 값을 전달
        self.sinr_values = [[] for _ in range(self.num_sector)]

    def start_beamforming_training(self):
        self.sinr_values = []

        beacon_frame = self.create_beacon_frame_with_trn_r()
        for station in self.stations:
            station.receive_bti(beacon_frame)
            station.receive_trn_r(beacon_frame)

    def create_beacon_frame_with_trn_r(self):
        return {'SNR': SNR(), 'trn_r': 'TRN-R data'}

    def recieve(self, sector):
        successful_ssw_count = 0
        received_signals = []
        if sector >= len(self.sinr_values):
            return [], successful_ssw_count  # successful_ssw_count를 리스트가 아닌 정수로 반환
        for i in range(STS):
            for station in self.stations:
                if not station.pair and sector == station.tx_sector_AP:
                    signal = station.send_ssw(i, sector)
                    if signal is not None:
                        received_signals.append(signal)
                    if station.data_success:
                        successful_ssw_count += 1
            if len(received_signals) > 1:
                sinr_values = [SINR(signal, [s for s in received_signals if s != signal]) for signal in received_signals]
                self.sinr_values[sector].extend(sinr_values)
        if not self.sinr_values[sector]:
            return [], successful_ssw_count  # successful_ssw_count를 리스트가 아닌 정수로 반환
        else:
            return self.sinr_values[sector], successful_ssw_count  # successful_ssw_count를 리스트가 아닌 정수로 반환

    def broadcast_ack(self):
        ack_frame = "ACK_DATA"
        for station in self.stations:
            if station.pair == False:
                station.receive_ack_frame(ack_frame)

    def next_bi(self):
        self.start_beamforming_training()
        for i in range(self.num_sector):
            successful_ssw_count, sinr_values = self.recieve(i)  # sinr_values 반환값 추가
        self.broadcast_ack()

    def all_stations_paired(self):
        return all(station.pair for station in self.stations)

import random
class Station:
    def __init__(self, id, STS):  # STS 매개변수 추가
        self.id = id
        self.pair = False
        self.tx_sector_AP = None
        self.rx_sector = None
        self.collisions = 0
        self.data_success = False
        self.sectors = [i for i in range(1, 5)]
        self.backoff_count = random.randint(1, STS)

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
