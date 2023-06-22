import time
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

        for sts, signal, station, time in received_signals:
            if not collisions[sts]:
                self.ssw_list[sector].append((station.id, signal, time))  # Store the signal in the sector-specific list


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


    def get_best_sectors(self, beacon_frame):
        snr_values = beacon_frame['SNR']
        best_sector = np.argmax(snr_values)
        return best_sector

    def receive_trn_r(self,beacon_frame):
        best_rx_sector = self.get_best_rx_sector()
        self.rx_sector = best_rx_sector


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


def SNR_STA():
    max_signal_level = 80
    min_signal_level = 30
    num_signal_levels = 4
    random_signal_levels = np.random.uniform(min_signal_level, max_signal_level, num_signal_levels)
    random_signal_levels = np.around(random_signal_levels, 4)
    return random_signal_levels


def SNR_AP(station_id, distance):
    d0 = 1
    PL_d0 = 2
    n = 2
    PL_d = PL_d0 + 10 * n * np.log10(distance/d0)
    max_signal_level = 100 
    min_signal_level = 40 
    num_signal_levels = 16
    random_signal_levels = np.random.uniform(min_signal_level - PL_d, max_signal_level - PL_d, num_signal_levels)
    random_signal_levels = np.around(random_signal_levels, 4)
    return random_signal_levels


def mle_update_STS(failed_STA_count, total_STA_count):
    if total_STA_count != 0:  # Avoid division by zero
        failure_rate = failed_STA_count / total_STA_count
    else:
        failure_rate = 0

    if failure_rate > 0.5:
        new_STS = 16  # Increase STS
    elif failure_rate < 0.2:
        new_STS = 4  # Decrease STS
    else:
        new_STS = 8  # Keep STS unchanged

    return new_STS

with open('total_STS_Cosa.txt', 'a') as sts_file :
    for episode in range(1000):
        STS = [32] * 16
        AP = AccessPoint(num_stations=500, STS=STS)
        
        connected_stations = []
        successful_ssw_count = 0
        bi = 0

        AP.start_beamforming_training()

        while not AP.all_stations_paired():
            connected_stations = [station for station in AP.stations if station.pair]

            for i in range(len(AP.num_sector)):
                AP.receive(i)
                successful_ssw_count = len(AP.ssw_list)
                failed_STA_count = len([station for station in AP.stations if not station.pair])
                AP.broadcast_ack()


                new_STS = mle_update_STS(failed_STA_count, len(AP.stations))

                AP.update_STS(i, new_STS)
                AP.total_STS_used += new_STS

                
                successful_ssw_count = 0
             
                
                if not AP.all_stations_paired():
                    AP.next_bi()
                    bi += 1
            
        print(f"Episode: {episode}")
        print(f"Total_STS_used: {AP.total_STS_used}")
            
        sts_file.write(str(AP.total_STS_used) + "\n")
