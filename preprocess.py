import os
import csv
import re
from scapy.all import rdpcap
from scapy.layers.inet import IP, TCP


def convert_to_netflow(pcap_folder, num_files=None):
    """
        Convert flow data to netflow format
    """
    flow_stats = {}
    target = {}
    filenames = os.listdir(pcap_folder)
    if num_files is None:
        num_files = len(filenames)
    for i in range(num_files):
        filename = filenames[i]
        if filename.endswith('.pcap'):
            with open(os.path.join(pcap_folder, filename), 'rb') as pcap_file:
                print("processing file:", filename)
                packets = rdpcap(pcap_file)

                for packet in packets:
                    if IP in packet and TCP in packet:
                        flow_key = (packet[IP].src, packet[IP].dst, packet[IP].sport, \
                                    packet[IP].dport, packet[IP].proto)

                        if flow_key not in flow_stats:
                            flow_stats[flow_key] = {
                                'packets': 0,
                                'octets': 0,
                                'duration': 0,
                                'tcp_flags': [],
                                'ip_protocols': set(),
                                'start_time': packet.time,
                                'end_time': packet.time
                            }
                            target[flow_key] = []

                        flow_stats[flow_key]['packets'] += 1
                        flow_stats[flow_key]['octets'] += len(bytes(packet))
                        flow_stats[flow_key]['start_time'] = min(packet.time, flow_stats[flow_key]['start_time'])
                        flow_stats[flow_key]['end_time'] = max(packet.time, flow_stats[flow_key]['end_time'])
                        if len(flow_stats[flow_key]['tcp_flags']) < 10:
                            flow_stats[flow_key]['tcp_flags'].append(int(packet[TCP].flags))
                        flow_stats[flow_key]['ip_protocols'].add(packet[IP].proto)

                        target[flow_key].append(len(bytes(packet)))
                    else:
                        print("packet not in IP/TCP format")

                for flow_key in flow_stats.keys():
                    flow_stats[flow_key]['tcp_flags'] += [0] * (10 - len(flow_stats[flow_key]['tcp_flags']))  # pad to 10
                    flow_stats[flow_key]['duration'] = round(flow_stats[flow_key]['end_time'] -
                                                             flow_stats[flow_key]['start_time'], 2)
                    target[flow_key] = target[flow_key] + [0] * (20 - len(target[flow_key])) \
                        if len(target[flow_key]) < 20 else target[flow_key][:20]  # truncate or pad to 20

    return flow_stats, target


def save_to_csv(datas, filename, header):
    """
        Save netflow format's data to csv file
    """
    if header is not None:
        fieldnames = header
        with open(filename, mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for key in datas.keys():
                writer.writerow(datas[key])
    else:
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([i for i in range(20)])
            for key in datas.keys():
                writer.writerow(datas[key])


def print_stats(flow_stats):
    """
        Print flow statistics
    """
    flow_num = 0
    for flow_key, stats in flow_stats.items():
        print(f"======flow: {flow_num}======")
        print("Flow:", flow_key)
        print("Packets:", stats['packets'])
        print("Octets (bytes):", stats['octets'])
        print("Duration:", stats['duration'])
        print("TCP Flags:", stats['tcp_flags'])
        print("IP Protocols:", stats['ip_protocols'])
        print("====================")
        flow_num += 1


if __name__ == "__main__":
    benign_pcap_folder = "./datasets/DoH/Benign/AdGuard"
    malicious_pcap_folder = "./datasets/DoH/Malicious/"
    benign_flow_stats, benign_target = convert_to_netflow(benign_pcap_folder, 1)
    malicious_flow_stats, malicious_target = convert_to_netflow(malicious_pcap_folder)

    header = ['packets', 'octets', 'duration', 'tcp_flags', 'ip_protocols', 'start_time', 'end_time']
    save_to_csv(benign_flow_stats, "datasets/DoH/bng_data.csv", header)
    save_to_csv(benign_target, "datasets/DoH/bng_label.csv", None)
    save_to_csv(malicious_flow_stats, "datasets/DoH/mal_data.csv", header)
    save_to_csv(malicious_target, "datasets/DoH/mal_label.csv", None)
