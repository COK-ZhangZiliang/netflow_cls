import os
import csv
import pyshark
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def parse_single_pcap(filename):
    """
    Parse a single pcap file and return flows.
    """
    logging.info(f"Parsing {filename}...")
    with pyshark.FileCapture(filename, only_summaries=False, keep_packets=False) as cap:
        flows = defaultdict(list)
        for pkt in cap:
            try:
                src_ip = pkt.ip.src
                dst_ip = pkt.ip.dst
                src_port = pkt[pkt.transport_layer].srcport
                dst_port = pkt[pkt.transport_layer].dstport
                protocol = pkt.transport_layer
                flow_key = (src_ip, dst_ip, src_port, dst_port, protocol)
                pkt_time = datetime.utcfromtimestamp(float(pkt.sniff_timestamp))
                flows[flow_key].append((pkt_time, int(pkt.length)))
            except AttributeError:
                continue
    return flows


def parse_pcap(pcap_folder, num_files=None, max_workers=4):
    """
    Parse pcap files in the folder and return a dictionary of flows using multiple threads.
    """
    filenames = []
    for f in os.listdir(pcap_folder):
        if f.endswith(".pcap"):
            filenames.append(os.path.join(pcap_folder, f))
    if num_files is not None:
        filenames = filenames[:(len(filenames) // 2)]

    flows = defaultdict(list)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_filename = {executor.submit(parse_single_pcap, filename): filename for filename in filenames}
        for future in as_completed(future_to_filename):
            filename = future_to_filename[future]
            try:
                flow = future.result()
                # Combine flows from different files
                for key, value in flow.items():
                    flows[key].extend(value)
            except Exception as exc:
                logging.info(f"{filename} generated an exception: {exc}")

    return flows


def flows_to_traces(flows, timewindow=1):
    """
    Aggregate flows into traces_2 with a time window of 'timewindow'.
    """
    traces = []
    for flow_key, packets in flows.items():
        packets.sort(key=lambda x: x[0])
        start_time = packets[0][0]
        end_time = start_time + timedelta(seconds=timewindow)
        packet_count = 0
        byte_count = 0
        for pkt_time, pkt_length in packets:
            if pkt_time > end_time:
                traces.append((*flow_key, start_time, end_time, packet_count, byte_count))
                start_time = pkt_time
                end_time = start_time + timedelta(seconds=timewindow)
                packet_count = 0
                byte_count = 0
            packet_count += 1
            byte_count += pkt_length
        traces.append((*flow_key, start_time, end_time, packet_count, byte_count))
    return traces


def save_to_csv(traces, csv_file):
    """
    Save traces to a csv file.
    """
    with open(csv_file, mode='w', newline='') as file:
        logging.info(f"Saving to {csv_file}...")
        writer = csv.writer(file)
        writer.writerow(
            ['Source IP', 'Destination IP', 'Source Port', 'Destination Port', 'Protocol', 'Start Time', 'End Time',
             'Packet Count', 'Byte Count'])
        for trace in traces:
            writer.writerow(trace)


if __name__ == '__main__':
    benign_pcap_folder = "./datasets/benign"
    malicious_pcap_folder = "./datasets/malicious"
    timewindows = [0.1, 1, 5, 10, 15, 20, 25, 30]
    mal_flows = parse_pcap(malicious_pcap_folder, max_workers=64)
    for timewindow in timewindows:
        mal_traces = flows_to_traces(mal_flows, timewindow)
        save_to_csv(mal_traces, f"./datasets/traces/mal_{timewindow}.csv")
    bng_flows = parse_pcap(benign_pcap_folder, max_workers=32)
    for timewindow in timewindows:
        bng_traces = flows_to_traces(bng_flows, timewindow)
        save_to_csv(bng_traces, f"datasets/traces/bng_{timewindow}.csv")