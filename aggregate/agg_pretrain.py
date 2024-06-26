import os
import csv
import logging

import pandas as pd
import numpy as np
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def sort_and_return(x, y):
    """
    Sort x and y and return them.
    """
    combined = sorted(zip(x, y), key=lambda pair: pair[0])
    sorted_x, sorted_y = zip(*combined)
    return list(sorted_x), list(sorted_y)


def flows_to_traces(flows, sample_rate):
    """
    Convert flows to traces.
    """
    traces = defaultdict(list)

    if sample_rate == 'inf':  # packets' length sequence
        for idx, flow in enumerate(flows['ip.len']):
            for packet in flow:
                traces[idx].append((1, packet))
        return traces

    for idx, flow in enumerate(zip(flows['frame.time_epoch'], flows['ip.len'])):
        start_time = flow[0][0]
        end_time = start_time + sample_rate
        packets_num, bytes_num = 0, 0
        for time_stamp, packet_len in zip(flow[0], flow[1]):
            if time_stamp > end_time:
                traces[idx].append((packets_num, bytes_num))
                while time_stamp > end_time:
                    start_time = end_time
                    end_time = start_time + sample_rate
                packets_num, bytes_num = 0, 0
            packets_num += 1
            bytes_num += packet_len
        traces[idx].append((packets_num, bytes_num))
    return traces


def save_to_csv(traces, csv_file):
    """
    Save traces to a csv file.
    """
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    with open(csv_file, mode='w', newline='') as file:
        logging.info(f"Saving to {csv_file}...")
        writer = csv.writer(file)
        writer.writerow(['traces'])
        for trace in traces.items():
            writer.writerow([trace[1]])


if __name__ == '__main__':
    sample_rate = ['inf', 0.1, 1, 5, 10, 15, 20, 25, 30, 60, 120, 180]

    # Load the files
    file = "../datasets/pretrain/pretrain.txt"
    logging.info("Loading files...")
    packets = pd.read_csv(file)
    logging.info("Having loaded packets!")

    # Convert packets to flows
    logging.info("Converting packets to flows...")
    packets = packets.groupby(packets.columns[0:4].tolist()).filter(lambda x: len(x) >= 10)
    flows = packets.groupby(packets.columns[0:4].tolist()).apply(
    lambda g: pd.Series(sort_and_return(g['frame.time_epoch'], g['ip.len']), 
                        index=['frame.time_epoch', 'ip.len'])
    ).reset_index(drop=True)
    print(flows)
    logging.info("Having converted packets to flows!")

    # Convert flows to traces for different sample rates
    for rate in sample_rate:
        logging.info(f"Converting flows to traces with sample rate {rate}...")
        traces = flows_to_traces(flows, rate)
        save_to_csv(traces, f'../datasets/pretrain/traces/traces_{rate}.csv')
