import os
import csv
import logging
import json
from collections import defaultdict

import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')


def sort_and_return(x, y, z):
    """
    Sort x and y and return them.
    """
    combined = sorted(zip(x, zip(y, z)), key=lambda pair: pair[0])
    sorted_x, sorted_yz = zip(*combined)
    sorted_y, sorted_z = zip(*sorted_yz)
    return list(sorted_x), list(sorted_y), sorted_z[0]


def read_data_and_labels(data_path):
    """
    Read data and labels from the given path.
    """
    label_path = data_path.rsplit('.', 1)[0] + '.label'

    logging.info(f"Reading data from {data_path.split('/')[-1]}...")
    data = []
    with open(data_path, 'r') as file:
        for line in file:
            line = list(map(int, line.strip().split()))
            data.append(line)
    data = pd.DataFrame(data)

    with open(label_path, 'r') as file:
        labels = file.read().strip()
    labels = list(map(int, list(labels)))

    # assert that the length of the labels is the same as the length of the data
    assert len(labels) == len(data), \
        f"Length of labels {len(labels)} is not equal to the length of data {len(data)}"
    data['label'] = labels

    
    return data


def flows_to_traces(flows, sample_rate):
    """
    Convert flows to traces.
    """
    traces = defaultdict(lambda: [[], 0])

    if sample_rate == 'inf':  # packets' length sequence
        for idx, flow in flows.iterrows():
            for packet in flow['ip.len']:
                traces[idx][0].append((1, packet))
            traces[idx][1] = flow['label']
        return traces

    sample_rate*=10**6
    for idx, flow in flows.iterrows():
        start_time = flow['frame.time_epoch'][0]
        end_time = start_time + sample_rate
        packets_num, bytes_num = 0, 0
        for time_stamp, packet_len in zip(flow['frame.time_epoch'], flow['ip.len']):
            if time_stamp > end_time:
                traces[idx][0].append((packets_num, bytes_num))
                while time_stamp > end_time:
                    start_time = end_time
                    end_time = start_time + sample_rate
                packets_num, bytes_num = 0, 0
            packets_num += 1
            bytes_num += packet_len
        traces[idx][0].append((packets_num, bytes_num))
        traces[idx][1] = flow['label']
    
    return traces


def save_to_csv(traces, csv_file):
    """
    Save traces to a csv file.
    """
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    is_exist = os.path.exists(csv_file)
    with open(csv_file, mode='a', newline='') as file:
        logging.info(f"Saving to {csv_file}...")
        writer = csv.writer(file)
        if not is_exist:
            writer.writerow(['flow idx', 'traces'])
        for trace in traces.items():
            writer.writerow(trace)


if __name__ == '__main__':
    sub_datasets = ["brute", "lrscan", "web", "misc", "malware"]
    sample_rate = ['inf', 1, 5, 10, 15, 20, 25, 30, 60, 120, 180]
    data_dir = "../datasets/H_V"

    for sub_dataset in sub_datasets:
        data_paths = []
        
        # for file in os.listdir(data_dir):
        #     if file.endswith('.data'):
        #         data_paths.append(os.path.join(data_dir, file))
        with open(f'../datasets/H_V/config.json', 'r') as file:
            config = json.load(file)
            for file_name in config[sub_dataset].keys():
                data_paths.append(f'{data_dir}/{file_name}.data')
        print(data_paths)
        os.makedirs(f'../datasets/H_V/traces2/{sub_dataset}', exist_ok=True)  

        for data_path in data_paths:
            # Load the files
            data = read_data_and_labels(data_path)
            logging.info("Having loaded the data!")

            # Convert packets to flows
            logging.info("Converting packets to flows...")
            data = data.groupby(data.columns[1:5].tolist()).filter(lambda x: len(x) >= 10)
            flows = data.groupby(data.columns[1:5].tolist()).apply(
            lambda g: pd.Series(sort_and_return(g.iloc[:, 5], g.iloc[:, 7], g['label']), 
                                index=['frame.time_epoch', 'ip.len', 'label'])
            ).reset_index(drop=True)
            print(flows)
            logging.info("Having converted packets to flows!")
  
            # Convert flows to traces for different sample rates
            for rate in sample_rate:
                logging.info(f"Converting flows to traces with sample rate {rate}...")
                traces = flows_to_traces(flows, rate)
                save_to_csv(traces, f'../datasets/H_V/traces2/{sub_dataset}/traces_{rate}.csv')
