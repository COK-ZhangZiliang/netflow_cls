import os
import csv
import json

import pandas as pd

from DoH import aggregate_netflow


def save_netflow_to_csv(netflow_records, output_file):
    """
    Save NetFlow records to a CSV file."""
    fieldnames = ['duration', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol',
                'total_packets', 'total_bytes', 'tos', 'tcp_flags']
    
    if not os.path.exists(output_file):
        with open(output_file, 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()

    with open(output_file, 'a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writerows(netflow_records)


def process_data_files(input_files, active_timeout, type):
    """
    Process data files and save them to a CSV file."""
    for data in input_files:
        label = data.rsplit('.', 1)[0] + '.label'
        
        print(f"Processing {data}...")
        tmp = []
        with open(data, 'r') as file:
            for line in file:
                fields = line.strip().split()
                tmp.append(fields)
        data = pd.DataFrame(tmp)
        with open(label, 'r') as file:
            labels = file.read().strip()
            labels = list(map(int, list(labels)))
        
        assert len(labels) == len(data), \
            f"Length of labels {len(labels)} is not equal to the length of data {len(data)}"

        proto = [(1 << 4) + (1 << 5) + (1 << 6) + (1 << 7), (1 << 8)]  # bit mask for TCP and UDP
        data['label'] = labels
        data = data.dropna()
        data = data.astype({0: 'str', 1: 'str', 2: 'str', 3: 'int', 4: 'int', 5: 'float', 6: 'int', 7: 'int'})

        data['ip.src'] = data[1]
        data['ip.dst'] = data[2]
        data['tcp.srcport'] = data[3]
        data['tcp.dstport'] = data[4]
        data['frame.time_epoch'] = data[5].apply(lambda x: x / 10**6)  # convert to seconds
        data['ip.proto'] = data[6].apply(lambda x: 6 if x & proto[0] else 17 if x & proto[1] else 0)
        data['ip.len'] = data[7]
        data = data[['frame.time_epoch', 'ip.src', 'ip.dst', 'tcp.srcport', \
                     'tcp.dstport', 'ip.proto', 'ip.len', 'label']]  # select relevant columns
        data = data[data['ip.proto'] != 0]  # remove unknown packets
        data['ip.tos'] = 0
        data['tcp.flags'] = 0
        print(data)

        bng_tmp = '../datasets/H_V/bng_tmp.csv'
        mal_tmp = '../datasets/H_V/mal_tmp.csv'
        with open(bng_tmp, 'w', newline='') as file:
            bng_data = data[data['label'] == 0]
            bng_data.to_csv(file, index=False, header=True)
        with open(mal_tmp, 'w', newline='') as file:
            mal_data = data[data['label'] == 1]
            mal_data.to_csv(file, index=False, header=True)

        for ac_t in active_timeout:
            bng_netflow_records = aggregate_netflow(bng_tmp, ac_t)
            os.makedirs(f'../datasets/H_V/nfv5/benign', exist_ok=True)
            save_netflow_to_csv(bng_netflow_records, f'../datasets/H_V/nfv5/benign/{ac_t}.csv')
            mal_netflow_records = aggregate_netflow(mal_tmp, ac_t)
            os.makedirs(f'../datasets/H_V/nfv5/{type}', exist_ok=True)
            save_netflow_to_csv(mal_netflow_records, f'../datasets/H_V/nfv5/{type}/{ac_t}.csv')
        
        os.remove(bng_tmp)
        os.remove(mal_tmp)
            

def get_all_files(data_dir, type):
    data_paths = []
    with open(f'../datasets/H_V/config.json', 'r') as file:
        config = json.load(file)
        for file_name in config[type].keys():
            data_paths.append(f'{data_dir}/{file_name}.data')

    return data_paths


def main():
    data_dir = "../datasets/H_V"
    active_timeout = [1, 5, 10, 15, 20, 25, 30, 60]  # active timeout in seconds

    brute_files = get_all_files(data_dir, "brute")
    lrscan_files = get_all_files(data_dir, "lrscan")
    web_files = get_all_files(data_dir, "web")
    misc_files = get_all_files(data_dir, "misc")
    malware_files = get_all_files(data_dir, "malware")

    process_data_files(brute_files, active_timeout, type='brute')
    process_data_files(lrscan_files, active_timeout, type='lrscan')
    process_data_files(web_files, active_timeout, type='web')
    process_data_files(misc_files, active_timeout, type='misc')
    process_data_files(malware_files, active_timeout, type='malware')
    

if __name__ == "__main__":
    main()
