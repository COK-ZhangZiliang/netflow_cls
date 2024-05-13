import re

import torch

import numpy as np
import pandas as pd


def convert_m_to_number(x):
    """
    Convert 'x.x M' to number."""
    if isinstance(x, str): 
        match = re.search(r'(\d+(\.\d+)?)M', x)
        if match:
            return float(match.group(1)) * 1000000
    return x


def compress(avg_pkt_seq, seq_len):
    """
    Compress the sequence of average packet sizes to a fixed length."""
    need_compress = False
    if len(avg_pkt_seq) > seq_len * 2:
        need_compress = True

    if need_compress:
        sample_rate = len(avg_pkt_seq) // seq_len
        avg_pkt_seq = avg_pkt_seq[::sample_rate]
    
    avg_pkt_seq = avg_pkt_seq[:seq_len] if len(avg_pkt_seq) >= seq_len \
        else avg_pkt_seq + [0] * (seq_len - len(avg_pkt_seq))

    return avg_pkt_seq


def get_raw_netflow(record):
    """
    Get the raw netflow data from the record."""
    num_packets = [int(counts) for counts, _ in record]
    num_bytes = [int(bytes) for _, bytes in record]

    return [num_packets, num_bytes]


def process(data, seq_len, raw_netflow=False):
    """
    Process the data to a fixed length sequence of average packet sizes."""
    data = data.groupby(data.columns[0:4].tolist()).apply(
        lambda x: list(zip(x[5], x[6]))).reset_index(drop=True)
    print(data)

    if not raw_netflow:
        data = data.apply(lambda x: [[int(bytes)//int(counts)]*int(counts)
                                        for counts, bytes in x])
        data = data.apply(lambda x: [item for sublist in x for item in sublist])
        data = data.apply(lambda x: compress(x, seq_len))
    else:
        data = data.apply(lambda x: get_raw_netflow(x))
        data = data.apply(lambda x: [compress(x[0], seq_len), compress(x[1], seq_len)])

    return data


def load_and_transform_data(bng_data_path, mal_data_path, seq_len=800, raw_netflow=False):
    """
    Load and transform the data to a fixed length sequence of average packet sizes."""
    bng_data, mal_data = pd.read_csv(bng_data_path, header=None), pd.read_csv(mal_data_path, header=None)
    bng_data, mal_data = bng_data.iloc[:, 1:], mal_data.iloc[:, 1:]
    bng_data.columns, mal_data.columns = range(bng_data.shape[1]), range(mal_data.shape[1])
    bng_data, mal_data = bng_data.applymap(convert_m_to_number), mal_data.applymap(convert_m_to_number)
    print(bng_data, mal_data)

    print(f"Processing benign data...")
    bng_data = process(bng_data, seq_len, raw_netflow=raw_netflow)    

    print(f"Processing malicious data...")
    mal_data = process(mal_data, seq_len, raw_netflow=raw_netflow)

    data_len = min(len(bng_data), len(mal_data))
    data_for_cls = pd.concat([bng_data.sample(data_len), mal_data.sample(data_len)], axis=0)
    data_for_cls = np.array(data_for_cls.tolist())
    data_for_cls = torch.tensor(data_for_cls, dtype=torch.int32)
    channel = 2 if raw_netflow else 1
    data_for_cls = data_for_cls.reshape((2 * data_len, channel, -1))
    label_for_cls = torch.tensor([0] * data_len + [1] * data_len, dtype=torch.int32).reshape(-1)
    
    print(f"Data loaded and transformed!")
    print(data_for_cls, label_for_cls)

    return data_for_cls, label_for_cls