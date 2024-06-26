import os

import torch

import numpy as np
import pandas as pd


def compress(avg_pkt_seq, seq_len):
    """
    Compress the sequence of average packet sizes to a fixed length."""
    # need_compress = False
    # if len(avg_pkt_seq) > seq_len * 2:
    #     need_compress = True

    # if need_compress:
    #     sample_rate = len(avg_pkt_seq) // seq_len
    #     avg_pkt_seq = avg_pkt_seq[::sample_rate]
    
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
    # data = data.groupby(data.columns[0:4].tolist()).filter(lambda x: len(x) >= 2)  # filter out flows with less than 2 records
    data = data.groupby(data.columns[0:4].tolist()).apply(
        lambda x: list(zip(x['total_packets'], x['total_bytes']))).reset_index(drop=True)
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


def load_and_transform_data(data_path, seq_len=800, raw_netflow=False, type=0):
    """
    Load and transform the data to a fixed length sequence of average packet sizes."""
    data = pd.read_csv(data_path, usecols=['src_ip', 'dst_ip', 'src_port', 'dst_port', 
                                           'protocol', 'total_packets', 'total_bytes'],
                        dtype={'src_ip': str, 'dst_ip': str, 'src_port': str, 'dst_port': str,
                               'protocol': str, 'total_packets': int, 'total_bytes': int})
    print(data)

    print(f"Processing data...")
    data = process(data, seq_len, raw_netflow=raw_netflow)    
    data = torch.tensor(data.tolist(), dtype=torch.float32)
    channel = 2 if raw_netflow else 1
    data = data.reshape(data.shape[0], channel, -1)
    label = torch.full((data.shape[0],), type, dtype=torch.long)
    
    print(f"Data loaded and transformed!")
    print(data, label)

    return data, label


def load_avg_DoH(ac_t, dataset_folder):
    """
    Load the average packet sequence of DoH dataset of active timeout={ac_t}."""
    if os.path.exists(f"{dataset_folder}/tensor/avg/{ac_t}_data.pt") and \
        os.path.exists(f"{dataset_folder}/tensor/avg/{ac_t}_label.pt"):
        data_for_cls = torch.load(f"{dataset_folder}/tensor/avg/{ac_t}_data.pt")
        label_for_cls = torch.load(f"{dataset_folder}/tensor/avg/{ac_t}_label.pt")
    else:
        bng_data, bng_label = load_and_transform_data(f"{dataset_folder}/benign/{ac_t}.csv", seq_len=800, raw_netflow=False, type=0)
        dns2tcp, dns2tcp_label = load_and_transform_data(f"{dataset_folder}/dns2tcp/{ac_t}.csv", seq_len=800, raw_netflow=False, type=1)
        dnscat2, dnscat2_label = load_and_transform_data(f"{dataset_folder}/dnscat2/{ac_t}.csv", seq_len=800, raw_netflow=False, type=2)
        iodine, iodine_label = load_and_transform_data(f"{dataset_folder}/iodine/{ac_t}.csv", seq_len=800, raw_netflow=False, type=3)

        data_len = min(bng_data.shape[0], dns2tcp.shape[0], dnscat2.shape[0], iodine.shape[0])
        random_seed = 42
        torch.manual_seed(random_seed)
        indices_1, indices_2, indices_3, indices_4 = torch.randperm(bng_data.shape[0])[:data_len], \
                                                        torch.randperm(dns2tcp.shape[0])[:data_len], \
                                                        torch.randperm(dnscat2.shape[0])[:data_len], \
                                                        torch.randperm(iodine.shape[0])[:data_len]
        data_for_cls = torch.cat([bng_data[indices_1], dns2tcp[indices_2], dnscat2[indices_3], iodine[indices_4]], dim=0)
        label_for_cls = torch.cat([bng_label[indices_1], dns2tcp_label[indices_2], dnscat2_label[indices_3], iodine_label[indices_4]], dim=0)

        os.makedirs(f"{dataset_folder}/tensor/avg/", exist_ok=True)
        torch.save(data_for_cls, f"{dataset_folder}/tensor/avg/{ac_t}_data.pt")
        torch.save(label_for_cls, f"{dataset_folder}/tensor/avg/{ac_t}_label.pt")

    return data_for_cls, label_for_cls


def load_raw_DoH(ac_t, dataset_folder):
    """
    Load the raw netflow data of DoH dataset of active timeout={ac_t}."""
    if os.path.exists(f"{dataset_folder}/tensor/raw/{ac_t}_data.pt") and \
        os.path.exists(f"{dataset_folder}/tensor/raw/{ac_t}_label.pt"):
        data_for_cls = torch.load(f"{dataset_folder}/tensor/raw/{ac_t}_data.pt")
        label_for_cls = torch.load(f"{dataset_folder}/tensor/raw/{ac_t}_label.pt")
    else:
        bng_data, bng_label = load_and_transform_data(f"{dataset_folder}/benign/{ac_t}.csv", seq_len=800, raw_netflow=True, type=0)
        dns2tcp, dns2tcp_label = load_and_transform_data(f"{dataset_folder}/dns2tcp/{ac_t}.csv", seq_len=800, raw_netflow=True, type=1)
        dnscat2, dnscat2_label = load_and_transform_data(f"{dataset_folder}/dnscat2/{ac_t}.csv", seq_len=800, raw_netflow=True, type=2)
        iodine, iodine_label = load_and_transform_data(f"{dataset_folder}/iodine/{ac_t}.csv", seq_len=800, raw_netflow=True, type=3)

        data_len = min(bng_data.shape[0], dns2tcp.shape[0], dnscat2.shape[0], iodine.shape[0])
        random_seed = 42
        torch.manual_seed(random_seed)
        indices_1, indices_2, indices_3, indices_4 = torch.randperm(bng_data.shape[0])[:data_len], \
                                                        torch.randperm(dns2tcp.shape[0])[:data_len], \
                                                        torch.randperm(dnscat2.shape[0])[:data_len], \
                                                        torch.randperm(iodine.shape[0])[:data_len]
        data_for_cls = torch.cat([bng_data[indices_1], dns2tcp[indices_2], dnscat2[indices_3], iodine[indices_4]], dim=0)
        label_for_cls = torch.cat([bng_label[indices_1], dns2tcp_label[indices_2], dnscat2_label[indices_3], iodine_label[indices_4]], dim=0)

        os.makedirs(f"{dataset_folder}/tensor/raw/", exist_ok=True)
        torch.save(data_for_cls, f"{dataset_folder}/tensor/raw/{ac_t}_data.pt")
        torch.save(label_for_cls, f"{dataset_folder}/tensor/raw/{ac_t}_label.pt")

    return data_for_cls, label_for_cls     


def load_avg_CTU(dataset_folder):
    """
    Load CTU without background traffic."""
    bng_data, bng_label = load_and_transform_data(f"{dataset_folder}/bng.csv", seq_len=800, raw_netflow=False, type=0)
    mal_data, mal_label = load_and_transform_data(f"{dataset_folder}/mal.csv", seq_len=800, raw_netflow=False, type=1)

    # data_len = min(bng_data.shape[0], mal_data.shape[0])
    # random_seed = 42
    # torch.manual_seed(random_seed)
    # indices_1, indices_2 = torch.randperm(bng_data.shape[0])[:data_len], torch.randperm(mal_data.shape[0])[:data_len]
    # data_for_cls = torch.cat([bng_data[indices_1], mal_data[indices_2]], dim=0)
    # label_for_cls = torch.cat([bng_label[indices_1], mal_label[indices_2]], dim=0)
    data_for_cls = torch.cat([bng_data, mal_data], dim=0)
    label_for_cls = torch.cat([bng_label, mal_label], dim=0)

    return data_for_cls, label_for_cls


def load_avg_CTU2(dataset_folder):
    """
    Load CTU with background traffic."""
    # if os.path.exists(f"{dataset_folder}/data.pt") and os.path.exists(f"{dataset_folder}/label.pt"):
    #     data_for_cls = torch.load(f"{dataset_folder}/data.pt")
    #     label_for_cls = torch.load(f"{dataset_folder}/label.pt")
    #     return data_for_cls, label_for_cls
    
    bng_data, bng_label = load_and_transform_data(f"{dataset_folder}/bng.csv", seq_len=800, raw_netflow=False, type=0)
    mal_data, mal_label = load_and_transform_data(f"{dataset_folder}/mal.csv", seq_len=800, raw_netflow=False, type=1)
    back_data, back_label = load_and_transform_data(f"{dataset_folder}/back.csv", seq_len=800, raw_netflow=False, type=2)

    data_len = min(bng_data.shape[0], mal_data.shape[0], back_data.shape[0])
    random_seed = 42
    torch.manual_seed(random_seed)
    indices_1, indices_2, indices_3 = torch.randperm(bng_data.shape[0])[:data_len], \
                                      torch.randperm(mal_data.shape[0])[:data_len], \
                                      torch.randperm(back_data.shape[0])[:data_len]
    data_for_cls = torch.cat([bng_data[indices_1], mal_data[indices_2], back_data[indices_3]], dim=0)
    label_for_cls = torch.cat([bng_label[indices_1], mal_label[indices_2], back_label[indices_3]], dim=0)

    torch.save(data_for_cls, f"{dataset_folder}/data.pt")
    torch.save(label_for_cls, f"{dataset_folder}/label.pt")
    
    return data_for_cls, label_for_cls


def load_avg_UNSW(dataset_folder):
    bng_data, bng_label = load_and_transform_data(f"{dataset_folder}/bng.csv", seq_len=800, raw_netflow=False, type=0)
    mal_data, mal_label = load_and_transform_data(f"{dataset_folder}/mal.csv", seq_len=800, raw_netflow=False, type=1)
    
    data_len = min(bng_data.shape[0], mal_data.shape[0])
    random_seed = 42
    torch.manual_seed(random_seed)
    indices_1, indices_2 = torch.randperm(bng_data.shape[0])[:data_len], \
                            torch.randperm(mal_data.shape[0])[:data_len]
    
    data_for_cls = torch.cat([bng_data[indices_1], mal_data[indices_2]], dim=0)
    label_for_cls = torch.cat([bng_label[indices_1], mal_label[indices_2]], dim=0)

    return data_for_cls, label_for_cls