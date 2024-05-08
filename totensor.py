import pandas as pd
import numpy as np

import torch

def load_and_transform_data(data_path, data_len=1000):
    data = pd.read_csv(data_path)
    print(len(data))
    print("Loading data...")
    data = data.iloc[:, 0].apply(lambda x: eval(x))
    data = data.apply(lambda x: [[bytes//counts//10+1]*counts 
                                         for counts, bytes in x])
    data = data.apply(lambda x: [item for sublist in x for item in sublist])
    data = data.apply(lambda x: x[:data_len] if len(x) >= data_len else x + [0] * (data_len - len(x)))
    data = torch.tensor(np.array(data.tolist()), dtype=torch.float32)
    data = data.reshape((len(data), 1, -1))
    print("Data loaded and transformed!")

    print(data)
    return data

if __name__ == '__main__':
    sample_rate = ['inf']
    for rate in sample_rate:
        data = load_and_transform_data(f'./datasets/pretrain/traces/traces_{rate}.csv', 100)
        torch.save(data, f'./datasets/pretrain/traces/traces_{rate}_2.pt')