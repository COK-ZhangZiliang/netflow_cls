import pandas as pd
import numpy as np

import torch

def load_and_transform_data(data_path, data_lens=800):
    if type(data_lens) == int:
        data_lens = [data_lens]
    data = pd.read_csv(data_path)
    print(len(data))
    print("Loading data...")
    data = data.iloc[:, 0].apply(lambda x: eval(x))
    data = data.apply(lambda x: [[bytes//counts]*counts 
                                         for counts, bytes in x])
    data = data.apply(lambda x: [item for sublist in x for item in sublist])

    for data_len in data_lens:
        new_data = data.apply(lambda x: x[:data_len] if len(x) >= data_len else x + [0] * (data_len - len(x)))
        new_data = torch.tensor(np.array(new_data.tolist()), dtype=torch.float32)
        new_data = new_data.reshape((len(data), 1, -1))

        print("Data loaded and transformed!")
        print(new_data)

        torch.save(new_data, f'./datasets/pretrain/traces/{data_len}.pt')


if __name__ == '__main__':
    data_lens = [100, 200, 400, 800, 1600]
    data = load_and_transform_data(f'./datasets/pretrain/traces/traces_inf.csv', data_lens)
        