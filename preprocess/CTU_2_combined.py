# 2 classes & combined dataset
import os
import pandas as pd


if __name__ == '__main__':
    bng_file = f'../datasets/CTU-13-Dataset/bng.csv'
    mal_file = f'../datasets/CTU-13-Dataset/mal.csv'
    pd.DataFrame(columns=['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', \
                            'total_packets', 'total_bytes', 'label']).to_csv(bng_file, index=False)
    pd.DataFrame(columns=['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', \
                            'total_packets', 'total_bytes', 'label']).to_csv(mal_file, index=False)


    for i in range(13):
        file_folder = f'../datasets/CTU-13-Dataset/{i+1}'
        for file_name in os.listdir(file_folder):
            if file_name.endswith('.binetflow'):
                file_path = os.path.join(file_folder, file_name)
                break

        # Load data
        print(f"Loading {i+1}...")
        data = pd.read_csv(file_path)

        # Filter data
        data = data[data['Label'].str.contains('From-Botnet|From-Normal|To-Botnet|To-Normal')]
        data.rename(columns={'SrcAddr': 'src_ip', 'DstAddr': 'dst_ip',
                             'Sport': 'src_port', 'Dport': 'dst_port', 'Proto': 'protocol',
                             'TotPkts': 'total_packets', 'TotBytes': 'total_bytes',
                             'Label': 'label'}, inplace=True)
        data['label'] = data['label'].apply(lambda x: i+1 if 'From-Botnet' in x else 0)
        data = data[['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'total_packets', 'total_bytes', 'label']]

        bng_data = data[data['label'] == 0]
        mal_data = data[data['label'] == i+1]
        assert len(bng_data) + len(mal_data) == len(data)

        # Save data
        bng_data.to_csv(f'../datasets/CTU-13-Dataset/bng.csv', mode='a', header=False, index=False)
        mal_data.to_csv(f'../datasets/CTU-13-Dataset/mal.csv', mode='a', header=False, index=False)
        
        print(f"{i+1} done!")
        