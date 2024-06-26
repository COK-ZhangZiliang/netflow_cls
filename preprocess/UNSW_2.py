# 2 classes
import pandas as pd


if __name__ == '__main__':
    pd.DataFrame(columns=['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', \
                            'total_packets', 'total_bytes', 'label']).to_csv('../datasets/UNSW/bng.csv', index=False)
    pd.DataFrame(columns=['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', \
                            'total_packets', 'total_bytes', 'label']).to_csv('../datasets/UNSW/mal.csv', index=False)

    for i in range(4):
        file_path = f'../datasets/UNSW/UNSW-NB15_{i+1}.csv'
    
        # Load data
        print(f"Loading {i+1}...")
        data = pd.read_csv(file_path, header=None)

        # Filter data
        data.rename(columns={0: 'src_ip', 1: 'src_port', 2: 'dst_ip', 3: 'dst_port', 4: 'protocol',
                             16: 'total_packets', 7: 'total_bytes', 48: 'label'}, inplace=True)
        data = data[['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'total_packets', 'total_bytes', 'label']]
        data = data[data['total_packets'] > 0]  # trick to remove anomalies

        bng_data = data[data['label'] == 0]
        mal_data = data[data['label'] == 1]

        # Save data
        bng_data.to_csv(f'../datasets/UNSW/bng.csv', mode='a', header=False, index=False)
        mal_data.to_csv(f'../datasets/UNSW/mal.csv', mode='a', header=False, index=False)