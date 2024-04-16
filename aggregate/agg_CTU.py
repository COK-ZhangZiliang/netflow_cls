import os

from agg_DoH import *


if __name__ == '__main__':
    for i in range(13):
        file_folder = f'../datasets/CTU-13-Dataset/{i+1}'
        for file_name in os.listdir(file_folder):
            if file_name.endswith('.binetflow'):
                file_path = os.path.join(file_folder, file_name)
                break

        # Load netflows
        logging.info(f"Loading files from {file_path}...")
        netflows = pd.read_csv(file_path)
        print(netflows)
        logging.info("Having loaded netflows!")

        # Aggregate netflows to traces
        logging.info(f"Aggregating netflows to traces...")
        traces = netflows.groupby(netflows.columns[2:5].tolist()+netflows.columns[6:8].tolist()).agg(
            {'TotPkts': list, 'TotBytes': list, 'Label': list}
        ).reset_index(drop=True)
        print(traces)
        logging.info(f"Having aggregated netflows to traces!")
        traces.to_csv(f'../datasets/CTU-13-Dataset/{i+1}/traces.csv', index=False)
