import csv
import os

from collections import defaultdict


class NetFlowRecord:
    """
    Class to represent a NetFlow record."""
    def __init__(self, packet, timeout):
        self.start_time = float(packet['frame.time_epoch'])
        self.end_time = float(packet['frame.time_epoch'])
        self.total_packets = 1
        self.total_bytes = int(packet['ip.len'])
        self.tos = packet['ip.tos'] if packet['ip.tos'] else 0
        self.tcp_flags = int(packet['tcp.flags'], 16) if packet['tcp.flags'] else 0
        self.timeout = timeout

    def update(self, packet):
        self.end_time = float(packet['frame.time_epoch'])
        self.total_packets += 1
        self.total_bytes += int(packet['ip.len'])
        self.tcp_flags = self.tcp_flags | int(packet['tcp.flags'], 16) if packet['tcp.flags'] else self.tcp_flags

    def duration(self):
        return self.end_time - self.start_time


def aggregate_netflow(csv_file, active_timeout):
    """
    Aggregate NetFlow records from a CSV file."""
    flows = defaultdict(list)
    packets = []

    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            packets.append(row)

    packets.sort(key=lambda x: float(x['frame.time_epoch']))  # Sort all packets by its' timestamp

    for pkt in packets:
        key = (pkt['ip.src'], pkt['ip.dst'], pkt['tcp.srcport'], pkt['tcp.dstport'], pkt['ip.proto'])
        packet_time = float(pkt['frame.time_epoch'])

        if flows[key]:  # flow already exists
            if packet_time < flows[key][-1].timeout:
                flows[key][-1].update(pkt)
            else:  # a new netflow record for this flow
                timeout = int(packet_time) + active_timeout
                new_record = NetFlowRecord(pkt, timeout)
                flows[key].append(new_record)
        else:
            timeout = int(packet_time) + active_timeout
            new_record = NetFlowRecord(pkt, timeout)
            flows[key].append(new_record)

    # Dump all netflow records to a list
    netflow_records = []
    for key, records in flows.items():
        for record in records:
            src_ip, dst_ip, src_port, dst_port, protocol = key
            netflow_records.append({
                'duration': record.duration(),
                'src_ip': src_ip,
                'dst_ip': dst_ip,
                'src_port': src_port,
                'dst_port': dst_port,
                'protocol': protocol,
                'total_packets': record.total_packets,
                'total_bytes': record.total_bytes,
                'tos': record.tos,
                'tcp_flags': record.tcp_flags
            })

    return netflow_records


def save_netflow_to_csv(netflow_records, output_file):
    """
    Save NetFlow records to a CSV file."""
    with open(output_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=[
            'duration', 'src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol',
            'total_packets', 'total_bytes', 'tos', 'tcp_flags'
        ])
        writer.writeheader()
        writer.writerows(netflow_records)


def process_pcap_files(input_files, combined_csv, active_timeout, type):
    """
    Process PCAP files and generate NetFlow records."""
    with open(combined_csv, 'w', newline='') as outfile:
        writer = None
        for pcap_file in input_files:
            tmp_file = f"../datasets/DoH/tmp.csv"
            command = f"tshark -r {pcap_file} -T fields -e frame.time_epoch \
                -e ip.src -e ip.dst -e tcp.srcport -e tcp.dstport -e ip.proto \
                    -e ip.len -e ip.tos -e tcp.flags -E header=y -E separator=, -E quote=d -E occurrence=f > {tmp_file}"
            print(f"Processing {pcap_file}...")
            os.system(command)

            with open(tmp_file, 'r') as infile:
                reader = csv.DictReader(infile)
                if writer is None:
                    writer = csv.DictWriter(outfile, fieldnames=reader.fieldnames)
                    writer.writeheader()
                for row in reader:
                    writer.writerow(row)
            
    os.remove(tmp_file)

    for ac_t in active_timeout:
        netflow_records = aggregate_netflow(combined_csv, ac_t)
        os.makedirs(f'../datasets/DoH/nfv5/{type}', exist_ok=True)
        save_netflow_to_csv(netflow_records, f'../datasets/DoH/nfv5/{type}/{ac_t}.csv')


def main():
    bng_pcap_folder = '../datasets/DoH/benign'
    mal_pcap_folder = '../datasets/DoH/malicious'
    
    bng_pcap_files = [os.path.join(bng_pcap_folder, f) for f in os.listdir(bng_pcap_folder) if f.endswith('.pcap')]
    mal_pcap_files_dns2tcp = [os.path.join(mal_pcap_folder, f) for f in os.listdir(mal_pcap_folder) \
                              if f.startswith('dns2tcp') and f.endswith('.pcap')]
    mal_pcap_files_dnscat2 = [os.path.join(mal_pcap_folder, f) for f in os.listdir(mal_pcap_folder) \
                              if f.startswith('dsncat2') and f.endswith('.pcap')]
    mal_pcap_files_iodine = [os.path.join(mal_pcap_folder, f) for f in os.listdir(mal_pcap_folder) \
                             if f.startswith('iodine') and f.endswith('.pcap')]
    assert len(mal_pcap_files_dns2tcp) + len(mal_pcap_files_dnscat2) + len(mal_pcap_files_iodine) \
        == len(os.listdir(mal_pcap_folder))  # check if all files are covered
    
    bng_combined_csv = '../datasets/DoH/bng_combined.csv'
    mal_combined_dns2tcp_csv = '../datasets/DoH/mal_combined_dns2tcp.csv'
    mal_combined_dnscat2_csv = '../datasets/DoH/mal_combined_dnscat2.csv'
    mal_combined_iodine_csv = '../datasets/DoH/mal_combined_iodine.csv'
    active_timeout = [1, 5, 10, 15, 20, 25, 30, 60]  # active timeout in seconds

    # process_pcap_files(bng_pcap_files, bng_combined_csv, active_timeout, type='benign')
    # process_pcap_files(mal_pcap_files_dns2tcp, mal_combined_dns2tcp_csv, active_timeout, type='dns2tcp')
    process_pcap_files(mal_pcap_files_dnscat2, mal_combined_dnscat2_csv, active_timeout, type='dnscat2')
    process_pcap_files(mal_pcap_files_iodine, mal_combined_iodine_csv, active_timeout, type='iodine')


if __name__ == "__main__":
    main()
