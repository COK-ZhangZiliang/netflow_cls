# Description: This script parses all pcap files in the 
# specified folder and extracts the specified fields
PCAP_FOLDER="../datasets/benign"
OUTPUT_FILE="../datasets/benign.txt"

# check if tshark is installed
if ! command -v tshark &> /dev/null
then
    echo "tshark could not be found, please install it first."
    exit 1
fi

# clear the output file
> "$OUTPUT_FILE"
echo "ip.src,ip.dst,tcp.srcport,tcp.dstport,frame.time_epoch,ip.len" >> "$OUTPUT_FILE"

# search for all pcap files in the folder and parse them
for pcap in "$PCAP_FOLDER"/*.pcap
do
    echo "Processing file: $pcap"
    # extract the fields and append them to the output file
    tshark -r "$pcap" -Y "tcp" -T fields \
        -e ip.src -e ip.dst \
        -e tcp.srcport -e tcp.dstport \
        -e frame.time_epoch \
        -e ip.len \
        -E separator=, -E occurrence=f >> "$OUTPUT_FILE"
done

echo "All files have been processed."
