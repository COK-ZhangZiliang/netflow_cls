#!/bin/bash

# Description: This script processes all pcap files in the specified folder
# and converts them to netflow files in CSV format.

# set parameters
WORKDIR="../datasets/DoH/malicious"
ACTIVE_TIMEOUT=(60 120 180 240 300 360)
INACTIVE_TIMEOUT=60
NETFLOW_DIR="../datasets/DoH/tmp"
TARGET_DIR="../datasets/DoH/nfv5/malicious"
OUTPUT_FORMAT="fmt:%td,%sa,%da,%sp,%dp,%pr,%ipkt,%ibyt"

mkdir -p $TARGET_DIR

for act_timeout in ${ACTIVE_TIMEOUT[@]}; do
    TARGET_FILE="$TARGET_DIR/$act_timeout.csv"
    # clear the target file
    > "$TARGET_FILE"

    # process pcap files
    for pcap in $WORKDIR/*.pcap; do
        mkdir -p $NETFLOW_DIR

        echo "Processing $pcap..."
        nfpcapd -r $pcap -w $NETFLOW_DIR -e $act_timeout,$INACTIVE_TIMEOUT
        nfdump -R "$NETFLOW_DIR" -q -o "$OUTPUT_FORMAT" >> "$TARGET_FILE"
        echo "Done."

        # clean up
        rm -rf $NETFLOW_DIR
    done

done
