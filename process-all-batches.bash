#!/bin/bash

SESSION_NAME="sess_batches"

# Define the environment setup as a string
ENV_SETUP=""

echo "Allow about 260 seconds in total for all windows to start"

# Start the session with the first window
echo "Creating window batch1"
tmux new-session -d -s $SESSION_NAME -n "batch1" \
"bash -c 'python3 process-all-videos.py --gpu 1 --folder batch_clipped_long_1; echo \"[batch1 done] Press any key to exit...\"; read'"
sleep 5

# Create the rest of the windows
for i in {2..52}
do
    echo "Creating window batch$i"
    # Create a new window in the existing session
    tmux new-window -t $SESSION_NAME -n "batch$i" \
    "bash -c 'python3 process-all-videos.py --gpu ${i} --folder batch_clipped_long_${i}; echo \"[batch${i} done] Press any key to exit...\"; read'"
    sleep 5
done

# Attach to the session
tmux attach -t $SESSION_NAME
