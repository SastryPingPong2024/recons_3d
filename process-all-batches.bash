#!/bin/bash

SESSION_NAME="sess0"

# Define the environment setup as a string
ENV_SETUP=""

echo "Allow about 50 seconds in total for all windows to start"

# Start the session with the first window
echo "Creating window data0"
tmux new-session -d -s $SESSION_NAME -n "data0" \
"bash -c '$ENV_SETUP; python3 process-all-videos.py --exp_name data0_multi; echo \"[data0 done] Press any key to exit...\"; read'"
sleep 5

# Create the rest of the windows
for i in {1..9}
do
    echo "Creating window data$i"
    # Create a new window in the existing session
    tmux new-window -t $SESSION_NAME -n "data$i" \
    "bash -c '$ENV_SETUP; python3 process-all-videos.py --exp_name data${i}_multi; echo \"[data${i} done] Press any key to exit...\"; read'"
    sleep 5
done

# Attach to the session
tmux attach -t $SESSION_NAME
