#!/bin/bash

echo 'Wait for all nodes to join the Ray cluster'
total_nodes=$1

while true; do
    alive_nodes=$(ray list nodes -f state=ALIVE | awk "/Total:/ {print \$2}")
    if ! [[ "$alive_nodes" =~ ^[0-9]+$ ]]; then
    alive_nodes=0
    fi
    echo "Waiting for all nodes to join [$alive_nodes/$total_nodes]"
    if [ "$alive_nodes" -ge "$total_nodes" ]; then
        break
    fi
    sleep 5
done

echo 'All nodes joined the Ray cluster - continuing with the job submission' 
