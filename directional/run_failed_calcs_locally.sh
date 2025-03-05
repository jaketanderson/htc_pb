#!/bin/bash

total_count=$(python3 -c 'import prepare; print(prepare.total_count)')

for ((i=1; i<=total_count; i++)); do
    file="results/${i}/result.pickle"
    
    if [[ ! -f "$file" ]]; then
        echo "$file not found. Performing calculation."
	cp worker_logs/$i/system_input.pickle .; rm -r working_data
	python worker.py $i
    fi
done
