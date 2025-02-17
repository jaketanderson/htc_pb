"""
This is a short script to check the maximum
memory usage out of all finished jobs.
"""

import re

n_jobs = 1845

mem_vals = []
for i in range(n_jobs):
    with open(f"worker_logs/{i}/log.txt", "r") as f:
        for line in f.readlines()[::-1]:
            if "Memory (MB)" in line:
                mem_val = int(re.search(r'\d+', line).group())
                mem_vals.append(mem_val)

print(max(mem_vals))
