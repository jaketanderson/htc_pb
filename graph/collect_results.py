import os
import pickle

from prepare import total_count
from worker import System

results = []
for i in range(0, total_count):
    try:
        with open(f"results/{i}/result.pickle", "rb") as f:
            results.append(pickle.load(f))
    except:
        print(f"Process {i} has failed to run (results file does not exist)")

print(
    f"{len(results)}/{total_count} = {len(results)*100/total_count:0.2f}% of systems were evaluated successfully."
)

with open("all_results.pickle", "wb") as f:
    pickle.dump(results, f)
