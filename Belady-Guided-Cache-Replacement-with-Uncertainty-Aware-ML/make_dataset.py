import os
from trace_reader import read_champsim_trace
from belady_oracle import belady_dataset

script_dir = os.path.dirname(os.path.abspath(__file__))
trace_path = os.path.join(script_dir, "traces", "429.mcf-184B.trace.txt")

trace = read_champsim_trace(
    trace_path,
    max_accesses=5_000  # Start small to test
)

print(f"Loaded {len(trace)} addresses")

X, y = belady_dataset(trace, max_accesses=5_000)

print("Total samples:", len(y))
print("Evicted (0):", sum(1 for v in y if v == 0))
print("Kept (1):", sum(1 for v in y if v == 1))
