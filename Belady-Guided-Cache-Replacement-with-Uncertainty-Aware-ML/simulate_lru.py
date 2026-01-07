import os
from trace_reader import read_champsim_trace
from cache_lru import CacheLRU

script_dir = os.path.dirname(os.path.abspath(__file__))
trace_path = os.path.join(script_dir, "traces", "429.mcf-184B.trace.txt")

trace = read_champsim_trace(
    trace_path,
    max_accesses=1_000_000
)

cache = CacheLRU()

for addr in trace:
    cache.access(addr)

print(cache.stats())
