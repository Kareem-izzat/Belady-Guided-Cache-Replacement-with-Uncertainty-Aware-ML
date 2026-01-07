import os

MIN_VALID_ADDR = 0x1000  # ignore tiny addresses (stack markers, null, etc.)

def read_champsim_trace(filename, max_accesses=1_000_000):
    addresses = []

    with open(filename, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_accesses:
                break

            parts = line.strip().split()
            if len(parts) != 2:
                continue

            op, addr_str = parts

            try:
                addr = int(addr_str, 16)
            except ValueError:
                continue

            # filter meaningless addresses
            if addr < MIN_VALID_ADDR:
                continue

            addresses.append(addr)

    return addresses


if __name__ == "__main__":
    
    script_dir = os.path.dirname(os.path.abspath(__file__))

    trace_path = os.path.abspath(os.path.join(
        script_dir,
        "traces",
        "403.gcc-16B.trace.txt"
    ))

    print("Looking for trace at:", trace_path)
    print("File exists:", os.path.exists(trace_path))


    trace = read_champsim_trace(trace_path, 10)
    print("Found", len(trace), "addresses:")
    for a in trace:
        print(hex(a))
