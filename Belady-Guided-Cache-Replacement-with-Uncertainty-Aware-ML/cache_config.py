# Cache parameters
CACHE_SIZE = 32 * 1024        # 32 KB cache (restored)
BLOCK_SIZE = 64               # 64-byte cache line
ASSOCIATIVITY = 8             # 8-way set associative

NUM_BLOCKS = CACHE_SIZE // BLOCK_SIZE
NUM_SETS = NUM_BLOCKS // ASSOCIATIVITY
