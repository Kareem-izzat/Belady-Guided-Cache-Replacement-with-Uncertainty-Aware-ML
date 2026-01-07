import math
from cache_config import BLOCK_SIZE, NUM_SETS

BLOCK_OFFSET_BITS = int(math.log2(BLOCK_SIZE))
SET_INDEX_BITS = int(math.log2(NUM_SETS))

def get_block_address(addr):
    return addr >> BLOCK_OFFSET_BITS

def get_set_index(addr):
    return (addr >> BLOCK_OFFSET_BITS) & (NUM_SETS - 1)

def get_tag(addr):
    return addr >> (BLOCK_OFFSET_BITS + SET_INDEX_BITS)
