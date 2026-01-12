"""
Utility functions for cache simulation
"""

from cache_config import BLOCK_SIZE, NUM_SETS

def get_block_address(address):
    """Get block-aligned address"""
    return (address // BLOCK_SIZE) * BLOCK_SIZE

def get_set_index(address):
    """Get cache set index for an address"""
    block_addr = address // BLOCK_SIZE
    return block_addr % NUM_SETS

def get_tag(address):
    """Get tag for an address"""
    block_addr = address // BLOCK_SIZE
    return block_addr // NUM_SETS
