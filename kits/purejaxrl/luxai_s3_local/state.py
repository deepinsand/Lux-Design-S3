import functools
import chex
import flax
import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from luxai_s3.params import MAP_TYPES, EnvParams
from luxai_s3.utils import to_numpy
EMPTY_TILE = 0
NEBULA_TILE = 1
ASTEROID_TILE = 2

ENERGY_NODE_FNS = [
    lambda d, x, y, z: jnp.sin(d * x + y) * z, lambda d, x, y, z: (x / (d + 1) + y) * z
]

@struct.dataclass
class UnitState:
    position: chex.Array
    """Position of the unit with shape (2) for x, y"""
    energy: chex.Array #int
    """Energy of the unit"""

@struct.dataclass
class MapTile:
    energy: chex.Array #int
    """Energy of the tile, generated via energy_nodes and energy_node_fns"""
    tile_type: chex.Array #int
    """Type of the tile"""

@struct.dataclass
class EnvState:
    units: UnitState
    """Units in the environment with shape (T, N, 3) for T teams, N max units, and 3 features.

    3 features are for position (x, y), and energy
    """
    units_mask: chex.Array
    """Mask of units in the environment with shape (T, N) for T teams, N max units"""
    energy_nodes: chex.Array
    """Energy nodes in the environment with shape (N, 2) for N max energy nodes, and 2 features.

    2 features are for position (x, y)
    """
    
    energy_node_fns: chex.Array
    """Energy node functions for computing the energy field of the map. They describe the function with a sequence of numbers
    
    The first number is the function used. The subsequent numbers parameterize the function. The function is applied to distance of map tile to energy node and the function parameters.
    """

    # energy_field: chex.Array
    # """Energy field in the environment with shape (H, W) for H height, W width. This is generated from other state"""
    
    energy_nodes_mask: chex.Array
    """Mask of energy nodes in the environment with shape (N) for N max energy nodes"""
    relic_nodes: chex.Array
    """Relic nodes in the environment with shape (N, 2) for N max relic nodes, and 2 features.

    2 features are for position (x, y)
    """
    relic_node_configs: chex.Array
    """Relic node configs in the environment with shape (N, K, K) for N max relic nodes and a KxK relic configuration"""
    relic_nodes_mask: chex.Array
    """Mask of relic nodes in the environment with shape (N, ) for N max relic nodes"""
    relic_nodes_map_weights: chex.Array
    """Map of relic nodes in the environment with shape (H, W) for H height, W width. Each element is equal to the 1-indexed id of the relic node. This is generated from other state"""
    
    relic_spawn_schedule: chex.Array
    """Relic spawn schedule in the environment with shape (N, ) for N max relic nodes. Elements are the game timestep at which the relic node spawns"""

    map_features: MapTile
    """Map features in the environment with shape (W, H, 2) for W width, H height
    """

    sensor_mask: chex.Array
    """Sensor mask in the environment with shape (T, H, W) for T teams, H height, W width. This is generated from other state"""

    vision_power_map: chex.Array
    """Vision power map in the environment with shape (T, H, W) for T teams, H height, W width. This is generated from other state"""

    team_points: chex.Array
    """Team points in the environment with shape (T) for T teams"""
    team_wins: chex.Array
    """Team wins in the environment with shape (T) for T teams"""

    steps: int = 0
    """steps taken in the environment"""
    match_steps: int = 0
    """steps taken in the current match"""
    
@struct.dataclass
class EnvObs:
    """Partial observation of environment"""
    units: UnitState
    """Units in the environment with shape (T, N, 3) for T teams, N max units, and 3 features.

    3 features are for position (x, y), and energy
    """
    units_mask: chex.Array
    """Mask of units in the environment with shape (T, N) for T teams, N max units"""
    
    sensor_mask: chex.Array
    
    map_features: MapTile
    """Map features in the environment with shape (W, H, 2) for W width, H height
    """
    relic_nodes: chex.Array
    """Position of all relic nodes with shape (N, 2) for N max relic nodes and 2 features for position (x, y). Number is -1 if not visible"""
    relic_nodes_mask: chex.Array
    """Mask of all relic nodes with shape (N) for N max relic nodes"""
    team_points: chex.Array
    """Team points in the environment with shape (T) for T teams"""
    team_wins: chex.Array
    """Team wins in the environment with shape (T) for T teams"""
    steps: int = 0
    """steps taken in the environment"""
    match_steps: int = 0
    """steps taken in the current match"""
    