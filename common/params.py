import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from pydrake.all import MultibodyPlant, Context, RigidTransform, Cylinder
from functools import cached_property


@dataclass
class ParticleRopeParams:
    M: np.ndarray # Mass matrix for each dim of each particle
    x0: Optional[np.ndarray]
    v0: Optional[np.ndarray]
    plant: Optional[MultibodyPlant]
    nu: int

    num_particles: int = 10
    particle_dim: int = 3
    l: float = 1.0 # Distance between particles
    g: float = 9.81 
    N: int = 1000 # number of timesteps
    dt: float = 1 / 240
    stiffness: list[float] = 0
    damping: list[float] = 0
    fixed_end: bool = False
    fixed_end_pose: RigidTransform = field(
        default_factory=RigidTransform
    )  # pose of fixed_end_constraint
    finger_collision: Optional[Cylinder] = None

    @cached_property
    def plant_context(self) -> Context:
        return self.plant.CreateDefaultContext()

    @cached_property
    def plant_ad(self) -> MultibodyPlant:
        return self.plant.ToAutoDiffXd()

    @cached_property
    def plant_context_ad(self) -> Context:
        return self.plant_ad.CreateDefaultContext()

    @cached_property
    def plant_sym(self) -> MultibodyPlant:
        return self.plant.ToSymbolic()

    @cached_property
    def plant_context_sym(self) -> Context:
        return self.plant_sym.CreateDefaultContext()

    @property
    def nq(self) -> int: # Position dim
        return self.num_particles * self.particle_dim

    @property
    def nc(self) -> int: # Constraints dim
        if self.fixed_end:
            return self.num_particles + 1

        return self.num_particles

    @property
    def nz(self) -> int: # Total state dim
        return self.nq * 2 + self.nc

    @cached_property
    def rope_length(self) -> float:
        num_rope_links = self.num_particles + int(self.fixed_end)
        return float(self.l * num_rope_links)

    @property
    def z0(self) -> np.ndarray: # Initial state
        if self.x0 is None or self.v0 is None:
            return None

        return np.concatenate([self.x0, self.v0, [0]*self.nc])
