from typing import Dict, Any

import torch
from einops import rearrange, repeat, reduce

from .dataset import generate_empty_dataset


# https://pytorch.org/docs/stable/notes/randomness.html.
# no difference in performance and results with or without this line
# torch.use_deterministic_algorithms(True)


class SimulatorConstants:
    """
    Constants configuration class for the simulator

    Attributes:
        p_h:
            The probability that a burnable cell adjacent to a burning cell will catch fire
            at the next time step under normal conditions
        c_1:
            The coefficient of wind velocity
        c_2:
            The coefficient of wind direction
        a:
            The coefficient of ground elevation
        theta:
            The matrix of wind direction in degrees, measured clockwise from north
        theta_w:
            The direction of the wind in degrees, measured clockwise from north.
        v:
            The wind velocity, unit in m/s
        p_firebreak:
            The probability that a burnable cell will not catch fire even if it is adjacent to a burning cell
        p_continue_burn:
            The probability that a burning cell will continue to burn at the next time step
        device:
            The device to use
        dtype:
            The data type to use
    """

    p_h: torch.Tensor
    c_1: torch.Tensor
    c_2: torch.Tensor
    a: torch.Tensor
    theta: torch.Tensor | None
    theta_w: torch.Tensor
    v: torch.Tensor
    p_firebreak: torch.Tensor
    p_continue_burn: torch.Tensor
    device: torch.device
    dtype: torch.dtype

    def __init__(self, p_h: float = 0.58, c_1: float = 0.045, c_2: float = 0.131, a: float = 0.078, theta_w: int = 0,
                 v: float = 10, p_firebreak: float = 0.9, p_continue_burn: float = 0.5,
                 device: torch.device = torch.device('cpu'), dtype: torch.dtype = torch.float32, ) -> None:
        """
        Initialize the simulator constants with the default values.

        The user may tweak the coefficient of ground elevation (a), as it seems low for general terrain to be effective

        For custom theta_w, just set the value after initialization.

        Default values are copied from the paper https://doi.org/10.1016/j.amc.2008.06.046,
        with meteorological conditions:

        - Wind direction North
        - Wind speed 8-10 m/s (~5 Beaufort)
        - Average temperature 30 °C
        - Humidity 36%

        Parameters:
            p_h:
                The probability that a burnable cell adjacent to a burning cell will catch fire
                at the next time step under no wind and flat terrain
            c_1:
                The coefficient of wind velocity
            c_2:
                The coefficient of wind direction
            a:
                The coefficient of ground elevation
            theta_w:
                The direction of the wind in degrees, measured clockwise from north
            v:
                The wind velocity, unit in m/s
            p_firebreak:
                The probability that a burnable cell will not catch fire even if it is adjacent to a burning cell
            p_continue_burn:
                The probability that a burning cell will continue to burn at the next time step
            device:
                The device to use
            dtype:
                The data type to use
        """
        self.device = device
        self.dtype = dtype
        self.p_h = torch.tensor(p_h, dtype=dtype, device=device)
        self.c_1 = torch.tensor(c_1, dtype=dtype, device=device)
        self.c_2 = torch.tensor(c_2, dtype=dtype, device=device)
        self.a = torch.tensor(a, dtype=dtype, device=device)
        self.theta = None
        self.theta_w = torch.tensor(theta_w, dtype=dtype, device=device)
        self.V = torch.tensor(v, dtype=dtype, device=device)
        self.p_firebreak = torch.tensor(p_firebreak, dtype=dtype, device=device)
        self.p_continue_burn = torch.tensor(p_continue_burn, dtype=dtype, device=device)

    def __str__(self) -> str:
        return (
            f"SimulatorConstants(p_h={self.p_h}, c_1={self.c_1}, c_2={self.c_2}, a={self.a}, theta_w={self.theta_w}, "
            f"V={self.V}, p_firebreak={self.p_firebreak}, p_continue_burn={self.p_continue_burn}, "
            f"device={self.device}, dtype={self.dtype})")

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def theta_w(self) -> int:
        return self._theta_w

    @theta_w.setter
    def theta_w(self, value) -> None:
        self._theta_w = value  # in degrees, measured clockwise from north
        self.update_theta()

    def update_theta(self) -> None:
        """
        Update the wind direction matrix based on the wind direction (theta_w).
        """
        self.theta = torch.tensor([[315, 0, 45], [270, 0, 90], [225, 180, 135]], dtype=self.dtype, device=self.device)
        self.theta = torch.remainder(self.theta - self.theta_w, 360)
        self.theta[1, 1] = 0


class WildTorchSimulator:
    """
    WildTorch main simulator class

    Channel definition for fire_state:

    | Index | Description | Data type | Default value |
    |:---:|:---:|:---:|:---:|
    | 0 | burning | bool | False |
    | 1 | burned | bool | False |
    | 2 | firebreak | bool | False |

    Cell state in the simulator:

    | Description | burning_bit | burned_bit |
    |:---:|:---:|:---:|
    | burnable | 0 | 0 |
    | burning | 1 | 0 |
    | burned | 0 | 1 |

    Optimized state transition:

    ```
    some_cells.state = burning # Make initial ignition

    IF current.state == burning THEN
        IF NOT rand_int > p_count THEN
            current.state = burned # burn out
        FOR each neighbour in current.neighbours DO
            IF NOT neighbour.state == burned THEN
                IF rand_int > p_burn[neighbour.x, neighbour.y, direction_x, direction_y] THEN
                    neighbour.state = burning # propagate burning
    ```

    The device and data type from the simulator constants are used for the simulator.

    Attributes:
        simulator_constants:
            The simulator constants.
        wildfire_map:
            The wildfire map tensor.
        device:
            The device now using.
        dtype:
            The data type now using.
        p_propagate_constant:
            The constant part of the probability of propagation.
        p_burn:
            The probability of burning.
        fire_state:
            The current fire state tensor. [burning, burned, firebreak]
        seed:
            The initial random seed.
        maximum_step:
            The maximum number of steps to simulate.
        initial_ignition:
            The initial ignition map tensor.
        current_step:
            The current step of the simulation.
        terminated:
            Whether the simulation is terminated.
        truncated:
            Whether the simulation is truncated.
    """

    simulator_constants: SimulatorConstants
    wildfire_map: torch.Tensor
    device: torch.device
    dtype: torch.dtype

    p_propagate_constant: torch.Tensor
    p_burn: torch.Tensor
    fire_state: torch.Tensor

    seed: int
    maximum_step: int
    initial_ignition: torch.Tensor | None

    current_step: int
    terminated: bool
    truncated: bool

    def __str__(self) -> str:
        return (
            f"WildTorchSimulator(shape={self.wildfire_map[0].shape}, current_step={self.current_step}, "
            f"seed={self.seed}, terminated={self.terminated}, truncated={self.truncated}, "
            f"device={self.device}, dtype={self.dtype})")

    def __repr__(self) -> str:
        return self.__str__()

    def __init__(self,
                 wildfire_map: torch.Tensor = generate_empty_dataset(),
                 simulator_constants: SimulatorConstants = SimulatorConstants(),
                 maximum_step: int = 200,
                 initial_ignition: torch.Tensor | None = None,
                 seed: int | None = None,
                 ) -> None:
        """
        Initialize the simulator with the given wildfire map and simulator constants.

        Parameters:
            wildfire_map:
                The wildfire map tensor.
            simulator_constants:
                The simulator constants.
            maximum_step:
                The maximum number of steps to simulate.
            initial_ignition:
                The initial ignition map tensor.
            seed:
                The random seed to use.
        """
        self._simulator_constants = simulator_constants

        self._p_propagate_constant = None
        self.wildfire_map = wildfire_map
        self._fire_state = torch.zeros((3, *wildfire_map[0].shape), dtype=torch.bool, device=self.device)

        self.maximum_step = maximum_step
        if initial_ignition is None:
            self.initial_ignition = torch.zeros_like(wildfire_map[0], dtype=torch.bool, device=self.device)
        else:
            self.initial_ignition = initial_ignition.to(device=self.device, dtype=self.dtype)
            self._fire_state[0] = self.initial_ignition
        self.seed = seed

        self._current_step = 0

    @property
    def simulator_constants(self) -> SimulatorConstants:
        return self._simulator_constants

    @simulator_constants.setter
    def simulator_constants(self, value: SimulatorConstants) -> None:
        self._simulator_constants = value
        self._p_propagate_constant = self.compute_p_propagate_constant()

    @property
    def wildfire_map(self) -> torch.Tensor:
        return self._wildfire_map

    @wildfire_map.setter
    def wildfire_map(self, value: torch.Tensor) -> None:
        assert value.dim() == 3 and value.shape[0] == 3, "wildfire_map must be of shape (3, xxx, xxx)"
        self._wildfire_map = value.to(device=self.device, dtype=self.dtype)
        self._p_propagate_constant = self.compute_p_propagate_constant()

    @property
    def device(self) -> torch.device:
        return self.simulator_constants.device

    @property
    def dtype(self) -> torch.dtype:
        return self.simulator_constants.dtype

    @property
    def p_propagate_constant(self) -> torch.Tensor:
        return self._p_propagate_constant

    @property
    def fire_state(self) -> torch.Tensor:
        return self._fire_state

    @property
    def seed(self) -> int:
        return self._seed

    @seed.setter
    def seed(self, value: int | None) -> None:
        """
        Set the random seed.

        If the value is None, a new random seed is generated.

        Parameters:
            value (int): The random seed to use.
        """
        self._seed = value if value is not None else torch.Generator().seed()
        torch.manual_seed(self._seed)

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def terminated(self) -> bool:
        return self.current_step >= self.maximum_step

    @property
    def truncated(self) -> bool:
        return torch.sum(self.fire_state[0]) == 0

    def compute_p_propagate_constant(self) -> torch.Tensor:
        """
        Compute the constant part of the probability of propagation.

        This method is usually called after changing the wildfire map or simulator constants.

        This function follows the following formula:

        $$ p_{propagate}=p_h(1+p_{veg})(1+p_{den})p_wp_s $$

        in which,

        $$ p_w=\\exp(c_1V)\\exp(c_2V(\\cos(\\theta)-1)) $$

        $$ p_s=\\exp(a\\theta_s) $$

        Returns:
            The constant part of the probability of propagation.
        """
        p_h = self.simulator_constants.p_h
        p_veg = self.wildfire_map[0]
        p_den = self.wildfire_map[1]
        p_w = (torch.exp(self.simulator_constants.c_1 * self.simulator_constants.V) * torch.exp(
            self.simulator_constants.c_2 * self.simulator_constants.V * (
                    torch.cos(torch.deg2rad(self.simulator_constants.theta)) - 1)))
        p_s = torch.exp(self.simulator_constants.a * torch.deg2rad(self.wildfire_map[2]))

        p_veg = rearrange(p_veg, 'h w -> h w 1 1')
        p_den = rearrange(p_den, 'h w -> h w 1 1')
        p_s = rearrange(p_s, 'h w -> h w 1 1')
        p_w = rearrange(p_w, 'o1 o2 -> 1 1 o1 o2', o1=3, o2=3)

        return p_h * (1 + p_veg) * (1 + p_den) * p_w * p_s

    @property
    # @torch.compile
    def p_burn(self) -> torch.Tensor:
        """
        Compute the probability of burning for each cell.

        This functions follows the following formula:

        $$ p_{burn}=1-\\prod_{i=1}^{8}{(1-p_{propagate_i})} $$

        Returns:
            The probability of burning for each cell.
        """
        burning, burned, firebreak = self.fire_state

        conditional_tensor = torch.where(firebreak, 1 - self.simulator_constants.p_firebreak, 1)
        expanded_tensor = repeat(conditional_tensor, 'h w -> h w c1 c2', c1=3, c2=3)
        p_propagate = self.p_propagate_constant * expanded_tensor

        p_burn = torch.zeros_like(p_propagate, dtype=self.dtype, device=self.device)
        expanded_burning_map = repeat(burning, 'h w -> h w c1 c2', c1=3, c2=3)

        # any out-of-bounds access in p_propagate is avoided by the slicing
        # any out-of-bounds access in expanded_burning_map will result in 0

        p_burn[:-1, :-1, 0, 0] = torch.where(expanded_burning_map[1:, 1:, 0, 0], p_propagate[:-1, :-1, 0, 0], 0)
        p_burn[:-1, :, 0, 1] = torch.where(expanded_burning_map[1:, :, 0, 1], p_propagate[:-1, :, 0, 1], 0)
        p_burn[:-1, 1:, 0, 2] = torch.where(expanded_burning_map[1:, :-1, 0, 2], p_propagate[:-1, 1:, 0, 2], 0)
        p_burn[:, :-1, 1, 0] = torch.where(expanded_burning_map[:, 1:, 1, 0], p_propagate[:, :-1, 1, 0], 0)
        p_burn[:, :, 1, 1] = 0
        p_burn[:, 1:, 1, 2] = torch.where(expanded_burning_map[:, :-1, 1, 2], p_propagate[:, 1:, 1, 2], 0)
        p_burn[1:, :-1, 2, 0] = torch.where(expanded_burning_map[:-1, 1:, 2, 0], p_propagate[1:, :-1, 2, 0], 0)
        p_burn[1:, :, 2, 1] = torch.where(expanded_burning_map[:-1, :, 2, 1], p_propagate[1:, :, 2, 1], 0)
        p_burn[1:, 1:, 2, 2] = torch.where(expanded_burning_map[:-1, :-1, 2, 2], p_propagate[1:, 1:, 2, 2], 0)

        p_burn = 1 - reduce(1 - p_burn, 'h w c1 c2 -> h w', 'prod', c1=3, c2=3)
        return p_burn

    def step(self, force: bool = False) -> None:
        """
        Perform one step of simulation.

        Parameters:
            force:
                Whether to force calculating even if the simulation is terminated or truncated.
        """
        if not force and (self.terminated or self.truncated):
            return

        burning, burned, firebreak = self.fire_state
        p_burn = self.p_burn

        rand_propagate = torch.rand_like(p_burn, dtype=self.dtype, device=self.device)
        rand_continue = torch.rand_like(p_burn, dtype=self.dtype, device=self.device)

        # burnable patches have p_burn probability to be burning
        burnable = ~(burning | burned)
        new_burning = burnable & (rand_propagate < p_burn)
        burning[new_burning] = True

        # burning patches have p_continue_burn probability to continue burning
        will_burn_out = burning & (rand_continue >= self.simulator_constants.p_continue_burn)
        burning[will_burn_out] = False
        burned[will_burn_out] = True

        # burned remain burned
        self._current_step += 1

    def batch_forward(self,
                      step: int = 20,
                      ) -> None:
        """
        Perform multiple steps of simulation.

        Parameters:
            step: The number of steps to perform.
        """
        for i in range(step):
            self.step()

    @property
    def checkpoint(self) -> Dict[str, Any]:
        """
        Get the checkpoint of the simulator.

        {'seed', 'current_step', 'fire_state'}

        Returns:
            The checkpoint of the simulator.
        """
        return {
            'seed': self.seed,
            'current_step': self.current_step,
            'fire_state': self.fire_state.clone().detach(),  # will not change device
        }

    def load_checkpoint(self,
                        checkpoint: Dict[str, Any],
                        restore_seed: bool = True,
                        ) -> None:
        """
        Reset the simulator to the checkpoint.

        Parameters:
            checkpoint: The checkpoint to reset.
            restore_seed: Whether to reset the random seed as well.
        """
        if restore_seed:
            self.seed = checkpoint['seed']
        else:
            self._seed = None
        self._current_step = checkpoint['current_step']
        self._fire_state = checkpoint['fire_state'].clone().detach()

    def reset(self) -> None:
        """
        Reset the simulator.

        - The current step is set to 0.
        - The random seed is set to None.
        - The fire state is set to the initial ignition map.
        """
        self._current_step = 0
        self.seed = None
        self._fire_state = torch.zeros((3, *self.wildfire_map[0].shape), dtype=torch.bool, device=self.device)
        self._fire_state[0] = self.initial_ignition
