import torch


def cell_on_fire(fire_state: torch.Tensor) -> torch.Tensor:
    """
    Calculate the number of cells on fire.

    Higher values indicate more cells on fire, lower values indicate fewer cells on fire

    Parameters:
        fire_state:
            fire state of shape (3, H, W)

    Returns:
        number of cells on fire, shape ()
    """
    return torch.sum(fire_state[0])


def cell_burned_out(fire_state: torch.Tensor) -> torch.Tensor:
    """
    Calculate the number of cells burned out.

    Higher values indicate more cells burned out, lower values indicate fewer cells burned out

    Parameters:
        fire_state:
            fire state of shape (3, H, W)

    Returns:
        number of cells burned out, shape ()
    """
    return torch.sum(fire_state[1])


def weighted_cell_on_fire(fire_state: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Calculate the weighted number of cells on fire.

    Higher values indicate more valuable cells on fire, lower values indicate fewer valuable cells on fire

    Parameters:
        fire_state:
            fire state of shape (3, H, W)
        weights:
            weight matrix of shape (H, W)

    Returns:
        weighted number of cells on fire, shape ()
    """
    return torch.sum(fire_state[0] * weights)


def weighted_cell_burned_out(fire_state: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Calculate the weighted number of cells burned out.

    Higher values indicate more valuable cells burned out, lower values indicate fewer valuable cells burned out

    Parameters:
        fire_state:
            fire state of shape (3, H, W)
        weights:
            weight matrix of shape (H, W)

    Returns:
        weighted number of cells burned out, shape ()
    """
    return torch.sum(fire_state[1] * weights)


def calculate_fire_state_difference(fire_state_a: torch.Tensor, fire_state_b: torch.Tensor) -> torch.Tensor:
    """
    Calculate the difference between two fire states.

    | Value | a[x,y] is burnable | a[x,y] is not burnable |
    |:---:|:---:|:---:|
    | b[x,y] is burnable | 0 | 1 |
    | b[x,y] is not burnable | -1 | 0 |

    ```
    IF a[x,y] is burnable and b[x,y] is not burnable THEN
        return -1
    IF a[x,y] is NOT burnable and b[x,y] is burnable THEN
        return 1
    ELSE
        return 0
    ```

    Parameters:
        fire_state_a:
            fire state of shape (3, H, W)
        fire_state_b:
            fire state of shape (3, H, W)

    Returns:
        difference between fire states, shape (H, W)
    """
    assert fire_state_a.shape == fire_state_b.shape, "Fire states must have the same shape"

    burnable_a = ~(fire_state_a[0] | fire_state_a[1])
    burnable_b = ~(fire_state_b[0] | fire_state_b[1])

    output = torch.zeros_like(fire_state_a[0], dtype=torch.int)
    output[(burnable_a & ~burnable_b)] = -1
    output[(~burnable_a & burnable_b)] = 1

    return output


def saved_cells(fire_state_diff: torch.Tensor) -> torch.Tensor:
    """
    Calculate the number of cells saved.

    Higher values indicate more cells saved, lower values indicate more cells burned out

    Parameters:
        fire_state_diff:
            fire state difference of shape (H, W)

    Returns:
        number of cells saved, shape ()
    """
    return torch.sum(fire_state_diff)


def weighted_saved_cells(fire_state_diff: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """
    Calculate the weighted number of cells saved.

    Higher values indicate more valuable cells saved, lower values indicate more valuable cells burned out

    Parameters:
        fire_state_diff:
            fire state difference of shape (H, W)
        weights:
            weight matrix of shape (H, W)

    Returns:
        weighted number of cells saved, shape()
    """
    return torch.sum(fire_state_diff * weights)
