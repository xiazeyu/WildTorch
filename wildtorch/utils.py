from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Dict, Any, Union

import torch

# Dependencies checking begins here

if TYPE_CHECKING:
    # This import is only executed by type checkers; it will not run at runtime.
    pass

try:
    import numpy as np

    numpy_available = True
except ImportError:
    np = None
    numpy_available = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors
    import matplotlib.animation

    matplotlib_available = True
except ImportError:
    plt = None
    matplotlib = None
    matplotlib_available = False

try:
    import imageio.v3 as iio

    imageio_available = True
except ImportError:
    iio = None
    imageio_available = False


def check_np() -> bool:
    """
    Check if numpy is available

    Raises:
        ImportError:
            If numpy is not available.

    Returns:
        True if numpy is available
    """
    if not numpy_available:
        raise ImportError("numpy is not available. Please install it using `pip install numpy`")
    return True


def check_plt() -> bool:
    """
    Check if matplotlib is available

    Raises:
        ImportError:
            If matplotlib is not available.

    Returns:
        True if matplotlib is available
    """
    if not matplotlib_available:
        raise ImportError("matplotlib is not available. Please install it using `pip install matplotlib`")
    return True


def check_iio() -> bool:
    """
    Check if imageio is available

    Raises:
        ImportError:
            If imageio is not available.

    Returns:
        True if imageio is available
    """
    if not imageio_available:
        raise ImportError("imageio is not available. Please install it using `pip install imageio`")
    return True


# Dependencies checking ends here

def create_ignition(shape: tuple[int, int] = (128, 128),
                    pos: tuple[tuple[float, float], tuple[float, float]] = ((0.2, 0.2), (0.8, 0.8)),
                    size: tuple[float, float] = (0.01, 0.01),
                    mode: str = 'center',
                    count: int = 5,
                    ) -> torch.Tensor:
    """
    Create ignition map

    Parameters:
        shape:
            The size should match the size of the map.
        pos:
            The position of ignition.
        size:
            The size of ignition.
        mode:
            mode to generate ignition map.
        count:
            The number of ignition points.

    Raises:
        AssertionError:
            If the mode is not valid.

    Returns:
        Ignition map
    """
    assert mode in ['center', 'random-single', 'random-multi'], f"Invalid mode: {mode}"

    field = torch.zeros(shape, dtype=torch.bool)

    start_x, end_x = int(pos[0][0] * shape[0]), int(pos[1][0] * shape[0])
    start_y, end_y = int(pos[0][1] * shape[1]), int(pos[1][1] * shape[1])

    width = min(20, max(3, int(size[0] * shape[0] // 2)))
    height = min(20, max(3, int(size[1] * shape[1] // 2)))

    if mode == 'center':
        center_x = int((start_x + end_x) // 2)
        center_y = int((start_y + end_y) // 2)
        field[center_x - width:center_x + width, center_y - height:center_y + height] = True

    elif mode in ['random-single', 'random-multi']:
        count = count if mode == 'random-multi' else 1
        for _ in range(count):
            x = torch.randint(start_x, end_x, (1,)).item()
            y = torch.randint(start_y, end_y, (1,)).item()
            field[x - width:x + width, y - height:y + height] = True

    return field


def to_ndarray(tensor: 'Union[torch.Tensor, np.ndarray]') -> 'np.ndarray':
    """
    Convert `torch.tensor` or `numpy.ndarray` to `numpy.ndarray`

    Parameters:
        tensor:
            The tensor to convert.

    Raises:
        ImportError:
            If numpy is not available.
        TypeError:
            If the tensor type is not supported.

    Returns:
        The converted numpy array.
    """
    check_np()

    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise TypeError(f"Invalid input type: {type(tensor)}")


def colorize_array(array: 'np.ndarray',
                   cmap: str = 'viridis',
                   vmin: int = None,
                   vmax: int = None,
                   ) -> 'np.ndarray':
    """
    Colorize array using a colormap

    Parameters:
        array:
            The array to colorize.
        cmap:
            The colormap to use.
        vmin:
            The minimum value.
        vmax:
            The maximum value.

    Raises:
        ImportError:
            If matplotlib is not available.

    Returns:
        The colorized array.
    """
    check_plt()

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap)
    colored = cmap(norm(array))[..., :3]

    return colored  # (h, w, c)


def compose_vis_fire_state(fire_state: torch.Tensor) -> 'np.ndarray':
    """
    Compose fire state tensor into a single array for visualization

    Parameters:
        fire_state:
            The fire state tensor.

    Raises:
        ImportError:
            If numpy is not available.
        AssertionError:
            If the fire state shape is invalid.

    Returns:
        The composed fire state array.
    """
    check_np()
    assert fire_state.dim() == 3 and fire_state.shape[0] == 3, f"Invalid fire state shape: {fire_state.shape}"
    burning, burned, firebreak = fire_state
    composed_state = np.zeros(burning.shape, dtype=np.int32)

    composed_state[burning] = 1
    composed_state[burned] = 2
    composed_state[firebreak] = 3

    return composed_state


def compose_vis_wildfire_map(wildfire_map: torch.Tensor) -> 'np.ndarray':
    """
    Compose wildfire map into a single array for visualization

    Parameters:
        wildfire_map:
            The wildfire map tensor.

    Raises:
        AssertionError:
            If the wildfire map shape is invalid.

    Returns:
        The composed wildfire map array.
    """
    assert wildfire_map.dim() == 3 and wildfire_map.shape[0] == 3, f"Invalid wildfire_map shape: {wildfire_map.shape}"
    return to_ndarray(wildfire_map[0] * wildfire_map[1])


def overlay_arrays(array1: 'np.ndarray', array2: 'np.ndarray', alpha: float = 0.5) -> 'np.ndarray':
    """
    Overlay two arrays

    $$ out = \\alpha \\cdot arr_1+(1-\\alpha)\\cdot arr_2 $$

    Parameters:
        array1:
            The first array.
        array2:
            The second array.
        alpha:
            The overlay alpha value.

    Raises:
        AssertionError:
            If the shape of the arrays are not equal.
        AssertionError:
            If the alpha value is invalid.

    Returns:
        The overlaid array.
    """
    assert array1.shape == array2.shape, f"Shape mismatch: {array1.shape} != {array2.shape}"
    assert 0 <= alpha <= 1, f"Invalid alpha value: {alpha}"

    return alpha * array1 + (1 - alpha) * array2


def visualize_array(array: 'np.ndarray', **kwargs: Any):
    """
    Visualize array using matplotlib

    Parameters:
        array:
            The array to visualize.
        **kwargs:
            Additional keyword arguments.

    Raises:
        ImportError:
            If matplotlib is not available.

    Example:
        visualize_array(array)
    """
    check_plt()
    plt.imshow(array, **kwargs)
    plt.colorbar()
    plt.show()


def plot_stats(logs: list[Dict[str, Any]],
               keys: list[str],
               ):
    """
    Plot statistics from logs

    Parameters:
        logs:
            List of log entries.
        keys:
            List of keys to plot.

    Raises:
        ImportError:
            If matplotlib is not available.

    Examples:
        >>> plot_stats(logs, keys=['burning_cells', 'burned_cells'])
    """
    check_plt()
    for key in keys:
        plt.plot([log[key] for log in logs], label=key)
        plt.legend()
        plt.show()


def animate_snapshots(
        snapshots: list[Dict[str, Any]],
        wildfire_map: torch.Tensor | None = None,
        out_filename: str = 'runs/wildfire_simulation.mp4',
        fps: int = 24,
) -> 'np.ndarray':
    """
    Animate snapshots

    Parameters:
        snapshots:
            List of snapshots.
        wildfire_map:
            The wildfire map tensor.
        out_filename:
            Output filename. Defaults to 'wildfire_simulation.mp4'.
        fps:
            Frames per second. Defaults to 10.

    Raises:
        ImportError:
            If numpy is not available.
        ImportError:
            If imageio is not available.


    Returns:
        The movie array.

    Examples:
        >>> animate_snapshots(snapshots, fps=10)
    """

    check_np()
    check_iio()

    def process_frame(index_snapshot, wf_map=None):
        index, snapshot = index_snapshot
        fire_state = snapshot['fire_state'].cpu()
        vis_fire_state = colorize_array(compose_vis_fire_state(fire_state), vmin=0, vmax=3)

        if wf_map is not None:
            vis_wildfire_map = colorize_array(compose_vis_wildfire_map(wf_map))
            output = overlay_arrays(vis_fire_state, vis_wildfire_map, 0.6)
        else:
            output = vis_fire_state

        return index, (output * 255).astype(np.uint8)

    # Using ThreadPoolExecutor to parallelize frame processing
    with ThreadPoolExecutor() as executor:
        # Submit all tasks to the executor along with their indices
        futures = [executor.submit(process_frame, (index, snapshot), wildfire_map) for index, snapshot in
                   enumerate(snapshots)]

    processed_frames = []

    # Retrieve results as they are completed
    for future in as_completed(futures):
        index, processed_frame = future.result()
        processed_frames.append((index, processed_frame))

    processed_frames.sort(key=lambda x: x[0])

    movie_array_list = [frame for _, frame in processed_frames]
    movie_array = np.array(movie_array_list)

    with iio.imopen(out_filename, "w", plugin="pyav") as file:
        file.init_video_stream("libx264", fps=fps, force_keyframes=True)
        file.container_metadata["comment"] = "This video was created using WildTorch."
        file.write(movie_array,
                   # No more "height not divisible by 2"
                   filter_sequence=[('scale', '-2:1080')]
                   )

    return movie_array
