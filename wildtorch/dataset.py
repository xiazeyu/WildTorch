from typing import TYPE_CHECKING, Dict, Any, cast

if TYPE_CHECKING:
    # This import is only executed by type checkers; it will not run at runtime.
    from datasets.dataset_dict import DatasetDict
    from datasets import Dataset

import torch


def load_wildfire_sim_maps(device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                           dtype: torch.dtype = torch.float32,
                           batch_size: int | None = None,
                           ) -> 'DatasetDict':
    """
    Load the WildfireSimMaps dataset using the Hugging Face `datasets` library.

    This function checks if the `datasets` library is installed before attempting to load the dataset.
    If not installed, it prompts the user to install the library.

    Parameters:
        device:
            The device to load the dataset on.
        dtype:
            The data type to use for the dataset.
        batch_size:
            The batch size to use when loading the dataset.

    Raises:
        ImportError:
            If the `datasets` library is not installed.
        RuntimeError:
            If an error occurs while loading the dataset.

    Returns:
        A dataset object loaded from the 'xiazeyu/WildfireSimMaps' dataset.
    """
    try:
        from datasets import load_dataset
    except ImportError as err:
        # If the import fails, provide instructions on how to install the missing package
        raise ImportError(
            "The 'datasets' module is required but not installed. "
            "Please install it using 'pip install datasets'") from err

    try:
        print('loading dataset ...')
        # Attempt to load the dataset
        ds = load_dataset("xiazeyu/WildfireSimMaps", split="train")
    except Exception as err:
        # Handle other errors that might occur, such as network issues or incorrect dataset names
        raise RuntimeError(
            "An error occurred while loading the 'xiazeyu/WildfireSimMaps' dataset. Please check your internet "
            "connection or the dataset name.") from err

    ds = ds.with_format("torch", device=device)

    def preprocess_function(examples: Dict[str, Any]) -> Dict[str, Any]:
        # Reshape arrays based on the 'shape' field
        examples['density'] = [d.reshape(sh.tolist()).to(dtype=dtype) for d, sh in
                               zip(examples['density'], examples['shape'])]
        examples['slope'] = [s.reshape(sh.tolist()).to(dtype=dtype) for s, sh in
                             zip(examples['slope'], examples['shape'])]
        examples['canopy'] = [c.reshape(sh.tolist()).to(dtype=dtype) for c, sh in
                              zip(examples['canopy'], examples['shape'])]

        return examples

    ds = ds.map(preprocess_function, batched=True, batch_size=batch_size)

    return ds


def scale_tensor(input_tensor: torch.Tensor,
                 old_scale: tuple[float, float] = None,
                 new_scale: tuple[float, float] = None, ) -> torch.Tensor:
    """
    Scale the tensor from `old_scale` to `new_scale`.

    Parameters:
        input_tensor:
            The input tensor.
        old_scale:
            The old scale.
        new_scale:
            The new scale.

    Returns:
        The scaled tensor.
    """
    if old_scale is not None:
        old_min, old_max = old_scale
    else:
        old_min, old_max = torch.min(input_tensor), torch.max(input_tensor)

    if new_scale is not None:
        new_min, new_max = new_scale
    else:
        new_min, new_max = 0, 1

    scaled_tensor = (input_tensor - old_min) / (old_max - old_min)
    scaled_tensor = new_min + (new_max - new_min) * scaled_tensor

    return scaled_tensor


def transform_wildfire_sim_map(map_data: 'Dataset') -> torch.Tensor:
    """
    Transform and stack the map data into a tensor.

    The scale values are derived from the paper https://doi.org/10.1016/j.amc.2008.06.046,
    with meteorological conditions:

    - Wind direction North
    - Wind speed 8–10 m/s (~5 Beaufort)
    - Average temperature 30 °C
    - Humidity 36%

    Modify it to fit the range of your own dataset.

    Parameters:
        map_data:
            The map data loaded from the dataset.

    Returns:
        The map data tensor.
    """
    canopy = cast(torch.Tensor, map_data['canopy'])
    density = cast(torch.Tensor, map_data['density'])
    slope = cast(torch.Tensor, map_data['slope'])

    canopy = scale_tensor(canopy, (0, 75), (-0.3, 0.4))
    density = scale_tensor(density, (0, 0.3), (-0.4, 0.3))

    # Stack the map data into a single tensor
    map_tensor = torch.stack([canopy, density, slope])

    return map_tensor


def data_augmentation(map_data: torch.Tensor,
                      shape: tuple[int, int] = (256, 256),
                      ) -> torch.Tensor:
    """
    Apply data augmentation to the map data.

    This is just an example of how you can apply data augmentation to the map data tensor.

    Parameters:
        map_data:
            The map data tensor.
        shape:
            The shape to use for resizing and cropping.

    Returns:
        The augmented map data tensor.
    """
    from torchvision.transforms import v2, InterpolationMode

    transforms = v2.Compose([v2.RandomHorizontalFlip(p=0.5), v2.RandomVerticalFlip(p=0.5), v2.RandomChoice(
        [v2.Resize(size=shape, interpolation=InterpolationMode.BILINEAR, antialias=True),
         v2.RandomCrop(size=shape, padding=None),
         v2.RandomResizedCrop(size=shape, scale=(0., 1.), ratio=(0.75, 1.3),
                              interpolation=InterpolationMode.BILINEAR, antialias=True), ], [0.1, 0.45, 0.45]),
                             v2.ToDtype(dtype=torch.float32, scale=False), ])
    return transforms(map_data)


def load_example_dataset() -> torch.Tensor:
    """
    Load a single WildfireSimMap as a tensor.

    This is an example function to demonstrate how you can load a single map from the dataset.

    Returns:
        The map data tensor.
    """
    ds = load_wildfire_sim_maps()
    ds = transform_wildfire_sim_map(ds[0])

    # Not applying data augmentation here, but you can if you want
    # ds = data_augmentation(ds)

    return ds


def generate_empty_dataset(shape: tuple[int, int] = (256, 256),
                           device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                           dtype: torch.dtype = torch.float32,
                           ) -> torch.Tensor:
    """
    Generate an empty dataset tensor with the specified shape.

    Parameters:
        shape:
            The shape of the dataset.
        device:
            The device to create the dataset on.
        dtype:
            The data type to use for the dataset.

    Returns:
        The empty dataset tensor.
    """
    return torch.zeros((3, *shape), dtype=dtype, device=device)


if __name__ == "__main__":
    # Example usage
    try:
        dataset = load_wildfire_sim_maps()
        print("Dataset loaded successfully.")
        print(dataset['name'])
        wildfire_map = transform_wildfire_sim_map(dataset[0])
        wildfire_map = data_augmentation(wildfire_map)
    except Exception as e:
        print(f"Failed to load dataset: {e}")
