from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # This import is only executed by type checkers; it will not run at runtime.
    from .main import WildTorchSimulator

import json
import os

import torch

try:
    from torch.utils.tensorboard import SummaryWriter

    tensorboard_available = True
    print("Tensorboard available. Run `tensorboard --logdir=runs` to view logs.")
except ImportError:
    SummaryWriter = None
    tensorboard_available = False
    print("Warning: Tensorboard not available. Please run `pip install tensorboard` for better logging.")


class Logger:
    """
    Logger class for logging simulation statistics and snapshots.

    Attributes:
        logs:
            List of simulation statistics.
        p_burns:
            List of simulation p(burn).
        snapshots:
            List of simulation snapshots.
        tensorboard_writer:
            Tensorboard writer.
        log_dir:
            Directory to save logs and snapshots.
        json_filepath:
            Filepath for logs in JSON format.
        snapshots_filepath:
            Filepath for snapshots in PyTorch format.
        verbose:
            Print logs to console.
        disable_writing:
            Disable ANY writing to disk.
    """

    logs: list[dict[str, Any]]
    p_burns: list[torch.Tensor]
    snapshots: list[dict[str, Any]]
    tensorboard_writer: 'SummaryWriter | None'

    log_dir: str
    json_filepath: str
    snapshots_filepath: str
    verbose: bool
    disable_writing: bool

    def __init__(self,
                 log_dir: str | None = None,
                 comment: str = "",
                 json_filename: str = "logs.json",
                 snapshots_filename: str = "snapshots.pth",
                 disable_writing: bool = False,
                 verbose: bool = True,
                 ):
        """
        Logger class for logging simulation statistics and snapshots.

        Attributes:
            log_dir:
                Directory to save logs and snapshots.
            comment:
                Comment to append to the log directory. Not used if `log_dir` is provided.
            json_filename:
                Filename for logs in JSON format.
            snapshots_filename:
                Filename for snapshots in PyTorch format.
            disable_writing:
                Disable ANY writing to disk.
            verbose:
                Print logs to console.
        """
        if not log_dir:
            import socket
            from datetime import datetime

            current_time = datetime.now().strftime("%b%d_%H-%M-%S")
            log_dir = os.path.join(
                "runs", current_time + "_" + socket.gethostname() + comment
            )
        self.log_dir = log_dir

        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.json_filepath = os.path.join(log_dir, json_filename)
        self.snapshots_filepath = os.path.join(log_dir, snapshots_filename)
        self.verbose = verbose
        self.disable_writing = disable_writing

        self.logs = []
        self.snapshots = []
        self.p_burns = []
        self.tensorboard_writer = None
        if tensorboard_available and not self.disable_writing:
            self.tensorboard_writer = SummaryWriter(log_dir=log_dir, comment=comment)

    def __del__(self):
        # Ensure the tensorboard writer is closed when the logger is deleted
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()

    def log_stats(self, step: int = None, **kwargs: Any):
        """
        Log simulation statistics.

        Parameters:
            step: Simulation step.
            **kwargs: Simulation statistics to log.

        Examples:
            >>> logger = Logger()
            >>> logger.log_stats(step=0, burning_cells=100, burned_cells=50)
        """
        log_entry = {key: value for key, value in kwargs.items()}
        if step is not None:
            log_entry["step"] = step
        self.logs.append(log_entry)

        if self.verbose:
            print(log_entry)

        # Tensorboard
        if self.tensorboard_writer is not None:
            for key, value in kwargs.items():
                self.tensorboard_writer.add_scalar(key, value, step)

    def log_scalars_to_tensorboard(self, step: int = None, **kwargs: Any):
        """
        Log scalars to tensorboard.

        Parameters:
            step: Simulation step.
            **kwargs: Scalars to log to tensorboard.

        Examples:
            >>> logger = Logger()
            >>> logger.log_scalars_to_tensorboard(step=0, burning_cells={
                    'max': 100,
                    'min': 0,
                    'mean': 50,
                })
        """
        if self.tensorboard_writer is not None:
            for key, value in kwargs.items():
                self.tensorboard_writer.add_scalars(key, value, step)

    def log_image_to_tensorboard(self, **kwargs: Any):
        """
        Log image to tensorboard.

        Parameters:
            **kwargs: Scalars to log to tensorboard.

        Shape:
            img_tensor: Default is :math:`(3, H, W)`. You can use ``torchvision.utils.make_grid()`` to
            convert a batch of tensor into 3xHxW format or call ``add_images`` and let us do the job.
            Tensor with :math:`(1, H, W)`, :math:`(H, W)`, :math:`(H, W, 3)` is also suitable as long as
            corresponding ``dataformats`` argument is passed, e.g. ``CHW``, ``HWC``, ``HW``.

        Examples:
            >>> logger = Logger()
            >>> logger.log_image_to_tensorboard(tag='state',
                    img_tensor=torch.rand(3, 128, 128),
                    global_step=0)
        """
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_image(**kwargs)

    def log_images_to_tensorboard(self, **kwargs: Any):
        """
        Log images to tensorboard.

        Parameters:
            **kwargs: Scalars to log to tensorboard.

        Shape:
            img_tensor: Default is :math:`(N, 3, H, W)`. If ``dataformats`` is specified, other shape will be
            accepted. e.g. NCHW or NHWC.

        Examples:
            >>> logger = Logger()
            >>> logger.log_images_to_tensorboard(tag='state',
                    img_tensor=torch.rand(10, 3, 128, 128),
                    global_step=0)
        """
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_images(**kwargs)

    def log_video_to_tensorboard(self, **kwargs: Any):
        """
        Log video to tensorboard.

        Needs `moviepy`.

        Parameters:
            **kwargs: Scalars to log to tensorboard.

        Shape:
            vid_tensor: :math:`(N, T, C, H, W)`.
            The values should lie in [0, 255] for type `uint8` or [0, 1] for type `float`.

        Examples:
            >>> logger = Logger()
            >>> logger.log_video_to_tensorboard(tag='state',
                    vid_tensor=torch.rand(1, 10, 3, 128, 128),
                    global_step=0)
        """
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_video(**kwargs)

    def snapshot_simulation(self, simulator: 'WildTorchSimulator'):
        """
        Take a snapshot of the simulation.

        Parameters:
            simulator: Wildfire simulator instance.

        Examples:
            >>> logger = Logger()
            >>> logger.snapshot_simulation(simulator)
        """
        self.snapshots.append(simulator.checkpoint)

    def log_p_burn(self, simulator: 'WildTorchSimulator'):
        """
        Log p(burn) of the simulation.

        Parameters:
            simulator: Wildfire simulator instance.

        Examples:
            >>> logger = Logger()
            >>> logger.log_p_burn(simulator)
        """
        self.p_burns.append(simulator.p_burn.detach().clone())

    def save_logs(self):
        """
        Save logs to disk.

        Examples:
            >>> logger = Logger()
            >>> logger.save_logs()
        """
        if self.disable_writing:
            return

        if self.verbose:
            print(f"Logs saved: {self.json_filepath}")
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.flush()
            if self.verbose:
                print(f"Tensorboard logs saved: {self.tensorboard_writer.log_dir}")

    def save_snapshots(self):
        """
        Save snapshots to disk.

        Examples:
            >>> logger = Logger()
            >>> logger.save_snapshots()
        """
        torch.save(self.snapshots, self.snapshots_filepath)
        if self.verbose:
            print(f"Snapshots saved: {self.snapshots_filepath}")

    def load_snapshots(self, snapshots_filepath: str):
        """
        Load snapshots from disk.

        Parameters:
            snapshots_filepath:
                Filepath for snapshots in `.pth` format.

        Raises:
            FileNotFoundError:
                If snapshots file not found.

        Examples:
            >>> logger = Logger()
            >>> logger.load_snapshots()
        """
        if not os.path.exists(snapshots_filepath):
            raise FileNotFoundError(f"Snapshots file not found: {snapshots_filepath}")

        self.snapshots = torch.load(snapshots_filepath)
        if self.verbose:
            print(f"Snapshots loaded: {snapshots_filepath}")

    def clear_logs(self):
        """
        Clear logs.

        Examples:
            >>> logger = Logger()
            >>> logger.clear_logs()
        """
        self.logs = []
        if self.verbose:
            print("Logs cleared.")

    def clear_snapshots(self):
        """
        Clear snapshots.

        Examples:
            >>> logger = Logger()
            >>> logger.clear_snapshots()
        """
        self.snapshots = []
        if self.verbose:
            print("Snapshots cleared.")
