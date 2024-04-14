# WildTorch

[![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)
[![Read the Docs](https://readthedocs.org/projects/wildtorch/badge/)](https://wildtorch.readthedocs.io/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10968647.svg)](https://doi.org/10.5281/zenodo.10968647)

WildTorch: Leveraging GPU Acceleration for High-Fidelity, Stochastic Wildfire Simulations with PyTorch

### Installation

Install with minimal dependencies:

```shell
pip install wildtorch
```

Install with full dependencies (includes visualization and logging):

```shell
pip install 'wildtorch[full]'
```

### Quick Start

```shell
pip install 'wildtorch[full]'
```

```python
import wildtorch as wt

wildfire_map = wt.dataset.generate_empty_dataset()

simulator = wt.WildTorchSimulator(
    wildfire_map=wildfire_map,
    simulator_constants=wt.SimulatorConstants(p_continue_burn=0.7),
    initial_ignition=wt.utils.create_ignition(shape=wildfire_map[0].shape),
)

logger = wt.logger.Logger()

for i in range(200):
    simulator.step()
    logger.log_stats(
        step=i,
        num_cells_on_fire=wt.metrics.cell_on_fire(simulator.fire_state).item(),
        num_cells_burned_out=wt.metrics.cell_burned_out(simulator.fire_state).item(),
    )
    logger.snapshot_simulation(simulator)

logger.save_logs()
logger.save_snapshots()

```

### Demo

See Our Live Demo at [Hugging Face Space](https://xiazeyu-wildtorch.hf.space/).

### API Documents

See at Our [Read the Docs](https://wildtorch.readthedocs.io/).
