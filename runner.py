# import matplotlib.pyplot as plt

from wildtorch import dataset, metrics, utils, logger
from wildtorch.main import WildTorchSimulator, SimulatorConstants


class Runner:
    def __init__(self):
        wildfire_map = dataset.load_example_dataset()
        self.simulator = WildTorchSimulator(
            wildfire_map=wildfire_map,
            simulator_constants=SimulatorConstants(p_continue_burn=0.7, theta_w=180),
            initial_ignition=utils.create_ignition(shape=wildfire_map[0].shape),
        )
        self.logger = logger.Logger()

    def run(self):
        for i in range(200):
            self.simulator.step(force=True)
            self.logger.log_stats(
                step=i,
                num_cells_on_fire=metrics.cell_on_fire(self.simulator.fire_state).item(),
                num_cells_burned_out=metrics.cell_burned_out(self.simulator.fire_state).item(),
            )
            self.logger.snapshot_simulation(self.simulator)
            # self.logger.log_image_to_tensorboard(tag='fire_state',
            #                                      img_tensor=utils.compose_fire_state_with_map(self.simulator),
            #                                      global_step=i)
        self.logger.save_logs()
        self.logger.save_snapshots()
        # vis_fire_state = utils.compose_vis_fire_state(self.simulator.fire_state)
        # vis_wildfire_map = utils.compose_vis_wildfire_map(self.simulator.wildfire_map)
        # fire_state_image = utils.colorize_array(vis_fire_state)
        # wildfire_map_image = utils.colorize_array(vis_wildfire_map)
        #
        # plt.imshow(fire_state_image)
        # plt.show()
        # plt.imshow(wildfire_map_image)
        # plt.show()
        #
        # plt.imshow(utils.overlay_arrays(fire_state_image, wildfire_map_image))
        # plt.show()

        utils.animate_snapshots(self.logger.snapshots, self.simulator.wildfire_map)


if __name__ == "__main__":
    runner = Runner()
    runner.run()
