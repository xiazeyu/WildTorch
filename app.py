from typing import cast
import uuid
import os

import gradio as gr
import numpy as np
import pandas as pd
import torch

import wildtorch as wt

USE_OFFLINE_DATA = False
ENABLE_DOWNLOAD_SNAPSHOTS = False

if USE_OFFLINE_DATA:
    wildfire_sim_maps = torch.load('wildfire_sim_maps.pt')
else:
    wildfire_sim_maps = wt.dataset.load_wildfire_sim_maps()
    # torch.save(wildfire_sim_maps, 'wildfire_sim_maps.pt')

DEFAULT_SHAPE = (512, 512)
DEFAULT_STATE = {
    'ds': {
        'name': None,
        'shape': None,
        'data': None,
    },
    'constants': {
        'p_h': 0.58,
        'c_1': 0.045,
        'c_2': 0.131,
        'a': 0.078,
        'theta_w': 0,
        'v': 10,
        'p_firebreak': 0.9,
        'p_continue_burn': 0.6,
        'device': torch.device('cpu'),
        'dtype': torch.float32,
    },
    'ignition': None,
    'out_video_path': None,
    'snapshots_path': None,
    'checkpoint': None,
    'logger': None,
}

with (gr.Blocks() as demo):
    def remove_state_files(in_state):
        if in_state['out_video_path'] is not None:
            os.remove(in_state['out_video_path'])
        if in_state['snapshots_path'] is not None:
            os.remove(in_state['snapshots_path'])


    state_var = gr.State(DEFAULT_STATE, delete_callback=remove_state_files)
    with gr.Tabs(selected='tab_1') as tabs:
        with gr.Tab("1. Datasets", interactive=True, id='tab_1') as tab_1:
            sel_dataset = gr.Dropdown(cast(list, wildfire_sim_maps['name']) + ['empty'], label='Dataset')
            with gr.Row() as shape_row:
                sel_shape_h = gr.Number(label="Map Height", visible=False)
                sel_shape_w = gr.Number(label="Map Width", visible=False)

            with gr.Row() as preview_row:
                canopy_img = gr.Image(label="canopy")
                density_img = gr.Image(label="density")
                slope_img = gr.Image(label="slope")

            tab_1_confirm_btn = gr.Button("Confirm", interactive=True)


            @tab_1_confirm_btn.click(inputs=[state_var], outputs=[state_var, tabs])
            def jump_to_tab_2(in_state):
                return in_state, gr.Tabs(selected='tab_2')

        with gr.Tab("2. Simulation Constants and Initial Ignition", interactive=False, id='tab_2') as tab_2:
            with gr.Row():
                sel_p_h = gr.Slider(label="p_h",
                                    info="The probability that a burnable cell adjacent to a burning cell will "
                                         "catch fire at the next time step under normal conditions",
                                    value=DEFAULT_STATE['constants']['p_h'], minimum=0, maximum=1, step=0.01,
                                    interactive=True)
                sel_p_continue_burn = gr.Slider(label="p_continue_burn",
                                                info="The probability that a burning cell will continue to burn "
                                                     "at the next time step",
                                                value=DEFAULT_STATE['constants']['p_continue_burn'], minimum=0,
                                                maximum=1,
                                                step=0.01,
                                                interactive=True)
            with gr.Row():
                sel_a = gr.Slider(label="a",
                                  info="The coefficient of ground elevation",
                                  value=DEFAULT_STATE['constants']['a'], minimum=0, maximum=1, step=0.001,
                                  interactive=True)
                sel_p_firebreak = gr.Slider(label="p_firebreak",
                                            info="The probability that a burnable cell will not catch fire even "
                                                 "if it is adjacent to a burning cell",
                                            value=DEFAULT_STATE['constants']['p_firebreak'], minimum=0, maximum=1,
                                            step=0.01, interactive=True)
            with gr.Row():
                sel_c_1 = gr.Slider(label="c_1",
                                    info="The coefficient of wind velocity",
                                    value=DEFAULT_STATE['constants']['c_1'], minimum=0, maximum=1, step=0.001,
                                    interactive=True)
                sel_c_2 = gr.Slider(label="c_2",
                                    info="The coefficient of wind direction",
                                    value=DEFAULT_STATE['constants']['c_2'], minimum=0, maximum=1, step=0.001,
                                    interactive=True)
            with gr.Row():
                sel_theta_w = gr.Slider(label="theta_w",
                                        info="The direction of the wind in degrees, measured clockwise from north",
                                        value=DEFAULT_STATE['constants']['theta_w'], minimum=0, maximum=360, step=1,
                                        interactive=True)
                sel_v = gr.Slider(label="v",
                                  info="The wind velocity, unit in m/s",
                                  value=DEFAULT_STATE['constants']['v'], minimum=0, maximum=60, step=1,
                                  interactive=True)
            with gr.Row():
                sel_device = gr.Dropdown(label="device", choices=['cpu', 'cuda', 'mps'],
                                         info="The device to use",
                                         value='cpu', allow_custom_value=True, interactive=True)
                sel_dtype = gr.Dropdown(label="data type", choices=['float16', 'float32', 'float64'],
                                        info="The data type to use",
                                        value='float32', interactive=True)


            @gr.on(triggers=[sel_p_h.input, sel_c_1.input, sel_c_2.input, sel_a.input,
                             sel_theta_w.input, sel_v.input, sel_p_firebreak.input,
                             sel_p_continue_burn.input, sel_device.input, sel_dtype.input],
                   inputs=[state_var, sel_p_h, sel_c_1, sel_c_2, sel_a, sel_theta_w, sel_v,
                           sel_p_firebreak, sel_p_continue_burn, sel_device, sel_dtype],
                   outputs=[state_var])
            def update_constants_state(in_state, in_p_h, in_c_1, in_c_2, in_a, in_theta_w, in_v, in_p_firebreak,
                                       in_p_continue_burn,
                                       in_device, in_dtype):
                in_state['constants']['p_h'] = in_p_h
                in_state['constants']['c_1'] = in_c_1
                in_state['constants']['c_2'] = in_c_2
                in_state['constants']['a'] = in_a
                in_state['constants']['theta_w'] = in_theta_w
                in_state['constants']['v'] = in_v
                in_state['constants']['p_firebreak'] = in_p_firebreak
                in_state['constants']['p_continue_burn'] = in_p_continue_burn
                in_state['constants']['device'] = torch.device(in_device)
                in_state['constants']['dtype'] = {
                    'float16': torch.float16,
                    'float32': torch.float32,
                    'float64': torch.float64,
                }[in_dtype]
                return in_state


            sel_ignition_mode = gr.Dropdown(label="Initial Ignition", choices=['random', 'center', 'custom'],
                                            interactive=True)

            gr.Markdown(
                'to use custom ignition, please use the crop to fix the size, and then draw on the image. Please '
                'click on the green button once done. Drawing on the black will be good choices.')
            with gr.Row():
                custom_ignition_paint = gr.Paint(label="custom ignition", image_mode='L', interactive=True,
                                                 brush=gr.Brush(default_size=3, color_mode='fixed'))
                ignition_img_over_map = gr.Image(label="ignition over map")


            @gr.on(triggers=[sel_shape_h.input, sel_shape_w.input],
                   inputs=[state_var, sel_shape_h, sel_shape_w],
                   outputs=[state_var, canopy_img, density_img, slope_img, tab_2, custom_ignition_paint])
            def update_preview_row(in_state, in_h, in_w):
                shape = in_h, in_w
                data = wt.dataset.generate_empty_dataset(shape)
                in_state['ds']['shape'] = shape
                in_state['ds']['data'] = data
                in_state['ignition'] = None
                return in_state, gr.Image(wt.utils.colorize_array(np.array(data[0]))), gr.Image(
                    wt.utils.colorize_array(np.array(data[1]))), gr.Image(
                    wt.utils.colorize_array(np.array(data[2]))), gr.Tab(interactive=True), gr.ImageEditor(
                    crop_size=(shape[1], shape[0]))


            @sel_dataset.change(inputs=[state_var, sel_dataset],
                                outputs=[state_var, sel_shape_h, sel_shape_w, tab_2, custom_ignition_paint,
                                         canopy_img, density_img, slope_img])
            def update_shape_row(in_state, in_dataset):
                if in_dataset == 'empty':
                    shape = DEFAULT_SHAPE
                    data = wt.dataset.generate_empty_dataset(shape)
                    editable = True
                else:
                    idx_dict = {item['name']: index for index, item in enumerate(wildfire_sim_maps)}
                    shape = tuple(cast(torch.Tensor, wildfire_sim_maps[idx_dict[in_dataset]]['shape']).tolist())
                    data = wt.dataset.transform_wildfire_sim_map(wildfire_sim_maps[idx_dict[in_dataset]])
                    editable = False
                in_state['ds']['name'] = in_dataset
                in_state['ds']['shape'] = shape
                in_state['ds']['data'] = data
                return in_state, gr.Number(value=shape[0], interactive=editable, visible=True), gr.Number(
                    value=shape[1],
                    interactive=editable,
                    visible=True), gr.Tab(interactive=True), gr.ImageEditor(interactive=True,
                                                                            crop_size=(shape[1], shape[0])), gr.Image(
                    wt.utils.colorize_array(np.array(data[0]))), gr.Image(
                    wt.utils.colorize_array(np.array(data[1]))), gr.Image(
                    wt.utils.colorize_array(np.array(data[2])))


            tab_2_confirm_btn = gr.Button("Confirm", interactive=False)


            @sel_ignition_mode.input(
                inputs=[state_var, sel_ignition_mode, custom_ignition_paint],
                outputs=[state_var, ignition_img_over_map, tab_2_confirm_btn])
            def update_ignition_img(in_state, in_mode, in_custom):
                ignition = torch.zeros(in_state['ds']['shape'], dtype=torch.bool)

                if in_mode == 'random':
                    ignition = wt.utils.create_ignition(shape=in_state['ds']['shape'], mode='random-single')
                elif in_mode == 'center':
                    ignition = wt.utils.create_ignition(shape=in_state['ds']['shape'], mode='center')
                elif in_mode == 'custom':
                    if in_custom['composite'] is not None:
                        ignition_ndarray = in_custom['composite'] != 0
                        ignition = torch.tensor(ignition_ndarray)
                    else:
                        return in_state, gr.Image(
                            wt.utils.colorize_array(wt.utils.compose_vis_wildfire_map(in_state['ds']['data']),
                                                    cmap='grey')), gr.Button(interactive=False)

                in_state['ignition'] = ignition
                ignition_ndarray = wt.utils.to_ndarray(ignition)

                ignition__over_map = wt.utils.overlay_arrays(
                    wt.utils.colorize_array(ignition_ndarray),
                    wt.utils.colorize_array(wt.utils.compose_vis_wildfire_map(in_state['ds']['data']),
                                            cmap='grey'),
                    0.5
                )

                return in_state, gr.Image((ignition__over_map * 255).astype(np.uint8)), gr.Button(interactive=True)


            @custom_ignition_paint.change(
                inputs=[state_var, custom_ignition_paint],
                outputs=[state_var, sel_ignition_mode, ignition_img_over_map, tab_2_confirm_btn])
            def update_ignition_img_over_map(in_state, in_custom):
                if in_custom['composite'] is not None:
                    ignition_ndarray = in_custom['composite'] != 0
                    ignition = torch.tensor(ignition_ndarray)
                else:
                    return in_state, gr.Dropdown(), gr.Image(), gr.Button()
                in_state['ignition'] = ignition

                ignition__over_map = wt.utils.overlay_arrays(
                    wt.utils.colorize_array(ignition_ndarray),
                    wt.utils.colorize_array(wt.utils.compose_vis_wildfire_map(in_state['ds']['data']),
                                            cmap='grey'),
                    0.5
                )

                return in_state, gr.Dropdown(value='custom'), gr.Image(
                    (ignition__over_map * 255).astype(np.uint8)), gr.Button(interactive=True)

        with gr.Tab("3. Simulation Control", interactive=False, id='tab_3') as tab_3:
            @tab_2_confirm_btn.click(inputs=[state_var], outputs=[state_var, tabs, tab_3])
            def update_tab_34_components(in_state):
                return in_state, gr.Tabs(selected='tab_3'), gr.Tab(interactive=True)


            with gr.Row():
                with gr.Column():
                    gr.Markdown("## Memory Control")
                    checkpoint_cb = gr.Checkbox(label="Checkpoint -> Memory", value=False, interactive=True)
                    run_from_cp_cb = gr.Checkbox(label="Begin from Memory", value=False, interactive=True)
                    reset_btn = gr.Button("Reset Memory", interactive=True)
                with gr.Column():
                    gr.Markdown("## Misc Control")
                    sel_steps = gr.Number(label="Number of Steps", value=200, minimum=1, step=1, interactive=True)
                    auto_run_cb = gr.Checkbox(label="Auto Run", value=False, interactive=True)
                    auto_reseed_cb = gr.Checkbox(label="Auto Regenerate Seed when open Tab", value=False,
                                                 interactive=True)
                    track_p_burn_cb = gr.Checkbox(label="Track p(burn), slow", value=False, interactive=True)
                with gr.Column():
                    gr.Markdown("## Random Seed Control")
                    sel_seed = gr.Number(label="Random Seed", value=torch.Generator().seed(), minimum=0, step=1,
                                         interactive=True)
                    random_seed_btn = gr.Button("Randomize Seed", interactive=True)


                @random_seed_btn.click(inputs=[state_var], outputs=[state_var, sel_seed])
                def randomize_seed(in_state):
                    return in_state, torch.Generator().seed()

            with gr.Row():
                run_btn = gr.Button("Run Simulation", interactive=True)
                download_snap_btn = gr.DownloadButton(label="Download Snapshots", interactive=False, visible=False)

            progress_bar = gr.Progress(track_tqdm=True)

            with gr.Row():
                output_video = gr.Video(label="Simulation Video", interactive=False, autoplay=True)

                stats_plot = gr.LinePlot(title="Simulation Stats", interactive=True, height=600,
                                         width=600, )

        with gr.Tab("4. Advanced Simulation", interactive=False, id='tab_4') as tab_4:

            sel_tab4_step = gr.Slider(label='Step', minimum=0, step=1, value=0, interactive=True)
            with gr.Row():
                cof_tb = gr.Textbox(label='cell_on_fire', interactive=False)
                cbo_tb = gr.Textbox(label='cell_burned_out', interactive=False)
            with gr.Row():
                fire_state_img = gr.Image(label="Fire State", interactive=False)
                p_burn_plot = gr.Image(label="p(burn)", interactive=False)  # gr.Plot is bad at presenting
            stats_df = gr.DataFrame()


            @sel_tab4_step.input(inputs=[state_var, sel_tab4_step],
                                 outputs=[state_var, fire_state_img, p_burn_plot, cof_tb, cbo_tb])
            def update_tab4_step(in_state, in_user_step):

                o_fsi, o_pbp = gr.Image(), gr.Image()
                o_cof_tb, o_cbo_tb = gr.Textbox(), gr.Textbox()

                if in_state['logger'] is not None:
                    snapshot = in_state['logger'].snapshots[in_user_step]
                    log = in_state['logger'].logs[in_user_step]
                    o_fsi = gr.Image(
                        value=wt.utils.colorize_array(wt.utils.compose_vis_fire_state(snapshot['fire_state'])))
                    if len(in_state['logger'].p_burns) > 0:
                        p_burn_arr = in_state['logger'].p_burns[in_user_step].cpu().numpy()
                        o_pbp = gr.Image(
                            value=wt.utils.colorize_array(p_burn_arr))
                    o_cof_tb = gr.Textbox(value=str(log['num_cells_on_fire']))
                    o_cbo_tb = gr.Textbox(value=str(log['num_cells_burned_out']))

                return in_state, o_fsi, o_pbp, o_cof_tb, o_cbo_tb


            @tab_3.select(
                inputs=[state_var, auto_run_cb, sel_steps, sel_seed, auto_reseed_cb, checkpoint_cb, run_from_cp_cb,
                        track_p_burn_cb],
                outputs=[state_var, output_video, tab_4, stats_plot, download_snap_btn, sel_seed, sel_tab4_step,
                         stats_df])
            def auto_run_simulation(in_state, in_auto_run, in_steps, in_seed, in_auto_reseed, in_checkpoint_cb,
                                    in_run_from_cp_cb, in_track_p_burn_cb):
                o_s = in_state
                o_v = gr.Video()
                o_t = gr.Tab()
                o_lp = gr.LinePlot()
                o_dsb = gr.DownloadButton()
                o_ts = gr.Slider()
                o_sdf = pd.DataFrame()
                if in_auto_reseed:
                    in_seed = torch.Generator().seed()
                if in_auto_run:
                    o_s, o_v, o_t, o_lp, o_dsb, o_ts, o_sdf = run_simulation(in_state,
                                                                             in_steps,
                                                                             in_seed,
                                                                             in_checkpoint_cb,
                                                                             in_run_from_cp_cb,
                                                                             in_track_p_burn_cb)
                return o_s, o_v, o_t, o_lp, o_dsb, in_seed, o_ts, o_sdf


            @reset_btn.click(inputs=[state_var],
                             outputs=[state_var])
            def reset_simulation(in_state):
                if in_state['checkpoint'] is not None:
                    in_state['checkpoint'] = None
                    gr.Info('Checkpoint Cleared.')
                return in_state


            @run_btn.click(inputs=[state_var, sel_steps, sel_seed, checkpoint_cb, run_from_cp_cb, track_p_burn_cb],
                           outputs=[state_var, output_video, tab_4, stats_plot, download_snap_btn, sel_tab4_step,
                                    stats_df])
            def run_simulation(in_state, in_steps, in_seed, in_checkpoint_cb, in_run_from_cp_cb, in_track_p_burn_cb,
                               in_progress=gr.Progress(track_tqdm=True)):
                if in_state['out_video_path'] is None:
                    in_state['out_video_path'] = f'runs/{str(uuid.uuid4())}.mp4'
                simulator = wt.WildTorchSimulator(
                    wildfire_map=in_state['ds']['data'],
                    simulator_constants=wt.SimulatorConstants(
                        p_h=in_state['constants']['p_h'],
                        c_1=in_state['constants']['c_1'],
                        c_2=in_state['constants']['c_2'],
                        a=in_state['constants']['a'],
                        theta_w=in_state['constants']['theta_w'],
                        v=in_state['constants']['v'],
                        p_firebreak=in_state['constants']['p_firebreak'],
                        p_continue_burn=in_state['constants']['p_continue_burn'],
                        device=in_state['constants']['device'],
                        dtype=in_state['constants']['dtype'],
                    ),
                    maximum_step=in_steps,
                    initial_ignition=in_state['ignition'],
                    seed=in_seed,
                )

                if in_state['checkpoint'] is not None and in_run_from_cp_cb:
                    simulator.load_checkpoint(in_state['checkpoint'], restore_seed=False)

                logger = wt.logger.Logger(disable_writing=True, verbose=False)

                for i in in_progress.tqdm(range(in_steps)):
                    simulator.step()
                    logger.snapshot_simulation(simulator)
                    logger.log_stats(
                        step=i,
                        num_cells_on_fire=wt.metrics.cell_on_fire(simulator.fire_state).item(),
                        num_cells_burned_out=wt.metrics.cell_burned_out(simulator.fire_state).item(),
                    )
                    if in_track_p_burn_cb:
                        logger.log_p_burn(simulator)

                gr.Info('Simulation Completed. Generating Video ...')

                in_state['logger'] = logger

                if in_checkpoint_cb:
                    in_state['checkpoint'] = simulator.checkpoint

                if ENABLE_DOWNLOAD_SNAPSHOTS:
                    logger.snapshots_filepath = in_state['snapshots_path'] = f'runs/{str(uuid.uuid4())}.pt'
                    logger.save_snapshots()
                    can_download_snapshots = True
                else:
                    can_download_snapshots = False

                wt.utils.animate_snapshots(logger.snapshots, simulator.wildfire_map,
                                           out_filename=in_state['out_video_path'])

                m_stats_df = pd.DataFrame(logger.logs)
                m_stats_df = m_stats_df.melt(id_vars=["step"], var_name="key", value_name="value")

                o_stats_df = pd.DataFrame(logger.logs)
                return in_state, gr.Video(value=in_state['out_video_path']), gr.Tab(interactive=True), gr.LinePlot(
                    m_stats_df, x='step', y='value', color="key", color_legend_position="bottom",
                    tooltip=["step", "key", "value"], container=False, ), gr.DownloadButton(
                    value=in_state['snapshots_path'], interactive=can_download_snapshots,
                    visible=can_download_snapshots), gr.Slider(maximum=in_steps - 1), o_stats_df

demo.queue().launch(share=True)
