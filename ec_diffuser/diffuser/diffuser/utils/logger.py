import os
import json


class Logger:

    def __init__(self, renderer, logpath, vis_freq=10, max_render=8):
        self.renderer = renderer
        self.savepath = logpath
        self.vis_freq = vis_freq
        self.max_render = max_render

    def log(self, t, samples, state, rollout=None, goal=None, front_bg=None, side_bg=None, traj_number=None, **kwargs):
        if self.vis_freq == 999 or t % self.vis_freq != 0:
            return

        ## render image of plans
        if hasattr(self.renderer, 'require_bg'):
            if traj_number is not None:
                self.renderer.composite(
                    os.path.join(self.savepath, f'traj{traj_number}_{t}.png'),
                    samples.observations[:self.max_render],
                    front_bg,
                    side_bg,
                    **kwargs,
                )
            else:
                self.renderer.composite(
                os.path.join(self.savepath, f'{t}.png'),
                samples.observations[:self.max_render],
                front_bg,
                side_bg,
                **kwargs,
            )
        else:
            self.renderer.composite(
                os.path.join(self.savepath, f'{t}.png'),
                samples.observations[:self.max_render],
                **kwargs,
            )

        ## render video of plans
        # self.renderer.render_plan(
        #     os.path.join(self.savepath, f'{t}_plan.mp4'),
        #     samples.actions[:self.max_render],
        #     samples.observations[:self.max_render],
        #     state,
        # )

        if not hasattr(self.renderer, 'require_bg') and rollout is not None:
            ## render video of rollout thus far
            self.renderer.render_rollout(
                os.path.join(self.savepath, f'rollout.mp4'),
                rollout,
                goal,
                fps=80,
            )
    def log_joint(self, rollout1, goal1, rollout2, goal2):
        ## render video of rollout thus far
        self.renderer.render_rollout_joint(
            os.path.join(self.savepath, f'rollout_joint.mp4'),
            rollout1,
            goal1,
            rollout2,
            goal2,
            fps=80,
        )

    def finish(self, t, score, total_reward, terminal, diffusion_experiment, value_experiment):
        json_path = os.path.join(self.savepath, 'rollout.json')
        json_data = {'score': score, 'step': t, 'return': total_reward, 'term': terminal,
            'epoch_diffusion': diffusion_experiment.epoch, 'epoch_value': value_experiment.epoch}
        json.dump(json_data, open(json_path, 'w'), indent=2, sort_keys=True)
        print(f'[ utils/logger ] Saved log to {json_path}')
