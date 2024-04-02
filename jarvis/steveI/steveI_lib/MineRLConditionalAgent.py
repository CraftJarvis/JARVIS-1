import numpy as np
import torch as th
import inspect
from gym3.types import DictType

from jarvis.steveI.steveI_lib.VPT.agent import MineRLAgent, resize_image, AGENT_RESOLUTION, \
     default_device_type, set_default_torch_device, CameraHierarchicalMapping, \
     ActionTransformer, ACTION_TRANSFORMER_KWARGS, POLICY_KWARGS, PI_HEAD_KWARGS
from jarvis.steveI.steveI_lib.embed_conditioned_policy import MinecraftAgentPolicy


class MineRLConditionalAgent(MineRLAgent):
    def __init__(self, device=None, policy_kwargs=None, pi_head_kwargs=None):
        """Same as parent but with different policy (only different import).
        (MinecraftAgentPolicy from embed_conditioned_policy.py, not from VPT.policy)
        """

        if device is None:
            device = default_device_type()
        self.device = th.device(device)
        # Set the default torch device for underlying code as well
        set_default_torch_device(self.device)
        self.action_mapper = CameraHierarchicalMapping(n_camera_bins=11)
        action_space = self.action_mapper.get_action_space_update()
        action_space = DictType(**action_space)

        self.action_transformer = ActionTransformer(**ACTION_TRANSFORMER_KWARGS)

        if policy_kwargs is None:
            policy_kwargs = POLICY_KWARGS
        if pi_head_kwargs is None:
            pi_head_kwargs = PI_HEAD_KWARGS

        agent_kwargs = dict(policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs, action_space=action_space)

        self.policy = MinecraftAgentPolicy(**agent_kwargs).to(device)
        self.reset(cond_scale=None)
        self._dummy_first = th.from_numpy(np.array((False,))).to(device)

    def reset(self, cond_scale=None):
        """Reset agent to initial state (i.e., reset hidden state)
        If cond_scale is None, we use a batch size of 1. Otherwise,
        we use a batch size of two, since we are using the classifier-free
        guidance version of the policy."""
        if cond_scale is None:
            self.hidden_state = self.policy.initial_state(1)
        else:
            self.hidden_state = self.policy.initial_state(2)
        self.cond_scale = cond_scale

    def get_agent_input_pov(self, frame: np.ndarray):
        agent_input_pov = resize_image(frame, AGENT_RESOLUTION)[None]
        return agent_input_pov

    def take_action_on_frame(self, agent_input_pov: np.ndarray):
        """
        Get agent's action for given agent_input_pov (not for minerl_obs like get_action()).

        Agent's hidden state is tracked internally. To reset it,
        call `reset()`.
        """
        # Changed line from parent, doesn't, used to be: agent_input = self._env_obs_to_agent()minerl_obs
        agent_input = {"img": th.from_numpy(agent_input_pov).to(self.device)}

        # The "first" argument could be used to reset tell episode
        # boundaries, but we are only using this for predicting (for now),
        # so we do not hassle with it yet.
        agent_action, self.hidden_state, _ = self.policy.act(
            agent_input, self._dummy_first, self.hidden_state,
            stochastic=True
        )
        minerl_action = self._agent_action_to_env(agent_action)
        return minerl_action

    def get_action(self, minerl_obs, goal_embed):
        """
        Get agent's action for given MineRL observation.

        Agent's hidden state is tracked internally. To reset it,
        call `reset()`.
        """
        agent_input = self._env_obs_to_agent(minerl_obs, goal_embed)
        # The "first" argument could be used to reset tell episode
        # boundaries, but we are only using this for predicting (for now),
        # so we do not hassle with it yet.
        agent_action, self.hidden_state, _ = self.policy.act(
            agent_input, self._dummy_first, self.hidden_state,
            stochastic=True, cond_scale=self.cond_scale
        )
        minerl_action = self._agent_action_to_env(agent_action)
        return minerl_action

    def _env_obs_to_agent(self, minerl_obs, goal_embed, device=None):
        """
        Turn observation from MineRL environment into model's observation

        Returns torch tensors.
        """
        if device is None:
            device = self.device

        agent_input = resize_image(minerl_obs["img"], AGENT_RESOLUTION)[None]
        agent_input = {"img": th.from_numpy(agent_input).to(device)}

        # MODIFIED
        agent_input['mineclip_embed'] = th.from_numpy(goal_embed).to(device)

        return agent_input

    def _env_action_to_agent(self, minerl_action_transformed, to_torch=False, check_if_null=False, device=None):
        """
        Turn action from MineRL to model's action.

        Note that this will add batch dimensions to the action.
        Returns numpy arrays, unless `to_torch` is True, in which case it returns torch tensors.

        If `check_if_null` is True, check if the action is null (no action) after the initial
        transformation. This matches the behaviour done in OpenAI's VPT work.
        If action is null, return "None" instead
        """
        if device is None:
            device = self.device

        minerl_action = self.action_transformer.env2policy(minerl_action_transformed)
        if check_if_null:
            if np.all(minerl_action["buttons"] == 0) and np.all(minerl_action["camera"] == self.action_transformer.camera_zero_bin):
                return None

        # Add batch dims if not existant
        if minerl_action["camera"].ndim == 1:
            minerl_action = {k: v[None] for k, v in minerl_action.items()}
        action = self.action_mapper.from_factored(minerl_action)
        if to_torch:
            action = {k: th.from_numpy(v).to(device) for k, v in action.items()}
        return action
    
def configure_optimizers(model, weight_decay, learning_rate):

    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (th.nn.Linear, )
    blacklist_weight_modules = (th.nn.LayerNorm, th.nn.Embedding)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
            # random note: because named_modules and named_parameters are recursive
            # we will see the same tensors p many many times. but doing it this way
            # allows us to know which parent module any tensor p belongs to...
            if pn.endswith('bias'):
                # all biases will not be decayed
                no_decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)
            else:
                decay.add(fpn)

    # If a parameter is in both decay and no_decay, remove it from decay.
    decay = decay - no_decay

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
    assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                % (str(param_dict.keys() - union_params), )
    
    # Print the keys that are in each set in a comma-separated list.
    # print(f"decay keys: {', '.join(sorted(list(decay)))}")
    # print(f"no decay keys: {', '.join(sorted(list(no_decay)))}")

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
    use_fused = 'fused' in inspect.signature(th.optim.AdamW).parameters
    print(f"using fused AdamW: {use_fused}")
    extra_args = dict(fused=True) if use_fused else dict()

    # Make sure that all parameters are CUDA and floating point Tensor.
    # This is required by fused optimizer.
    # for group in optim_groups:
    #     for p in group["params"]:
    #         if not p.is_cuda:
    #             print(f"WARNING: parameter {p} is not CUDA tensor. Fixing for AdamW with fused optimizer.")
    #             p.data = p.data.cuda()
    #         if not p.is_floating_point():
    #             print(f"WARNING: parameter {p} is not floating point type. Fixing for AdamW with fused optimizer.")
    #             p.data = p.data.float()
    optimizer = th.optim.AdamW(optim_groups, lr=learning_rate, **extra_args)

    return optimizer