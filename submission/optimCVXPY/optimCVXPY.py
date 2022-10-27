# coding=utf-8
# Copyright (c) 2019-2021, Alibaba Group. All rights reserved.
#
# Licensed under the Mozilla Public License (MPL) v2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.mozilla.org/en-US/MPL/2.0/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Given an observation, this tutor does DCOPF optimization to get the best continuous actions.

This code is mainly based on l2rpn_baselines.OptimCVXPY, with the following differences:
- modify the workflow of the original code, mainly on act() method.
- include brutal force search of discrete actions. 
- revise and comment the code to make it clearer.
"""
import os 
import copy
import logging
import warnings
import cvxpy as cp
import numpy as np

from grid2op.Agent import BaseAgent
from grid2op.Action import BaseAction
from grid2op.Backend import PandaPowerBackend
from grid2op.Observation import BaseObservation
from lightsim2grid import LightSimBackend
from lightsim2grid.gridmodel import init

logger = logging.getLogger(__name__)


OPTIMCVXPY_CONFIG = {
    # margin_th_limit: `float`
    #     In the "unsafe state" this agent will try to minimize the thermal limit violation.
    #     A "thermal limit violation" is defined as having a flow (in dc) above 
    #     `margin_th_limit * thermal_limit_mw`.
    #     The model is particularly sensitive to this parameter.
    "margin_th_limit": 0.9,
    # alpha_por_error: `float`
    #     alpha_por_error takes into account the previous errors on the flows (in an additive fashion)
    #     new flows are 1/x(theta_or - theta_ex) * alpha_por_error . (prev_flows - obs.p_or)
    "alpha_por_error": 0.5,
    # rho_danger: `float`
    #     If any `obs.rho` is above `rho_danger`, then the agent will use the
    #     "unsafe grid" optimization routine and try to apply curtailment,
    #     redispatching and action on storage unit to set back the grid into a safe state.
    "rho_danger": 0.95,
    # rho_safe: `float`
    #     If all `obs.rho` are below `rho_safe`, then the agent will use the
    #     "safe grid" optimization routine and try to set back the grid into
    #     a reference state.
    "rho_safe": 0.85,
    # penalty_curtailment_unsafe: `float`
    #     The cost of applying a curtailment in the objective function. Applies only in "unsafe" mode.
    #     Default value is 0.1, should be >= 0.
    "penalty_curtailment_unsafe": 0.1,
    # penalty_redispatching_unsafe: `float`
    #     The cost of applying a redispatching in the objective function. Applies only in "unsafe" mode.
    #     Default value is 0.03, should be >= 0.
    "penalty_redispatching_unsafe": 0.03,
    # penalty_storage_unsafe: `float`
    #     The cost of applying a storage in the objective function. Applies only in "unsafe" mode.
    #     Default value is 0.3, should be >= 0.
    "penalty_storage_unsafe": 0.3,
    # penalty_curtailment_safe: `float`
    #     The cost of applying a curtailment in the objective function. Applies only in "safe" mode.
    #     Default value is 0.0, should be >= 0.
    "penalty_curtailment_safe": 0.0,
    # penalty_redispatching_safe: `float`
    #     The cost of applying a redispatching in the objective function. Applies only in "safe" mode.
    #     Default value is 0.0, should be >= 0.
    "penalty_redispatching_safe": 0.0,
    # penalty_storage_safe: `float`
    #     The cost of applying a storage in the objective function. Applies only in "safe" mode.
    #     Default value is 0.0, should be >= 0.
    "penalty_storage_safe": 0.0,
    # weight_storage_target: `float`
    "weight_redisp_target": 1.0,
    # weight_redisp_target: `float`
    "weight_storage_target": 1.0,
    # weight_curtail_target: `float`
    "weight_curtail_target": 1.0,
    # margin_rounding: `float`
    #     A margin taken to avoid rounding issues that could lead to infeasible
    #     actions due to "redispatching above max_ramp_up" for example.
    "margin_rounding": 0.01,
    # margin_sparse: `float`
    #     A margin taken when converting the output of the optimization routine
    #     to grid2op actions: if some values are below this value, then they are
    #     set to zero.
    "margin_sparse": 5e-3,
    # max_iter: `int`
    # maximum iteration for OSQP optimization
    "max_iter": 50000,
}


def convert_from_vect(action_space, act):
    """
    Helper to convert an action, represented as a numpy array as an :class:`grid2op.BaseAction` instance.

    Parameters
    ----------
    act: ``numppy.ndarray``
        An action cast as an :class:`grid2op.BaseAction.BaseAction` instance.

    Returns
    -------
    res: :class:`grid2op.Action.Action`
        The `act` parameters converted into a proper :class:`grid2op.BaseAction.BaseAction` object.
    """
    res = action_space({})
    res.from_vect(act)
    return res


def load_action_to_grid2op(action_space, action_vec_path, action_threshold=None, return_counts=False):
    """Load actions saved in npz file and convert them to grid2op action class instances. 
    
    Args:
        action_space: obtain from env.action_space
        action_vec_path: a path ending with .npz. The loaded all_actions has two keys 'action_space', 'counts'.
            all_actions['action_space'] is a (N, K) matrix, where each row is an action.
        action_threshold: if provided, will filter out actions with counts below this.
        return_counts: if True, return the counts.
    """
    # load all_actions from npz file with two keys 'action_space', 'counts'
    assert action_vec_path.endswith('.npz') and os.path.exists(action_vec_path), FileNotFoundError(
        f'file_paths {action_vec_path} does not contain a valid npz file.')
    data = np.load(action_vec_path, allow_pickle=True)
    all_actions = data['action_space']
    # verify loaded all_actions
    assert isinstance(all_actions, np.ndarray), RuntimeError(
        f'Expect {action_vec_path} to be an ndarray, got {type(all_actions)}')
    action_dim = action_space.size()
    assert all_actions.shape[1] == action_dim, RuntimeError(
        f'Expect {action_vec_path} to be an ndarray of shape {action_dim}, got {all_actions.shape[1]}')
    counts = data['counts']
    if action_threshold is not None:
        num_actions = (counts >= action_threshold).sum()
        all_actions = all_actions[:num_actions]
        counts = counts[:num_actions]
        logger.info('{} actions loaded.'.format(num_actions))
    all_actions = [convert_from_vect(action_space, action) for action in all_actions]

    if return_counts:
        return all_actions, counts
    return all_actions


class OptimCVXPY(BaseAgent):
    """
    This agent choses its action by resolving, at each `agent.act(...)` call an optimization routine
    that is then converted to a grid2op action.
    It has 3 main behaviours:
    - `safe grid`: when the grid is safe, it tries to reconnect any disconnected lines. 
    - `unsafe grid`: when the grid is unsafe, it tries to set it back to a "safe" state (all flows
    below their thermal limit). The strategy is:
        - Try to reconnect lines
        - Try to reset the grid to reference state (all elements to bus one)
        - If back2ref does not work, seach topology action.
        - optimizing storage units, curtailment and redispatching.
    
    See OPTIMCVXPY_CONFIG for a list of default configs.
    """
    SOLVER_TYPES = [cp.OSQP, cp.SCS, cp.SCIPY]
    # NB: SCIPY rarely converge
    # SCS converged almost all the time, but is inaccurate.

    def __init__(
        self, 
        env, 
        action_space, 
        action_space_path,
        config: OPTIMCVXPY_CONFIG,
        time_step=1,
        verbose=1,
        ):
        BaseAgent.__init__(self, action_space)
        self.do_nothing = action_space({})

        self.config = config
        self._get_grid_info(env)
        self._init_params(env)
        self._load_topo_actions(action_space_path)

        # other arguments
        self.max_iter = config["max_iter"]
        self.flow_computed = np.full(env.n_line, np.NaN, dtype=float)  # for debug only
        self.time_step = time_step
        self.verbose = verbose

    def _get_grid_info(self, env):
        self.n_line = env.n_line
        self.n_sub = env.n_sub
        self.n_load = env.n_load
        self.n_gen = env.n_gen
        self.n_storage = env.n_storage
        
        self.line_or_to_subid = copy.deepcopy(env.line_or_to_subid)
        self.line_ex_to_subid = copy.deepcopy(env.line_ex_to_subid)
        self.load_to_subid = copy.deepcopy(env.load_to_subid)
        self.gen_to_subid = copy.deepcopy(env.gen_to_subid)
        self.storage_to_subid = copy.deepcopy(env.storage_to_subid)
        self.storage_Emax = copy.deepcopy(env.storage_Emax)

    def _init_params(self, env):

        self.margin_rounding = float(self.config["margin_rounding"])
        self.margin_sparse = float(self.config["margin_sparse"])
        self.rho_danger = float(self.config["rho_danger"])
        self.rho_safe = float(self.config["rho_safe"])

        self._margin_th_limit = cp.Parameter(
            value=self.config["margin_th_limit"], nonneg=True)
        self._penalty_curtailment_unsafe = cp.Parameter(
            value=self.config["penalty_curtailment_unsafe"], nonneg=True)
        self._penalty_redispatching_unsafe = cp.Parameter(
            value=self.config["penalty_redispatching_unsafe"], nonneg=True)
        self._penalty_storage_unsafe = cp.Parameter(
            value=self.config["penalty_storage_unsafe"], nonneg=True)
        self._penalty_curtailment_safe = cp.Parameter(
            value=self.config["penalty_curtailment_safe"], nonneg=True)
        self._penalty_redispatching_safe = cp.Parameter(
            value=self.config["penalty_redispatching_safe"], nonneg=True)
        self._penalty_storage_safe = cp.Parameter(
            value=self.config["penalty_storage_safe"], nonneg=True)
        self._weight_redisp_target = cp.Parameter(
            value=self.config["weight_redisp_target"], nonneg=True)
        self._weight_storage_target = cp.Parameter(
            value=self.config["weight_storage_target"], nonneg=True)
        self._weight_curtail_target = cp.Parameter(
            value=self.config["weight_curtail_target"], nonneg=True)
        self._alpha_por_error = cp.Parameter(
            value=self.config["alpha_por_error"], nonneg=True)

        self.nb_max_bus = 2 * self.n_sub
        self._storage_setpoint = 0.5 * self.storage_Emax     
        SoC = np.zeros(shape=self.nb_max_bus)
        for bus_id in range(self.nb_max_bus):
            SoC[bus_id] = 0.5 * self._storage_setpoint[self.storage_to_subid == bus_id].sum()
        self._storage_target_bus = cp.Parameter(  # use 1.0* as modifying cp.Parameter will modify its value array
            shape=self.nb_max_bus, value=1.0 * SoC, nonneg=True)
        self._storage_power_obs = cp.Parameter(value=0.)

        powerlines_x, powerlines_g, powerlines_b, powerlines_ratio = self._get_powerline_impedance(env)
        self._powerlines_x = cp.Parameter(  # use 1.0* as modifying cp.Parameter will modify its value array
            shape=powerlines_x.shape, value=1.0 * powerlines_x, pos=True)
        self._powerlines_g = cp.Parameter(  # use 1.0* as modifying cp.Parameter will modify its value array
            shape=powerlines_x.shape, value=1.0 * powerlines_g, pos=True)
        self._powerlines_b = cp.Parameter(  # use 1.0* as modifying cp.Parameter will modify its value array
            shape=powerlines_x.shape, value=1.0 * powerlines_b, neg=True)   
        self._powerlines_ratio = cp.Parameter(  # use 1.0* as modifying cp.Parameter will modify its value array
            shape=powerlines_x.shape, value=1.0 * powerlines_ratio, pos=True)
        self._prev_por_error = cp.Parameter(
            shape=powerlines_x.shape, value=np.zeros(env.n_line))

        self.vm_or = cp.Parameter(
            shape=self.n_line, value=np.ones(self.n_line), pos=True)
        self.vm_ex = cp.Parameter(
            shape=self.n_line, value=np.ones(self.n_line), pos=True)    

        self.bus_or = cp.Parameter(  # use 1* as modifying cp.Parameter will modify its value array
            shape=self.n_line, value=1 * self.line_or_to_subid, integer=True)
        self.bus_ex = cp.Parameter(  # use 1* as modifying cp.Parameter will modify its value array
            shape=self.n_line, value=1 * self.line_ex_to_subid, integer=True)
        self.bus_load = cp.Parameter(  # use 1* as modifying cp.Parameter will modify its value array
            shape=self.n_load, value=1 * self.load_to_subid, integer=True)
        self.bus_gen = cp.Parameter(  # use 1* as modifying cp.Parameter will modify its value array
            shape=self.n_gen, value=1 * self.gen_to_subid, integer=True)
        self.bus_storage = cp.Parameter(  # use 1* as modifying cp.Parameter will modify its value array
            shape=self.n_storage, value=1 * self.storage_to_subid, integer=True)

        this_zeros_ = np.zeros(self.nb_max_bus)
        self.load_per_bus = cp.Parameter(  # use 1.0* as modifying cp.Parameter will modify its value array
            shape=self.nb_max_bus, value=1.0 * this_zeros_, nonneg=True)
        self.gen_per_bus = cp.Parameter(  # use 1.0* as modifying cp.Parameter will modify its value array
            shape=self.nb_max_bus, value=1.0 * this_zeros_, nonneg=True)
        self.redisp_up = cp.Parameter(  # use 1.0* as modifying cp.Parameter will modify its value array
            shape=self.nb_max_bus, value=1.0 * this_zeros_, nonneg=True)
        self.redisp_down = cp.Parameter(  # use 1.0* as modifying cp.Parameter will modify its value array
            shape=self.nb_max_bus, value=1.0 * this_zeros_, nonneg=True)
        self.curtail_down = cp.Parameter(  # use 1.0* as modifying cp.Parameter will modify its value array
            shape=self.nb_max_bus, value=1.0 * this_zeros_, nonneg=True)
        self.curtail_up = cp.Parameter(  # use 1.0* as modifying cp.Parameter will modify its value array
            shape=self.nb_max_bus, value=1.0 * this_zeros_, nonneg=True)
        self.storage_down = cp.Parameter(  # use 1.0* as modifying cp.Parameter will modify its value array
            shape=self.nb_max_bus, value=1.0 * this_zeros_, nonneg=True)
        self.storage_up = cp.Parameter(  # use 1.0* as modifying cp.Parameter will modify its value array
            shape=self.nb_max_bus, value=1.0 * this_zeros_, nonneg=True)
        self._th_lim_mw = cp.Parameter(  # use 1.0* as modifying cp.Parameter will modify its value array
            shape=self.n_line, value=1.0 * env.get_thermal_limit(), nonneg=True)
        
        self._past_dispatch = cp.Parameter(
            shape=self.nb_max_bus, value=np.zeros(self.nb_max_bus)) 
        self._past_state_of_charge = cp.Parameter(
            shape=self.nb_max_bus, value=np.zeros(self.nb_max_bus), nonneg=True)
        
    def _get_powerline_impedance(self, env):
        # read the powerline impedance from lightsim grid or pandapower grid
        if isinstance(env.backend, LightSimBackend): 
            line_info = env.backend._grid.get_lines()
            trafo_info = env.backend._grid.get_trafos()
        elif isinstance(env.backend, PandaPowerBackend):
            pp_net = env.backend._grid
            grid_model = init(pp_net) 
            line_info = grid_model.get_lines()
            trafo_info = grid_model.get_trafos()
        else:
            # no powerline information available
            raise RuntimeError(
                f"Unkown backend type: {type(env.backend)}. If you want to use "
                "OptimCVXPY, you need to provide the reactance of each powerline / "
                "transformer in per unit in the `lines_x` parameter.")

        powerlines_x = np.array(
            [float(el.x_pu) for el in line_info] + 
            [float(el.x_pu) for el in trafo_info])
        powerlines_g = np.array(
            [(1 / (el.r_pu + 1j * el.x_pu)).real for el in line_info] + 
            [(1 / (el.r_pu + 1j * el.x_pu)).real for el in trafo_info])
        powerlines_b = np.array(
                [(1 / (el.r_pu + 1j * el.x_pu)).imag for el in line_info] + 
                [(1 / (el.r_pu + 1j * el.x_pu)).imag for el in trafo_info])
        powerlines_ratio = np.array(
                [1.0] * len(line_info) + 
                [el.ratio for el in trafo_info])

        return powerlines_x, powerlines_g, powerlines_b, powerlines_ratio

    def _load_topo_actions(self, action_space_path):
        # load actions from the actions_space.npz
        self.topo_actions_to_check = load_action_to_grid2op(
            self.action_space, action_space_path)
        self.num_topo_actions_to_check = len(self.topo_actions_to_check)

    def _update_topo_param(self, obs: BaseObservation):
        # update topology to record disconnected lines and changed busbars of all elements.
        tmp_ = 1 * obs.line_or_to_subid
        tmp_ [obs.line_or_bus == 2] += obs.n_sub
        self.bus_or.value[:] = tmp_
        tmp_ = 1 * obs.line_ex_to_subid
        tmp_ [obs.line_ex_bus == 2] += obs.n_sub
        self.bus_ex.value[:] = tmp_
        
        # "disconnect" in the model the line disconnected
        # it should be equilavent to connect them all (at both side) to the slack
        self.bus_ex.value [(obs.line_or_bus == -1) | (obs.line_ex_bus == -1)] = 0
        self.bus_or.value [(obs.line_or_bus == -1) | (obs.line_ex_bus == -1)] = 0
         
        tmp_ = 1 * obs.load_to_subid
        tmp_[obs.load_bus == 2] += obs.n_sub
        self.bus_load.value[:] = tmp_
        
        tmp_ = 1 * obs.gen_to_subid
        tmp_[obs.gen_bus == 2] += obs.n_sub
        self.bus_gen.value[:] = tmp_
        
        if self.bus_storage is not None:
            tmp_ = 1 * obs.storage_to_subid
            tmp_[obs.storage_bus == 2] += obs.n_sub
            self.bus_storage.value[:] = tmp_
        
        # Normalize voltage according to standards
        self.vm_or.value[:] = np.array([v_or / 138 if v_or < 147 else v_or / 161 if v_or < 171 else v_or / 345 for v_or in obs.v_or])
        self.vm_ex.value[:] = np.array([v_ex / 138 if v_ex < 147 else v_ex / 161 if v_ex < 171 else v_ex / 345 for v_ex in obs.v_ex])

    def _update_th_lim_param(self, obs: BaseObservation):
        threshold_ = 1.
        # take into account reactive value (and current voltage) in thermal limit
        self._th_lim_mw.value[:] =  (0.001 * obs.thermal_limit)**2 * obs.v_or **2 * 3. - obs.q_or**2
        # if (0.001 * obs.thermal_limit)**2 * obs.v_or **2 * 3. - obs.q_or**2 is too small, I put 1
        mask_ok = self._th_lim_mw.value >= threshold_
        self._th_lim_mw.value[mask_ok] = np.sqrt(self._th_lim_mw.value[mask_ok])
        self._th_lim_mw.value[~mask_ok] = threshold_ 

    def _update_storage_power_obs(self, obs: BaseObservation):
        # self._storage_power_obs.value += obs.storage_power.sum()
        self._storage_power_obs.value = 0.0

    def _update_inj_param(self, obs: BaseObservation):
        self.load_per_bus.value[:] = 0.
        self.gen_per_bus.value[:] = 0.
        load_p = 1.0 * obs.load_p
        load_p *= (obs.gen_p.sum() - self._storage_power_obs.value) / load_p.sum() 
        for bus_id in range(self.nb_max_bus):
            self.load_per_bus.value[bus_id] += load_p[self.bus_load.value == bus_id].sum()
            self.gen_per_bus.value[bus_id] += obs.gen_p[self.bus_gen.value == bus_id].sum()

    def _add_redisp_const_per_bus(self, obs: BaseObservation, bus_id: int):
        # add the constraint on the redispatching
        self.redisp_up.value[bus_id] = obs.gen_margin_up[self.bus_gen.value == bus_id].sum()
        self.redisp_down.value[bus_id] = obs.gen_margin_down[self.bus_gen.value == bus_id].sum()

    def _add_storage_const_per_bus(self, obs: BaseObservation, bus_id: int):
        if self.bus_storage is None:
            return
            
        # limit in MW
        stor_down = obs.storage_max_p_prod[self.bus_storage.value == bus_id].sum()
        # limit due to energy (if almost empty)
        stor_down = min(
            stor_down,
            obs.storage_charge[self.bus_storage.value == bus_id].sum() * (60. / obs.delta_time))
        self.storage_down.value[bus_id] = stor_down
        
        # limit in MW
        stor_up = obs.storage_max_p_absorb[self.bus_storage.value == bus_id].sum()
        # limit due to energy (if almost full)
        stor_up = min(
            stor_up,
            (obs.storage_Emax - obs.storage_charge)[self.bus_storage.value == bus_id].sum() * (60. / obs.delta_time))
        self.storage_up.value[bus_id] = stor_up

    def _remove_margin_rounding(self):
        # bring down all valid constraints by margin_rounding
        self.storage_down.value[self.storage_down.value > self.margin_rounding] -= self.margin_rounding
        self.storage_up.value[self.storage_up.value > self.margin_rounding] -= self.margin_rounding
        self.curtail_down.value[self.curtail_down.value > self.margin_rounding] -= self.margin_rounding
        self.curtail_up.value[self.curtail_up.value > self.margin_rounding] -= self.margin_rounding
        self.redisp_up.value[self.redisp_up.value > self.margin_rounding] -= self.margin_rounding
        self.redisp_down.value[self.redisp_down.value > self.margin_rounding] -= self.margin_rounding

    def _update_constraints_param_unsafe(self, obs: BaseObservation):
        tmp_ = 1.0 * obs.gen_p
        tmp_[~obs.gen_renewable] = 0.
        
        for bus_id in range(self.nb_max_bus):
            # redispatching
            self._add_redisp_const_per_bus(obs, bus_id) 
            
            # curtailment
            mask_ = (self.bus_gen.value == bus_id) & obs.gen_renewable
            self.curtail_down.value[bus_id] = 0.
            self.curtail_up.value[bus_id] = tmp_[mask_].sum()
            
            # storage
            self._add_storage_const_per_bus(obs, bus_id)
            
        self._remove_margin_rounding()

    def _update_constraints_param_safe(self, obs):
        tmp_ = 1.0 * obs.gen_p
        tmp_[~obs.gen_renewable] = 0.

        for bus_id in range(self.nb_max_bus):
            # redispatching
            self._add_redisp_const_per_bus(obs, bus_id) 

            # curtailment
            mask_ = (self.bus_gen.value == bus_id) & obs.gen_renewable
            self.curtail_down.value[bus_id] = obs.gen_p_before_curtail[mask_].sum() - tmp_[mask_].sum()
            
            # storage
            self._add_storage_const_per_bus(obs, bus_id)

            # storage target
            if self.bus_storage is not None:
                self._storage_target_bus.value[bus_id] = self._storage_setpoint[self.bus_storage.value == bus_id].sum()
            
            # past information
            if self.bus_storage is not None:
                self._past_state_of_charge.value[bus_id] = obs.storage_charge[self.bus_storage.value == bus_id].sum()
            self._past_dispatch.value[bus_id] = obs.target_dispatch[self.bus_gen.value == bus_id].sum()
        
        self.curtail_up.value[:] = 0.  # never do more curtailment in "safe" mode
        self._remove_margin_rounding()

    def _validate_param_values(self):
        self.storage_down._validate_value(self.storage_down.value)
        self.storage_up._validate_value(self.storage_up.value)
        self.curtail_down._validate_value(self.curtail_down.value)
        self.curtail_up._validate_value(self.curtail_up.value)
        self.redisp_up._validate_value(self.redisp_up.value)
        self.redisp_down._validate_value(self.redisp_down.value)
        self._th_lim_mw._validate_value(self._th_lim_mw.value)
        self._storage_target_bus._validate_value(self._storage_target_bus.value)
        self._past_dispatch._validate_value(self._past_dispatch.value)
        self._past_state_of_charge._validate_value(self._past_state_of_charge.value)

    def update_parameters(self, obs: BaseObservation, safe: bool = False):
        ## update the topology information
        self._update_topo_param(obs)
        
        ## update the thermal limit
        self._update_th_lim_param(obs)
        
        ## update the load / gen bus injected values
        self._update_inj_param(obs)

        ## update the constraints parameters
        if safe:
            self._update_constraints_param_safe(obs)
        else:
            self._update_constraints_param_unsafe(obs)
        
        # check that all parameters have correct values
        # for example non negative values for non negative parameters
        self._validate_param_values()

    def _aux_compute_kcl(self, inj_bus, f_or):  # calculate inj_bus +/- f_or on each bus
        KCL_eq = []
        for bus_id in range(self.nb_max_bus):
            tmp = inj_bus[bus_id]
            if np.any(self.bus_or.value == bus_id):
                tmp +=  cp.sum(f_or[self.bus_or.value == bus_id])
            if np.any(self.bus_ex.value == bus_id):
                tmp -=  cp.sum(f_or[self.bus_ex.value == bus_id])
            KCL_eq.append(tmp)
        return KCL_eq

    def _mask_theta_zero(self):  # find busbar that has no element connected to 
        theta_is_zero = np.full(self.nb_max_bus, True, bool)
        theta_is_zero[self.bus_or.value] = False
        theta_is_zero[self.bus_ex.value] = False
        theta_is_zero[self.bus_load.value] = False
        theta_is_zero[self.bus_gen.value] = False
        if self.bus_storage is not None:
            theta_is_zero[self.bus_storage.value] = False
        theta_is_zero[0] = True  # slack bus
        return theta_is_zero

    def _solve_problem(self, prob, solver_type=None):
        # [Optopnal] try different solvers until one finds a good solution...
        if solver_type is None:
            for solver_type in type(self).SOLVER_TYPES:
                res = self._solve_problem(prob, solver_type=solver_type)
                if res:
                    if self.verbose:
                        logger.info(f"Solver {solver_type} has converged. Stopping solver search now.")
                    return True
            return False
        
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                if solver_type is cp.OSQP:
                    tmp_ = prob.solve(  # prevent warm start (for now)
                        solver=solver_type, verbose=0, warm_start=False, max_iter=self.max_iter)
                elif solver_type is cp.SCS:
                    tmp_ = prob.solve(solver=solver_type, warm_start=False, max_iters=1000)
                else:
                    tmp_ = prob.solve(solver=solver_type, warm_start=False)
                                    
            if np.isfinite(tmp_):
                return True
            else:
                logger.warning(f"Problem diverged with dc approximation for {solver_type}, infinite value returned")
                raise cp.error.SolverError("Infinite value")

        except cp.error.SolverError as exc_:
            logger.warning(f"Problem diverged with dc approximation for {solver_type}: {exc_}")
            return False

    def run_dc(self, obs: BaseObservation):
        """This method allows to perform a dc approximation from
        the state given by the observation.
        
        To make sure that `sum P = sum C` in this system, the **loads**
        are scaled up.
        
        This function can primarily be used to retrieve the active power
        in each branch of the grid.
        Parameters
        ----------
        obs : BaseObservation
            The observation (used to get the topology and the injections)
        
        Examples
        ---------
        You can use it with:
        
        .. code-block:: python
        
            import grid2op
            from l2rpn_baselines.OptimCVXPY import OptimCVXPY
            env_name = "l2rpn_case14_sandbox"
            env = grid2op.make(env_name)
            agent = OptimCVXPY(env.action_space, env)
            obs = env.reset()
            conv = agent.run_dc(obs)
            if conv:
                print(f"flows are: {agent.flow_computed}")
            else:
                print("DC powerflow has diverged")
    
        """
        # update the parameters for the injection and topology
        self._update_topo_param(obs)
        self._update_inj_param(obs)
        
        # define the variables
        theta = cp.Variable(shape=self.nb_max_bus)
        
        # temporary variables
        f_or = cp.multiply(1. / self._powerlines_x , (theta[self.bus_or.value] - theta[self.bus_ex.value]))
        inj_bus = self.load_per_bus - self.gen_per_bus
        KCL_eq = self._aux_compute_kcl(inj_bus, f_or)  # calculate inj_bus +/- f_or on each bus
        theta_is_zero = self._mask_theta_zero()  # find busbar that has no element connected to 
        # constraints
        constraints = ([theta[theta_is_zero] == 0] + [el == 0 for el in KCL_eq])
        # no real cost here
        cost = 1.
        
        # solve
        prob = cp.Problem(cp.Minimize(cost), constraints)
        has_converged = self._solve_problem(prob, solver_type=cp.OSQP)
        
        # format the results
        if has_converged:
            self.flow_computed[:] = f_or.value
        else:
            logger.error(
                f"Problem diverged with dc approximation for all solver ({type(self).SOLVER_TYPES}). "
                "Is your grid connected (one single connex component) ?")
            self.flow_computed[:] = np.NaN
            
        return has_converged

    def reset(self, obs: BaseObservation):
        """
        This method is called at the beginning of a new episode.
        It is implemented by agents to reset their internal state if needed.
        Attributes
        -----------
        obs: :class:`grid2op.Observation.BaseObservation`
            The first observation corresponding to the initial state of the environment.
        """
        self._prev_por_error.value[:] = 0.
        conv_ = self.run_dc(obs)
        if conv_:
            self._prev_por_error.value[:] = self.flow_computed - obs.p_or
        else:
            self.logger.warning(
                "Impossible to intialize the OptimCVXPY agent because the DC powerflow did not converge.")

    def compute_optimum_unsafe(self):
        # variables
        theta = cp.Variable(shape=self.nb_max_bus)  # at each bus
        curtailment_mw = cp.Variable(shape=self.nb_max_bus)  # at each bus
        storage = cp.Variable(shape=self.nb_max_bus)  # at each bus
        redispatching = cp.Variable(shape=self.nb_max_bus)  # at each bus
        
        # usefull quantities
        f_or = cp.multiply(1. / self._powerlines_x , (theta[self.bus_or.value] - theta[self.bus_ex.value]))
        f_or_corr = f_or - self._alpha_por_error * self._prev_por_error
        inj_bus = (self.load_per_bus + storage) - (self.gen_per_bus + redispatching - curtailment_mw)
        energy_added = cp.sum(curtailment_mw) + cp.sum(storage) - cp.sum(redispatching) - self._storage_power_obs

        KCL_eq = self._aux_compute_kcl(inj_bus, f_or)  # calculate inj_bus +/- f_or on each bus
        theta_is_zero = self._mask_theta_zero()  # find busbar that has no element connected to 
        
        # constraints
        constraints = (
            # slack bus
            [theta[theta_is_zero] == 0]
            # KCL
            + [el == 0 for el in KCL_eq]
            # limit redispatching to possible values
            + [redispatching <= self.redisp_up, redispatching >= -self.redisp_down]
            # limit curtailment
            + [curtailment_mw <= self.curtail_up, curtailment_mw >= -self.curtail_down]
            # limit storage
            + [storage <= self.storage_up, storage >= -self.storage_down]
            # bus and generator variation should sum to 0. (not sure it's mandatory)
            + [energy_added == 0]
            )
        
        # objective
        # cost = cp.norm1(gp_var) + cp.norm1(lp_var)
        cost = (
            self._penalty_curtailment_unsafe * cp.sum_squares(curtailment_mw) + 
            self._penalty_storage_unsafe * cp.sum_squares(storage) +
            self._penalty_redispatching_unsafe * cp.sum_squares(redispatching) +
            cp.sum_squares(cp.pos(cp.abs(f_or_corr) - self._margin_th_limit * self._th_lim_mw))
        )
        
        # solve
        prob = cp.Problem(cp.Minimize(cost), constraints)
        has_converged = self._solve_problem(prob, solver_type=cp.OSQP)
        
        if has_converged:
            self.flow_computed[:] = f_or.value
            res = (curtailment_mw.value, storage.value, redispatching.value)
            self._storage_power_obs.value = 0.
        else:
            logger.error("compute_optimum_unsafe: Problem diverged. No continuous action will be applied.")
            self.flow_computed[:] = np.NaN
            tmp_ = np.zeros(shape=self.nb_max_bus)
            res = (1.0 * tmp_, 1.0 * tmp_, 1.0 * tmp_)
        
        return  res

    def compute_optimum_safe(self, obs: BaseObservation, l_id=None):
        if l_id is not None:
            # TODO why reconnecting it on busbar 1 ?
            self.bus_ex.value[l_id] = obs.line_ex_to_subid[l_id]
            self.bus_or.value[l_id] = obs.line_or_to_subid[l_id]
        
        # variables
        theta = cp.Variable(shape=self.nb_max_bus)  # at each bus
        curtailment_mw = cp.Variable(shape=self.nb_max_bus)  # at each bus
        storage = cp.Variable(shape=self.nb_max_bus)  # at each bus
        redispatching = cp.Variable(shape=self.nb_max_bus)  # at each bus
        
        # usefull quantities
        f_or = cp.multiply(1. / self._powerlines_x , (theta[self.bus_or.value] - theta[self.bus_ex.value]))
        f_or_corr = f_or - self._alpha_por_error * self._prev_por_error
        inj_bus = (self.load_per_bus + storage) - (self.gen_per_bus + redispatching - curtailment_mw)
        energy_added = cp.sum(curtailment_mw) + cp.sum(storage) - cp.sum(redispatching) - self._storage_power_obs
        
        KCL_eq = self._aux_compute_kcl(inj_bus, f_or)  # calculate inj_bus +/- f_or on each bus
        theta_is_zero = self._mask_theta_zero()  # find busbar that has no element connected to 
        
        dispatch_after_this = self._past_dispatch + redispatching
        state_of_charge_after = self._past_state_of_charge + storage / (60. / obs.delta_time)
        
        # constraints
        constraints = (
            # slack bus
            [theta[theta_is_zero] == 0]
            # KCL
            + [el == 0 for el in KCL_eq]
            # I impose here that the flows are bellow the limits
            + [f_or_corr <= self._margin_th_limit * self._th_lim_mw]
            + [f_or_corr >= -self._margin_th_limit * self._th_lim_mw]
            # limit redispatching to possible values
            + [redispatching <= self.redisp_up, redispatching >= -self.redisp_down]
            # limit curtailment
            + [curtailment_mw <= self.curtail_up, curtailment_mw >= -self.curtail_down]
            # limit storage
            + [storage <= self.storage_up, storage >= -self.storage_down]
            # bus and generator variation should sum to 0. (not sure it's mandatory)
            + [energy_added == 0]
            )
    
        # objective
        cost = (
            self._penalty_curtailment_safe * cp.sum_squares(curtailment_mw) +  
            self._penalty_storage_safe * cp.sum_squares(storage) +
            self._penalty_redispatching_safe * cp.sum_squares(redispatching) +
            self._weight_redisp_target * cp.sum_squares(dispatch_after_this)  +
            self._weight_storage_target * cp.sum_squares(state_of_charge_after - self._storage_target_bus) +
            # I want curtailment to be negative
            self._weight_curtail_target * cp.sum_squares(curtailment_mw + self.curtail_down)   
        )
        
        # solve the problem
        prob = cp.Problem(cp.Minimize(cost), constraints)
        has_converged = self._solve_problem(prob, solver_type=cp.OSQP)
            
        if has_converged:
            self.flow_computed[:] = f_or.value
            res = (curtailment_mw.value, storage.value, redispatching.value)
            self._storage_power_obs.value = 0.
        else:
            logger.error("compute_optimum_safe: Problem diverged. No continuous action will be applied.")
            self.flow_computed[:] = np.NaN
            tmp_ = np.zeros(shape=self.nb_max_bus)
            res = (1.0 * tmp_, 1.0 * tmp_, 1.0 * tmp_)
        
        return  res

    def _clean_vect(self, curtailment, storage, redispatching):
        """remove the value too small and set them at 0."""
        curtailment[np.abs(curtailment) < self.margin_sparse] = 0.
        storage[np.abs(storage) < self.margin_sparse] = 0.
        redispatching[np.abs(redispatching) < self.margin_sparse] = 0.

    def to_grid2op(
        self,
        obs: BaseObservation,
        curtailment: np.ndarray,
        storage: np.ndarray,
        redispatching: np.ndarray,
        base_action: BaseAction =None,
        safe=False) -> BaseAction:
        """Convert the action (given as vectors of real number output of the optimizer)
        to a valid grid2op action.
        Parameters
        ----------
        obs : BaseObservation
            The current observation, used to get some information about the grid
        curtailment : np.ndarray
            Representation of the curtailment
        storage : np.ndarray
            Action on storage units
        redispatching : np.ndarray
            Action on redispatching
        base_action : BaseAction, optional
            The previous action to modify (if any), by default None
        safe: bool, optional
            Whether this function is called from the "safe state" (in this case it allows to reset 
            all curtailment for example) or not.
            
        Returns
        -------
        BaseAction
            The action taken represented as a grid2op action
        """
        self._clean_vect(curtailment, storage, redispatching)
        
        if base_action is None:
            base_action = self.action_space()
        
        # storage
        if base_action.n_storage and np.any(np.abs(storage) > 0.):
            storage_ = np.zeros(shape=base_action.n_storage)
            storage_[:] = storage[self.bus_storage.value]
            base_action.storage_p = storage_
        
        # curtailment
        # be carefull here, the curtailment is given by the optimizer in the amount of MW you remove, 
        # grid2op expects a maximum value
        if np.any(np.abs(curtailment) > 0.):
            curtailment_mw = np.zeros(shape=base_action.n_gen) -1.
            gen_curt = obs.gen_renewable & (obs.gen_p > 0.1)
            idx_gen = self.bus_gen.value[gen_curt]  # gen_to_subid
            tmp_ = curtailment[idx_gen]
            modif_gen_optim = tmp_ != 0.
            aux_ = curtailment_mw[gen_curt]
            aux_[modif_gen_optim] = (
                obs.gen_p[gen_curt][modif_gen_optim] - 
                tmp_[modif_gen_optim] * 
                obs.gen_p[gen_curt][modif_gen_optim] /
                self.gen_per_bus.value[idx_gen][modif_gen_optim]
            )
            aux_[~modif_gen_optim] = -1.
            curtailment_mw[gen_curt] = aux_
            curtailment_mw[~gen_curt] = -1.    
                       
            if safe:
                 # id of the generators that are "curtailed" at their max value
                 # in safe mode i remove all curtailment
                gen_id_max = (curtailment_mw >= obs.gen_p_before_curtail) & obs.gen_renewable
                if np.any(gen_id_max):
                    curtailment_mw[gen_id_max] = base_action.gen_pmax[gen_id_max]
            base_action.curtail_mw = curtailment_mw
        elif safe and np.abs(self.curtail_down.value).max() == 0.:
            # if curtail_down is all 0. then it means all generators are at their max
            # output in the observation, curtailment is de facto to 1, I "just"
            # need to tell it.
            vect = 1.0 * base_action.gen_pmax
            vect[~obs.gen_renewable] = -1.
            base_action.curtail_mw = vect
            
        # redispatching
        if np.any(np.abs(redispatching) > 0.):
            redisp_ = np.zeros(obs.n_gen)
            gen_redi = obs.gen_redispatchable  #  & (obs.gen_p > self.margin_sparse)
            idx_gen = self.bus_gen.value[gen_redi]
            tmp_ = redispatching[idx_gen]
            redisp_avail = np.zeros(self.nb_max_bus)
            for bus_id in range(self.nb_max_bus):
                if redispatching[bus_id] > 0.:
                    redisp_avail[bus_id] = obs.gen_margin_up[self.bus_gen.value == bus_id].sum()
                elif redispatching[bus_id] < 0.:
                    redisp_avail[bus_id] = obs.gen_margin_down[self.bus_gen.value == bus_id].sum()
            # NB: I cannot reuse self.redisp_up above because i took some "margin" in the optimization
            # this leads obs.gen_max_ramp_up / self.redisp_up to be > 1.0 and...
            # violates the constraints of the environment...
            
            # below I compute the numerator: by what the total redispatching at each
            # node should be split between the different generators connected to it
            prop_to_gen = np.zeros(obs.n_gen)
            redisp_up = np.zeros(obs.n_gen, dtype=bool)
            redisp_up[gen_redi] = tmp_ > 0.
            prop_to_gen[redisp_up] = obs.gen_margin_up[redisp_up]
            redisp_down = np.zeros(obs.n_gen, dtype=bool)
            redisp_down[gen_redi] = tmp_ < 0.
            prop_to_gen[redisp_down] = obs.gen_margin_down[redisp_down]
            
            # avoid numeric issues
            nothing_happens = (redisp_avail[idx_gen] == 0.) & (prop_to_gen[gen_redi] == 0.)
            set_to_one_nothing = 1.0 * redisp_avail[idx_gen]
            set_to_one_nothing[nothing_happens] = 1.0
            redisp_avail[idx_gen] = set_to_one_nothing  # avoid 0. / 0. and python sends a warning
            
            if np.any(np.abs(redisp_avail[idx_gen]) <= self.margin_sparse):
                logger.warning(
                    "Some generator have a dispatch assign to them by "
                    "the optimizer, but they don't have any margin. "
                    "The dispatch has been canceled (this was probably caused "
                    "by the optimizer not meeting certain constraints).")
                this_fix_ = 1.0 * redisp_avail[idx_gen]
                too_small_here = np.abs(this_fix_) <= self.margin_sparse
                tmp_[too_small_here] = 0.
                this_fix_[too_small_here] = 1.
                redisp_avail[idx_gen] = this_fix_
                
            # Now I split the output of the optimization between the generators
            redisp_[gen_redi] = tmp_ * prop_to_gen[gen_redi] / redisp_avail[idx_gen]
            redisp_[~gen_redi] = 0.
            base_action.redispatch = redisp_

        return base_action

    def reco_line(self, obs):
        line_stat_s = obs.line_status
        cooldown = obs.time_before_cooldown_line
        maintenance = obs.time_next_maintenance
        can_be_reco = ~line_stat_s & (cooldown == 0) & (maintenance != 1)
        
        min_rho = np.inf
        action_chosen = None  
        obs_simu_chosen = None 

        if np.any(can_be_reco):
            actions = [self.action_space({"set_line_status": [(id_, +1)]}) for id_ in np.where(can_be_reco)[0]]
            for action in actions:
                # obs_simu, _reward, _done, _info = obs.simulate(action, time_step=0)
                obs_simu, _reward, _done, _info = obs.simulate(action, time_step=self.time_step)
                if (obs_simu.rho.max() < min_rho) & (len(_info['exception']) == 0):
                    # return combined_action
                    action_chosen = action
                    obs_simu_chosen = obs_simu
                    min_rho = obs_simu.rho.max()

        return action_chosen, obs_simu_chosen

    def recover_reference_topology(self, observation, base_action, min_rho=None):
        if min_rho is None:
            min_rho = observation.rho.max()

        action_chosen = None 
        obs_chosen = None  

        ref_actions = self.action_space.get_back_to_ref_state(observation).get('substation', None)
        
        if ref_actions is not None:
            for action in ref_actions:
                if observation.time_before_cooldown_sub[int(action.as_dict()['set_bus_vect']['modif_subs_id'][0])] > 0:
                    continue 
                combined_action = base_action + action
                obs_simu, _reward, _done, _info = observation.simulate(combined_action, time_step=self.time_step)
                if (obs_simu.rho.max() < min_rho) & (len(_info['exception']) == 0):
                    action_chosen = action
                    obs_chosen = obs_simu
                    min_rho = obs_simu.rho.max()
                    
        return action_chosen, obs_chosen, min_rho

    def change_substation_topology(self, observation, base_action, min_rho=None):
        
        if min_rho is None:
            min_rho = observation.rho.max()
        
        action_chosen = None
        obs_chosen = None  

        for action in self.topo_actions_to_check:
            for id_, k in enumerate(action._change_bus_vect):  # copied from BaseAction.as_dict
                if k!= 0:
                    _, _, substation_id = action._obj_caract_from_topo_id(id_)
                    break  # action affects only one substation, so break as long as one is found
            if not observation.time_before_cooldown_sub[substation_id]:
                combined_action = base_action + action
                obs_simu, _reward, _done, _info = observation.simulate(combined_action, time_step=self.time_step)
                if (obs_simu.rho.max() < min_rho) & (len(_info['exception']) == 0):
                    action_chosen = action
                    obs_chosen = obs_simu
                    min_rho = obs_simu.rho.max()
                
        return action_chosen, obs_chosen, min_rho

    def get_topology_action(self, observation, base_action, min_rho=None):
        if min_rho is None:
            min_rho = observation.rho.max()
        
        action_chosen, obs_chosen, rho_chosen = self.recover_reference_topology(observation, base_action, min_rho)
        if action_chosen is not None:
            if obs_chosen.rho.max() < 0.99:
                return action_chosen, obs_chosen
        
        action_chosen, obs_chosen, rho_chosen = self.change_substation_topology(observation, base_action, rho_chosen)
        if action_chosen is not None: 
        # if action_chosen is not None and (min_rho - rho_chosen > 0.1): 
            return action_chosen, obs_chosen

        return None, None
    
    def act(
        self,
        observation: BaseObservation,
        reward=None,
        done: bool=False):

        if done:
            return self.do_nothing

        if observation.current_step == 0:
            self.flow_computed[:] = np.NaN
            self._prev_por_error.value[:] = 0.

        prev_ok = np.isfinite(self.flow_computed)
        # only keep the negative error (meaning I underestimated the flow)
        self._prev_por_error.value[prev_ok] = np.minimum(self.flow_computed[prev_ok] - observation.p_or[prev_ok], 0.)
        self._prev_por_error.value[~prev_ok] = 0.

        act = self.action_space()
        _obs = None
        _obs_simu, _reward, _done, _info = observation.simulate(act, time_step=1)
        if len(_info['exception']) == 0:
            _obs = _obs_simu

        reco_act, reco_obs = self.reco_line(observation)
        if reco_act is not None:
            act = act + reco_act
            _obs = reco_obs 
        
        self.flow_computed[:] = np.NaN
        if observation.rho.max() > self.rho_danger:
            # I attempt to make the grid more secure
            
            topo_act, topo_obs = self.get_topology_action(observation, act)

            if topo_act is not None:
                act = act + topo_act
                _obs = topo_obs
            
            self._update_storage_power_obs(observation)
            if _obs is not None:
                self.update_parameters(_obs, safe=False)
            else:
                self.update_parameters(observation, safe=False)
            # solve the problem
            curtailment, storage, redispatching = self.compute_optimum_unsafe()
            # get back the grid2op representation
            act = self.to_grid2op(observation, curtailment, storage, redispatching, base_action=act, safe=False)
        
        return act


def test_OptimCVXPY():
    import random
    import grid2op
    from lightsim2grid import LightSimBackend
    bk_cls = LightSimBackend

    env = grid2op.make('l2rpn_wcci_2022', backend=bk_cls())

    line_actions = env.action_space.get_all_unitary_line_change(env.action_space)
    curtail_actions = env.action_space.get_all_unitary_curtail(env.action_space)  # NOTE by default min_value=0.5
    redispatch_actions = env.action_space.get_all_unitary_redispatch(env.action_space)
    storage_actions = env.action_space.get_all_unitary_storage(env.action_space)

    # assume directory of this code file has a file named actions_space.npz
    action_space_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'actions_space.npz')
    config = OPTIMCVXPY_CONFIG.copy()
    agent = OptimCVXPY(
        env, 
        env.action_space, 
        action_space_path=action_space_path,
        config=config,
        verbose=True
    )

    trial = 0
    done = True
    while trial < 100:
        if done:
            print(f'trial {trial}.')
            obs = env.reset()
            done = False
            agent.reset(obs)

        action = random.choice(line_actions) + \
            random.choice(curtail_actions) + random.choice(redispatch_actions) + random.choice(storage_actions)
        obs, _, done, _ = env.step(action)

        if not done:
            action = agent.act(obs)
            obs, _, done, _ = env.step(action)

        trial += 1


if __name__ == "__main__":
    test_OptimCVXPY()

    print('Test over.')