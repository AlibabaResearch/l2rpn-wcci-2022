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
Classes and functions related to creating an agent for submission.

create a symbolic link
ln -s /path/to/file /path/to/symlink
update a symolic link
ln -sf /path/to/file /path/to/symlink

Submission workflow:
1. create a symlink to the code file (.py)
2. create/update a symlink to the action_space_path (.npz)
3. zip the current folder. However, do not do this yourself, run unit-test using L2RPN_wcci2022_starting_kit/run_test_submission.py

"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .optimCVXPY import OPTIMCVXPY_CONFIG, OptimCVXPY


def make_agent(env, this_directory_path):

    config = OPTIMCVXPY_CONFIG.copy()
    action_space_path = os.path.join(this_directory_path, 'actions_space.npz')
    time_step = 1

    # just for record
    if os.path.islink(action_space_path):
        print(f'The current action_space_path {action_space_path} links to {os.readlink(action_space_path)}')

    agent = OptimCVXPY(
        env, 
        env.action_space, 
        action_space_path=action_space_path, 
        config=config, 
        time_step=time_step, 
        verbose=1
        )

    return agent