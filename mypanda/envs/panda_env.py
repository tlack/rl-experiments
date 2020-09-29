import gym
from gym import error, spaces, utils
from gym.utils import seeding

import os
import pybullet as p
import pybullet_data
import math
import numpy as np
import random
import time

# https://stackoverflow.com/a/39662359
def is_notebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


MAX_EPISODE_LEN = 100


class PandaEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self):
        self.step_counter = 0
        self.episode_counter = 0
        self.episode_reward = 0
        self.action_muting = 0.2
        self.steps_per_episode = MAX_EPISODE_LEN
        self.goal = "bumps"
        p.connect(p.GUI if is_notebook() else p.DIRECT)
        p.resetDebugVisualizerCamera(
            cameraDistance=1.5,
            cameraYaw=0,
            cameraPitch=-40,
            cameraTargetPosition=[0.55, -0.35, 0.2],
        )
        self.action_space = spaces.Box(np.array([-1] * 4), np.array([1] * 4))
        self.observation_space = spaces.Box(np.array([-1] * 8), np.array([1] * 8))
        self.n_goals = 0

    def step(self, action):
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

        if os.path.exists("/tmp/render.txt"):
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        else:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)

        orientation = p.getQuaternionFromEuler([0.0, -math.pi, math.pi / 2.0])
        dv = self.action_muting
        dx = action[0] * dv
        dy = action[1] * dv
        dz = action[2] * dv
        fingers = action[3]

        currentPose = p.getLinkState(self.pandaUid, 11)
        currentPosition = currentPose[0]
        newPosition = [
            currentPosition[0] + dx,
            currentPosition[1] + dy,
            currentPosition[2] + dz,
        ]
        jointPoses = p.calculateInverseKinematics(
            self.pandaUid, 11, newPosition, orientation
        )[0:7]

        p.setJointMotorControlArray(
            self.pandaUid,
            list(range(7)) + [9, 10],
            p.POSITION_CONTROL,
            list(jointPoses) + 2 * [fingers],
        )

        p.stepSimulation()

        state_object, _ = p.getBasePositionAndOrientation(self.objectUid)
        state_robot = p.getLinkState(self.pandaUid, 11)[0]
        state_fingers = (
            p.getJointState(self.pandaUid, 9)[0],
            p.getJointState(self.pandaUid, 10)[0],
        )

        def cosine_similarity(vector, matrix):
            return (
                np.sum(vector * matrix, axis=1)
                / (np.sqrt(np.sum(matrix ** 2, axis=1)) * np.sqrt(np.sum(vector ** 2)))
            )[::-1]

        def dist(a, b):
            diff = abs(np.sum(np.array(a) - np.array(b)))
            return diff

        obj_dist = dist(self.object_loc, state_object)
        # print('od', obj_dist)

        if self.goal == "bumps":
            if obj_dist > 0.1:
                print("BUMPED!")
                reward = 100
                done = True
                self.n_goals += 1
            else:
                # print(np.array(state_object) - np.array(newPosition))
                diff = dist(state_object, newPosition) * 3
                reward = 1 - diff
                # print('reward2', cosine_similarity(np.array([state_object]), np.array([state_robot])))
                done = False
        elif self.goal == "touches":
            diff = dist(state_object, newPosition) * 10
            if diff < 0:
                print("BUMPED!")
                self.n_goals += 1
                reward = 10
                done = False
            else:
                reward = 1 - diff

        self.episode_reward += reward
        self.step_counter += 1

        if self.step_counter > self.steps_per_episode:
            if self.episode_counter % 10 == 0:

                def f(n):
                    return ",".join([f"{x:02f}" for x in n])

                print(
                    f"reward (this, avg/step): {f([reward, self.episode_reward / (self.step_counter+1)])}"
                )
                print(
                    f"goals: {n_goals} / {f([self.n_goals / (self.episode_counter+1)])}"
                )
                time.sleep(1)
                reward = 0
                done = True

        info = {"object_position": state_object}
        self.observation = state_robot + state_fingers + state_object
        # print(self.observation)

        if done:
            self.episode_counter += 1

        return np.array(self.observation).astype(np.float32), reward, done, info

    def reset(self):
        self.episode_reward = 0
        self.step_counter = 0
        p.resetSimulation()
        p.configureDebugVisualizer(
            p.COV_ENABLE_RENDERING, 0
        )  # we will enable rendering after we loaded everything
        urdfRootPath = pybullet_data.getDataPath()
        p.setGravity(0, 0, -40)

        planeUid = p.loadURDF(
            os.path.join(urdfRootPath, "plane.urdf"), basePosition=[0, 0, -0.65]
        )

        rest_poses = [0, -0.215, 0, -2.57, 0, 2.356, 2.356, 0.08, 0.08]
        self.pandaUid = p.loadURDF(
            os.path.join(urdfRootPath, "franka_panda/panda.urdf"), useFixedBase=True
        )
        for i in range(7):
            p.resetJointState(self.pandaUid, i, rest_poses[i])
        p.resetJointState(self.pandaUid, 9, 0.08)
        p.resetJointState(self.pandaUid, 10, 0.08)
        tableUid = p.loadURDF(
            os.path.join(urdfRootPath, "table/table.urdf"), basePosition=[0.5, 0, -0.65]
        )

        trayUid = p.loadURDF(
            os.path.join(urdfRootPath, "tray/traybox.urdf"), basePosition=[0.65, 0, 0]
        )

        # state_object= [random.uniform(0.5,0.8),random.uniform(-0.2,0.2),0.05]
        state_object = [
            0.75 + random.uniform(-0.05, 0.05),
            0 + random.uniform(-0.05, 0.05),
            0.1,
        ]
        self.object_loc = state_object
        self.objectUid = p.loadURDF(
            os.path.join(urdfRootPath, "random_urdfs/000/000.urdf"),
            basePosition=state_object,
        )
        state_robot = p.getLinkState(self.pandaUid, 11)[0]
        state_fingers = (
            p.getJointState(self.pandaUid, 9)[0],
            p.getJointState(self.pandaUid, 10)[0],
        )
        self.observation = state_robot + state_fingers + tuple(state_object)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        return np.array(self.observation).astype(np.float32)

    def render(self, mode="human"):
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0.55, 0.35, 0.2],
            distance=1.2,
            yaw=0,
            pitch=-70,
            roll=0,
            upAxisIndex=2,
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60, aspect=float(960) / 720, nearVal=0.1, farVal=100.0
        )
        (_, _, px, _, _) = p.getCameraImage(
            width=960,
            height=720,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL,
        )

        rgb_array = np.array(px, dtype=np.uint8)
        rgb_array = np.reshape(rgb_array, (720, 960, 4))

        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _get_state(self):
        return self.observation

    def close(self):
        p.disconnect()
