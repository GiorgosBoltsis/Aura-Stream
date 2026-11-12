import numpy as np
import gymnasium as gym
from gymnasium import spaces

from .advisors import MODES, to_env_action
from .env import JointEnv


class JointGymEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(self, windows_by_dataset, context=True, migration_delay=1.0, seed=0,
                 mode="factored"):
        super().__init__()
        self.pools = [w for w in windows_by_dataset.values() if w]
        self.context = context
        self.migration_delay = migration_delay
        self.mode = mode
        self.planner, self.planner_dim = (MODES[mode][0], MODES[mode][1]) if mode in MODES else (None, 0)
        self._rng = np.random.default_rng(seed)
        probe = JointEnv(self.pools[0][0], context=context, planner=self.planner,
                         planner_dim=self.planner_dim)
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(probe.obs_dim,), dtype=np.float32)
        if mode in MODES:
            self.action_space = spaces.MultiDiscrete(MODES[mode][2])
        elif mode == "factored":
            self.action_space = spaces.MultiDiscrete([probe.n_bitrates, probe.n_edges])
        else:
            self.action_space = spaces.Discrete(probe.n_actions)
        self._env = None

    def make_env(self, window, seed, start):
        return JointEnv(window, context=self.context, migration_delay=self.migration_delay,
                        seed=seed, start=start, planner=self.planner, planner_dim=self.planner_dim)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        pool = self.pools[self._rng.integers(len(self.pools))]
        window = pool[self._rng.integers(len(pool))]
        self._env = self.make_env(window, int(self._rng.integers(2**31)),
                                  float(self._rng.uniform(0, window.duration)))
        return self._env.reset(), {}

    def step(self, action):
        obs, reward, done, info = self._env.step(to_env_action(self._env, action, self.mode))
        return obs, float(reward), bool(done), False, info
