import numpy as np
from tqdm import tqdm

from collections import defaultdict
from mushroom_rl.utils.record import VideoRecorder


class Core(object):
    """
    Implements the functions to run a generic algorithm.

    """
    def __init__(self, agent, mdp, callbacks_fit=None, callback_step=None, record_dictionary=None):
        """
        Constructor.

        Args:
            agent (Agent): the agent moving according to a policy;
            mdp (Environment): the environment in which the agent moves;
            callbacks_fit (list): list of callbacks to execute at the end of each fit;
            callback_step (Callback): callback to execute after each step;

        """
        self.agent = agent
        self.mdp = mdp
        self.callbacks_fit = callbacks_fit if callbacks_fit is not None else list()
        self.callback_step = callback_step if callback_step is not None else lambda x: None

        self._state = np.zeros((self.mdp.nr_envs,) + self.mdp.info.observation_space.shape)

        self._total_episodes_counter = 0
        self._total_steps_counter = 0
        self._current_episodes_counter = 0
        self._current_steps_counter = 0
        self._episode_steps = None
        self._n_episodes = None
        self._n_steps_per_fit = None
        self._n_episodes_per_fit = None

        if record_dictionary is None:
            record_dictionary = dict()
        self._record = self._build_recorder_class(**record_dictionary)

    def learn(self, n_steps=None, n_episodes=None, n_steps_per_fit=None,
              n_episodes_per_fit=None, render=False, quiet=False, record=False):
        """
        This function moves the agent in the environment and fits the policy using the collected samples.
        The agent can be moved for a given number of steps or a given number of episodes and, independently from this
        choice, the policy can be fitted after a given number of steps or a given number of episodes.
        The environment is reset at the beginning of the learning process.

        Args:
            n_steps (int, None): number of steps to move the agent;
            n_episodes (int, None): number of episodes to move the agent;
            n_steps_per_fit (int, None): number of steps between each fit of the
                policy;
            n_episodes_per_fit (int, None): number of episodes between each fit
                of the policy;
            render (bool, False): whether to render the environment or not;
            quiet (bool, False): whether to show the progress bar or not;
            record (bool, False): whether to record a video of the environment or not. If True, also the render flag
                should be set to True.

        """
        assert (n_episodes_per_fit is not None and n_steps_per_fit is None)\
            or (n_episodes_per_fit is None and n_steps_per_fit is not None)

        assert (render and record) or (not record), "To record, the render flag must be set to true"

        self._n_steps_per_fit = n_steps_per_fit
        self._n_episodes_per_fit = n_episodes_per_fit

        if n_steps_per_fit is not None:
            fit_condition = lambda: self._current_steps_counter >= self._n_steps_per_fit
        else:
            fit_condition = lambda: self._current_episodes_counter  >= self._n_episodes_per_fit

        self._run(n_steps, n_episodes, fit_condition, render, quiet, record, get_env_info=False)

    def evaluate(self, initial_states=None, n_steps=None, n_episodes=None,
                 render=False, quiet=False, record=False, get_env_info=False):
        """
        This function moves the agent in the environment using its policy.
        The agent is moved for a provided number of steps, episodes, or from a set of initial states for the whole
        episode. The environment is reset at the beginning of the learning process.

        Args:
            initial_states (np.ndarray, None): the starting states of each episode;
            n_steps (int, None): number of steps to move the agent;
            n_episodes (int, None): number of episodes to move the agent;
            render (bool, False): whether to render the environment or not;
            quiet (bool, False): whether to show the progress bar or not;
            record (bool, False): whether to record a video of the environment or not. If True, also the render flag
                should be set to True;
            get_env_info (bool, False): whether to return the environment info list or not.

        Returns:
            The collected dataset and, optionally, an extra dataset of
            environment info, collected at each step.

        """
        assert (render and record) or (not record), "To record, the render flag must be set to true"

        fit_condition = lambda: False

        return self._run(n_steps, n_episodes, fit_condition, render, quiet, record, get_env_info, initial_states)

    def _run(self, n_steps, n_episodes, fit_condition, render, quiet, record, get_env_info, initial_states=None):
        assert n_episodes is not None and n_steps is None and initial_states is None\
            or n_episodes is None and n_steps is not None and initial_states is None\
            or n_episodes is None and n_steps is None and initial_states is not None

        self._n_episodes = len( initial_states) if initial_states is not None else n_episodes

        if n_steps is not None:
            move_condition = lambda: self._total_steps_counter < n_steps

            steps_progress_bar = tqdm(total=n_steps,  dynamic_ncols=True, disable=quiet, leave=False)
            episodes_progress_bar = tqdm(disable=True)
        else:
            move_condition = lambda: self._total_episodes_counter < self._n_episodes

            steps_progress_bar = tqdm(disable=True)
            episodes_progress_bar = tqdm(total=self._n_episodes, dynamic_ncols=True, disable=quiet, leave=False)

        dataset, dataset_info = self._run_impl(move_condition, fit_condition, steps_progress_bar, episodes_progress_bar,
                                               render, record, initial_states)

        if get_env_info:
            return dataset, dataset_info
        else:
            return dataset

    def _run_impl(self, move_condition, fit_condition, steps_progress_bar, episodes_progress_bar, render, record,
                  initial_states):
        self._total_episodes_counter = 0
        self._total_steps_counter = 0
        self._current_episodes_counter = 0
        self._current_steps_counter = 0
        self._episode_steps = np.zeros(self.mdp.nr_envs, dtype=int)

        datasets = [[] for _ in range(self.mdp.nr_envs)]
        datasets_info = [defaultdict(list) for _ in range(self.mdp.nr_envs)]

        lasts = np.ones(self.mdp.nr_envs, dtype=bool)
        
        self.agent.setup_rollout()
        while move_condition():
            reset_indices = []
            for i in range(self.mdp.nr_envs):
                if lasts[i]:
                    reset_indices.append(i)
            if len(reset_indices) > 0:
                self.reset(reset_indices, initial_states)

            samples, step_infos = self._step(render, record)

            self.callback_step([samples])

            self._total_steps_counter += self.mdp.nr_envs
            self._current_steps_counter += self.mdp.nr_envs
            steps_progress_bar.update(self.mdp.nr_envs)

            for i in range(self.mdp.nr_envs):
                sample = samples[i]
                step_info = step_infos[i]

                if sample[-1]:
                    self._total_episodes_counter += 1
                    self._current_episodes_counter += 1
                    episodes_progress_bar.update(1)

                datasets[i].append(sample)

                lasts[i] = samples[i][-1]

                for key, value in step_info.items():
                    datasets_info[i][key].append(value)

            if fit_condition():
                self.agent.fit(datasets, datasets_info)
                self._current_episodes_counter = 0
                self._current_steps_counter = 0

                for dataset in datasets:
                    for c in self.callbacks_fit:
                        c(dataset)

                datasets = [[] for _ in range(self.mdp.nr_envs)]
                datasets_info = [defaultdict(list) for _ in range(self.mdp.nr_envs)]
                self.agent.setup_rollout()

        self.agent.stop()
        self.mdp.stop()

        if record:
            self._record.stop()

        steps_progress_bar.close()
        episodes_progress_bar.close()

        return datasets, datasets_info

    def _step(self, render, record):
        """
        Single step.

        Args:
            render (bool): whether to render or not.

        Returns:
            A tuple containing the previous state, the action sampled by the agent, the reward obtained, the reached
            state, the absorbing flag of the reached state and the last step flag.

        """
        action = self.agent.draw_action(self._state)
        next_state, reward, absorbing, step_info = self.mdp.step(action)

        self._episode_steps += 1

        if render:
            frame = self.mdp.render(record)

            if record:
                self._record(frame)

        last = np.logical_or(self._episode_steps >= self.mdp.info.horizon, absorbing)

        state = self._state
        next_state = self._preprocess(next_state.copy())
        self._state = next_state

        samples = []
        step_infos = []
        for i in range(self.mdp.nr_envs):
            samples.append((state[i], action[i], reward[i], next_state[i], absorbing[i], last[i]))
            step_infos.append(step_info[i])

        return samples, step_infos

    def reset(self, indices, initial_states=None):
        """
        Reset the state of the agent.

        """
        if initial_states is None or self._total_episodes_counter == self._n_episodes:
            initial_state = None
        else:
            initial_state = initial_states[self._total_episodes_counter]

        self.agent.episode_start()
        
        self._state[indices] = self._preprocess(self.mdp.reset(initial_state, indices).copy())
        self.agent.next_action = None
        self._episode_steps[indices] = 0

    def _preprocess(self, state):
        """
        Method to apply state preprocessors.

        Args:
            state (np.ndarray): the state to be preprocessed.

        Returns:
             The preprocessed state.

        """
        for p in self.agent.preprocessors:
            state = p(state)

        return state

    def _build_recorder_class(self, recorder_class=None, fps=None, **kwargs):
        """
        Method to create a video recorder class.

        Args:
            recorder_class (class): the class used to record the video. By default, we use the ``VideoRecorder`` class
                from mushroom. The class must implement the ``__call__`` and ``stop`` methods.

        Returns:
             The recorder object.

        """

        if not recorder_class:
            recorder_class = VideoRecorder

        if not fps:
            fps = int(1 / self.mdp.info.dt)

        return recorder_class(fps=fps, **kwargs)
