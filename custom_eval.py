from collections import defaultdict
import concurrent
import copy
from datetime import datetime
import functools
import gym
import logging
import math
import numpy as np
import os
import pickle
import tempfile
import time
from typing import (
    Callable,
    Container,
    DefaultDict,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
)

import ray
from ray.actor import ActorHandle
from ray.exceptions import RayError
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env.env_context import EnvContext
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.env.utils import gym_env_creator
from ray.rllib.evaluation.collectors.simple_list_collector import SimpleListCollector
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.metrics import (
    collect_episodes,
    collect_metrics,
    summarize_episodes,
)

from ray.rllib.utils.deprecation import (
    Deprecated,
    deprecation_warning,
    DEPRECATED_VALUE,
)
from ray.rllib.utils.framework import try_import_tf, try_import_torch


tf1, tf, tfv = try_import_tf()

logger = logging.getLogger(__name__)


class TrainerCustomEval:
    def evaluate(
        self,
        episodes_left_fn=None,  # deprecated
        duration_fn: Optional[Callable[[int], int]] = None,
    ) -> dict:
        """Evaluates current policy under `evaluation_config` settings.

        Note that this default implementation does not do anything beyond
        merging evaluation_config with the normal trainer config.

        Args:
            duration_fn: An optional callable taking the already run
                num episodes as only arg and returning the number of
                episodes left to run. It's used to find out whether
                evaluation should continue.
        """
        if episodes_left_fn is not None:
            deprecation_warning(
                old="Trainer.evaluate(episodes_left_fn)",
                new="Trainer.evaluate(duration_fn)",
                error=False,
            )
            duration_fn = episodes_left_fn

        # In case we are evaluating (in a thread) parallel to training,
        # we may have to re-enable eager mode here (gets disabled in the
        # thread).
        if (
            self.config.get("framework") in ["tf2", "tfe"]
            and not tf.executing_eagerly()
        ):
            tf1.enable_eager_execution()

        # Call the `_before_evaluate` hook.
        self._before_evaluate()

        # Sync weights to the evaluation WorkerSet.
        if self.evaluation_workers is not None:
            self.evaluation_workers.sync_weights(
                from_worker=self.workers.local_worker()
            )
            self._sync_filters_if_needed(self.evaluation_workers)

        if self.config["custom_eval_function"]:
            logger.info(
                "Running custom eval function {}".format(
                    self.config["custom_eval_function"]
                )
            )
            metrics = self.config["custom_eval_function"](
                self, self.evaluation_workers)
            if not metrics or not isinstance(metrics, dict):
                raise ValueError(
                    "Custom eval function must return "
                    "dict of metrics, got {}.".format(metrics)
                )
        else:
            if (
                self.evaluation_workers is None
                and self.workers.local_worker().input_reader is None
            ):
                raise ValueError(
                    "Cannot evaluate w/o an evaluation worker set in "
                    "the Trainer or w/o an env on the local worker!\n"
                    "Try one of the following:\n1) Set "
                    "`evaluation_interval` >= 0 to force creating a "
                    "separate evaluation worker set.\n2) Set "
                    "`create_env_on_driver=True` to force the local "
                    "(non-eval) worker to have an environment to "
                    "evaluate on."
                )

            # How many episodes/timesteps do we need to run?
            # In "auto" mode (only for parallel eval + training): Run as long
            # as training lasts.
            unit = self.config["evaluation_duration_unit"]
            eval_cfg = self.config["evaluation_config"]
            rollout = eval_cfg["rollout_fragment_length"]
            num_envs = eval_cfg["num_envs_per_worker"]
            duration = (
                self.config["evaluation_duration"]
                if self.config["evaluation_duration"] != "auto"
                else (self.config["evaluation_num_workers"] or 1)
                * (1 if unit == "episodes" else rollout)
            )
            num_ts_run = 0

            # Default done-function returns True, whenever num episodes
            # have been completed.
            if duration_fn is None:

                def duration_fn(num_units_done):
                    return duration - num_units_done

            logger.info(f"Evaluating current policy for {duration} {unit}.")

            metrics = None
            # No evaluation worker set ->
            # Do evaluation using the local worker. Expect error due to the
            # local worker not having an env.
            if self.evaluation_workers is None:
                # If unit=episodes -> Run n times `sample()` (each sample
                # produces exactly 1 episode).
                # If unit=ts -> Run 1 `sample()` b/c the
                # `rollout_fragment_length` is exactly the desired ts.
                iters = duration if unit == "episodes" else 1
                for _ in range(iters):
                    num_ts_run += len(self.workers.local_worker().sample())
                metrics = collect_metrics(
                    self.workers.local_worker(),
                    keep_custom_metrics=self.config["keep_per_episode_custom_metrics"],
                )

            # Evaluation worker set only has local worker.
            elif self.config["evaluation_num_workers"] == 0:
                # If unit=episodes -> Run n times `sample()` (each sample
                # produces exactly 1 episode).
                # If unit=ts -> Run 1 `sample()` b/c the
                # `rollout_fragment_length` is exactly the desired ts.
                iters = duration if unit == "episodes" else 1
                for _ in range(iters):
                    num_ts_run += len(self.evaluation_workers.local_worker().sample())

            # Evaluation worker set has n remote workers.
            else:
                # How many episodes have we run (across all eval workers)?
                num_units_done = 0
                round_ = 0
                while True:
                    units_left_to_do = duration_fn(num_units_done)
                    if units_left_to_do <= 0:
                        break

                    round_ += 1
                    batches = ray.get(
                        [
                            w.sample.remote()
                            for i, w in enumerate(
                                self.evaluation_workers.remote_workers()
                            )
                            if i * (1 if unit == "episodes" else rollout * num_envs)
                            < units_left_to_do
                        ]
                    )
                    # 1 episode per returned batch.
                    if unit == "episodes":
                        num_units_done += len(batches)
                    # n timesteps per returned batch.
                    else:
                        ts = sum(len(b) for b in batches)
                        num_ts_run += ts
                        num_units_done += ts

                    logger.info(
                        f"Ran round {round_} of parallel evaluation "
                        f"({num_units_done}/{duration} {unit} done)"
                    )

            if metrics is None:
                metrics = collect_metrics(
                    self.evaluation_workers.local_worker(),
                    self.evaluation_workers.remote_workers(),
                    keep_custom_metrics=self.config["keep_per_episode_custom_metrics"],
                )
            metrics["timesteps_this_iter"] = num_ts_run

        # Evaluation does not run for every step.
        # Save evaluation metrics on trainer, so it can be attached to
        # subsequent step results as latest evaluation result.
        self.evaluation_metrics = {"evaluation": metrics}

        # Also return the results here for convenience.
        return self.evaluation_metrics

# need modify RolloutWorker.sample()
from ray.util.debug import log_once
from ray.rllib.utils.debug import summarize
def sample(self):
    """Returns a batch of experience sampled from this worker.

    This method must be implemented by subclasses.

    Returns:
        A columnar batch of experiences (e.g., tensors).

    Examples:
        >>> print(worker.sample())
        SampleBatch({"obs": [1, 2, 3], "action": [0, 1, 0], ...})
    """

    if self.fake_sampler and self.last_batch is not None:
        return
    elif self.input_reader is None:
        raise ValueError(
            "RolloutWorker has no `input_reader` object! "
            "Cannot call `sample()`. You can try setting "
            "`create_env_on_driver` to True."
        )

    if log_once("sample_start"):
        logger.info(
            "Generating sample batch of size {}".format(
                self.rollout_fragment_length
            )
        )

    batches = [self.input_reader.next()]
    steps_so_far = (
        batches[0].count
        if self.count_steps_by == "env_steps"
        else batches[0].agent_steps()
    )

    # In truncate_episodes mode, never pull more than 1 batch per env.
    # This avoids over-running the target batch size.
    if self.batch_mode == "truncate_episodes":
        max_batches = self.num_envs
    else:
        max_batches = float("inf")

    while (
        steps_so_far < self.rollout_fragment_length and len(
            batches) < max_batches
    ):
        batch = self.input_reader.next()
        steps_so_far += (
            batch.count
            if self.count_steps_by == "env_steps"
            else batch.agent_steps()
        )
        batches.append(batch)
    batch = batches[0].concat_samples(
        batches) if len(batches) > 1 else batches[0]

    self.callbacks.on_sample_end(worker=self, samples=batch)

    # Always do writes prior to compression for consistency and to allow
    # for better compression inside the writer.
    self.output_writer.write(batch)

    # Do off-policy estimation, if needed.
    if self.reward_estimators:
        for sub_batch in batch.split_by_episode():
            for estimator in self.reward_estimators:
                estimator.process(sub_batch)

    if log_once("sample_end"):
        logger.info("Completed sample batch:\n\n{}\n".format(summarize(batch)))

    if self.compress_observations:
        batch.compress(bulk=self.compress_observations == "bulk")

    if self.fake_sampler:
        self.last_batch = batch
