# Copyright (c) 2022 - Present Myalo UG (haftungbeschränkt) (Mark Aron Szulyovszky, Daniel Szemerey) <info@dreamfaster.ai>. All rights reserved. See LICENSE in root folder.


from typing import Callable, List, Optional

import pandas as pd
import ray
from tqdm import tqdm

from ...base import Artifact, Composite, Pipeline, TrainedPipeline, X
from ...splitters import Fold
from ..types import Backend, Stage


def train_pipeline(
    self,
    func: Callable,
    pipeline: Pipeline,
    X: X,
    y: pd.Series,
    artifact: Artifact,
    splits: List[Fold],
    never_update: bool,
    backend: Backend,
    project_name: str,
    silent: bool,
):
    func = ray.remote(func).options(num_cpus=self.limit_threads)
    X = ray.put(X)
    y = ray.put(y)
    futures = [
        func.remote(
            X,
            y,
            artifact,
            pipeline,
            split,
            never_update,
            backend,
            project_name,
        )
        for split in splits
    ]
    return_value = ray.get(futures)
    ray.internal.free([X, y])
    return return_value


def backtest_pipeline(
    self,
    func: Callable,
    pipeline: TrainedPipeline,
    splits: List[Fold],
    X: pd.DataFrame,
    y: pd.Series,
    artifact: Artifact,
    backend: Backend,
    mutate: bool,
    silent: bool,
):
    func = ray.remote(func).options(num_cpus=self.limit_threads)
    X = ray.put(X)
    y = ray.put(y)
    pipeline = ray.put(pipeline)

    # result_refs = []
    # for split in splits:
    #     if len(result_refs) > 1:  # self.limit_threads:
    #         ready_refs, result_refs = ray.wait(result_refs, num_returns=1)
    #         ray.get(ready_refs)

    #     result_refs.append(
    #         func.remote(pipeline, split, X, y, artifact, backend, mutate),
    #     )

    # return_value_1 = sorted(ray.get(result_refs), key=lambda x: x[0].index[0])

    futures = [
        func.remote(pipeline, split, X, y, artifact, backend, mutate)
        for split in splits
    ]
    return_value = ray.get(futures)

    ray.internal.free([X, y, pipeline])
    return return_value


def process_child_transformations(
    self,
    func: Callable,
    list_of_child_transformations_with_index: List,
    composite: Composite,
    X: X,
    y: Optional[pd.Series],
    artifacts: Artifact,
    stage: Stage,
    backend: Backend,
    results_primary: Optional[List[pd.DataFrame]],
    tqdm: Optional[tqdm] = None,
):
    # list_of_child_transformations_with_index = list(
    #     list_of_child_transformations_with_index
    # )
    # available_resources = ray.available_resources()
    # if (
    #     len(list_of_child_transformations_with_index) == 1
    #     or "CPU" not in available_resources
    #     or ("CPU" in available_resources and available_resources["CPU"] < 2)
    # ):
    return [
        func(
            composite,
            index,
            child_transformation,
            X,
            y,
            artifacts,
            stage,
            backend,
            results_primary,
            tqdm,
        )
        for index, child_transformation in list_of_child_transformations_with_index
    ]
    # else:
    #     func = ray.remote(func)
    #     X = ray.put(X)
    #     y = ray.put(y)

    #     futures = [
    #         func.remote(
    #             composite,
    #             index,
    #             child_transformation,
    #             X,
    #             y,
    #             artifacts,
    #             stage,
    #             backend,
    #             results_primary,
    #         )
    #         for index, child_transformation in list_of_child_transformations_with_index
    #     ]
    # batch_size = 2

    # chunk processing of futures with ray
    # results = []
    # for i in range(0, len(futures), batch_size):
    #     results.extend(ray.get(futures[i : i + batch_size]))
    # while futures:
    #     if len(futures) < batch_size:
    #         batch_size = len(futures)
    #     ready_futures, futures = ray.wait(futures, num_returns=batch_size)
    #     result = ray.get(ready_futures)
    #     results.append(result)

    # return results


class RayBackend(Backend):
    name = "ray"
    process_child_transformations = process_child_transformations
    train_pipeline = train_pipeline
    backtest_pipeline = backtest_pipeline

    def __init__(self, limit_threads: Optional[int] = None):
        self.limit_threads = limit_threads
