import tqdm
import typing as t

import matplotlib.pyplot as plt
import tensorflow as tf
import vit_jax.models

from vit_jax.configs import common as common_config
from vit_jax.configs import models as models_config

import ml_collections
import numpy as np
from vit_jax import models


def get_accuracy(params: dict, ds_test: tf.data.Dataset, apply_function: t.Callable, batch_size: int) -> float:
    """Returns accuracy evaluated on the test set."""
    good = total = 0
    steps = len(ds_test) // batch_size
    for _, batch in zip(tqdm.trange(steps), ds_test.as_numpy_iterator()):
        predicted = apply_function(params, batch["image"][0, :])
        is_same = predicted.argmax(axis=-1) == batch["label"].argmax(axis=-1)
        good += is_same.sum()
        total += len(is_same.flatten())
    return good / total # TODO Return good and total separately


def preprocess_image(image: np.ndarray) -> np.ndarray:
    min_vals = tf.math.reduce_min(image, axis=[1, 2, 3], keepdims=True)
    max_vals = tf.math.reduce_max(image, axis=[1, 2, 3], keepdims=True)
    return (image - min_vals) / (max_vals - min_vals) - 0.5


def get_vit_config(batch_size: int, dataset: str) -> ml_collections.ConfigDict:
    config = common_config.get_config()
    config.tfds_manual_dir = "." # TODO ds_info kommt irgendwo anders her
    # Manually overwrite dataset config to use only validation split
    dataset_config = ml_collections.ConfigDict(
        {"total_steps": 10_000, "pp": {"test": "validation", "crop": 384}}
    )
    config.dataset = dataset
    config.update(dataset_config)
    config.batch = batch_size
    config.batch_eval = batch_size
    return config


def initialize_model(model_name: str) -> vit_jax.models.VisionTransformer:
    model_config = models_config.MODEL_CONFIGS[model_name]
    return models.VisionTransformer(num_classes=1000, **model_config)