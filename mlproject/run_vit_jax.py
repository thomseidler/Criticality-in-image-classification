import argparse

import jax
import numpy as np
import tensorflow as tf
import ml_collections
import matplotlib.pyplot as plt

import mlflow
import mantik
import functools
import argparse

from vit_jax import checkpoint
from vit_jax import models
from vit_jax import input_pipeline

from vit_jax.configs import common as common_config
from vit_jax.configs import models as models_config

import critical_images

import pathlib

def get_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--weights-path", type=str, default=".")
    parser.add_argument("--noise-steps", type=str)
    parser.add_argument("--data-cache-dir", type=str, default=".")
    return parser.parse_args()

if __name__ == "__main__":
    mlflow.autolog()

    commandline_arguments = get_arguments()
    model_name = commandline_arguments.model_name
    batch_size = commandline_arguments.batch_size
    noise_steps = list(map(int, commandline_arguments.noise_steps.replace("(", "").replace(")","").split(",")))
    weights_path = commandline_arguments.weights_path
    data_path = commandline_arguments.data_cache_dir

    #model_name = "ViT-B_32"

    dataset = "imagenet2012"
    #batch_size = 64  # 512


    config = critical_images.vit.get_vit_config(batch_size, dataset, data_path=data_path)

    ds_test = input_pipeline.get_data_from_tfds(config=config, mode="test")
    num_classes = input_pipeline.get_dataset_info(dataset, "validation")[
        "num_classes"
    ]

    ds_test = ds_test.shuffle(config.batch_eval)

    ds_test = ds_test.take(256) # TODO Das kann spaeter raus

    model_config = models_config.MODEL_CONFIGS[model_name]
    model = models.VisionTransformer(num_classes=1000, **model_config)

    model = critical_images.vit.initialize_model(model_name)

    params = checkpoint.load(f"{weights_path}/{model_name}_imagenet2012.npz")
    params["pre_logits"] = {}  # Need to restore empty leaf for Flax.

    # TODO I might want to replicate params to all devices on node

    vit_apply_repl = jax.jit(
        lambda params_repl, inputs: model.apply(
            dict(params=params_repl), inputs, train=False
        )
    )

    noise_schedule = critical_images.noise.NoiseSchedule(steps=5000)

    mlflow.log_param("image_size", config.pp.crop)
    mlflow.log_param("batch_size", config.batch_eval)
    # mlflow.log_param("validation_size", VALIDATION_SIZE)
    mlflow.log_param("weights", "imagenet2012")
    mlflow.log_param("validation_data", dataset)
    mlflow.log_param("model", model_name)

    for noise_step in noise_steps:

        noisify_inner = functools.partial(
            critical_images.noise.get_noisy_sample,
            step=noise_step,
            noise_schedule=noise_schedule,
        )

        def noisify(record):
            return {
                "image": noisify_inner(record["image"]),
                "label": record["label"],
            }

        # Apply noise
        ds_noise = ds_test.map(noisify)

        ds_noise = ds_noise.map(
            lambda record: {
                "image": critical_images.vit.preprocess_image(record["image"]),
                "label": record["label"],
            }
        )

        # TODO cleanup, more logging? - good, total, noise sigma and mu, transfer to and run on JUWELS, all weights for ViT, implement EffNet, debug jax

        mlflow.log_metric(
            "Accuracy",
            critical_images.vit.get_accuracy(params, ds_noise, apply_function=vit_apply_repl, batch_size=config.batch_eval),
            step=noise_step,
        )

