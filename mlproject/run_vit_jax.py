import flax
import jax
from matplotlib import pyplot as plt
import numpy as np
import optax
import tqdm
import tensorflow_datasets as tfds
import tensorflow as tf
import ml_collections
import matplotlib.pyplot as plt

import mlflow
import mantik
import functools

from vit_jax import checkpoint
from vit_jax import input_pipeline
from vit_jax import models
from vit_jax import input_pipeline

from vit_jax.configs import common as common_config
from vit_jax.configs import models as models_config

import PIL

import critical_images


def get_accuracy(params, ds_test, apply_function):
    """Returns accuracy evaluated on the test set."""
    good = total = 0
    steps = len(ds_test) // batch_size
    for _, batch in zip(tqdm.trange(steps), ds_test.as_numpy_iterator()):
        if _ == 0:
            fig, ax = plt.subplots()
            print((batch["image"][0, 0]))
            img = batch["image"][0, 0] + 0.5
            ax.imshow(img)
            fig.savefig("test.png")
        predicted = apply_function(params, batch["image"][0, :])
        is_same = predicted.argmax(axis=-1) == batch["label"].argmax(axis=-1)
        good += is_same.sum()
        total += len(is_same.flatten())
        print(good / total)
    return good / total


if __name__ == "__main__":
    try:  # Workaround for this bug: https://github.com/tensorflow/datasets/issues/5355
        MODEL_NAME = "ViT-B_32"

        dataset = "imagenet2012"
        batch_size = 64  # 512
        config = common_config.get_config()
        config.tfds_manual_dir = "."
        # Manually overwrite dataset config to use only validation split
        dataset_config = ml_collections.ConfigDict(
            {"total_steps": 10_000, "pp": {"test": "validation", "crop": 384}}
        )
        config.dataset = dataset
        config.update(dataset_config)
        config.batch = batch_size
        config.batch_eval = batch_size

        ds_test = input_pipeline.get_data_from_tfds(config=config, mode="test")
        num_classes = input_pipeline.get_dataset_info(dataset, "validation")[
            "num_classes"
        ]

        # ds_test = ds_test.shuffle(config.batch_eval)

        ds_test = ds_test.take(256)

        model_config = models_config.MODEL_CONFIGS[MODEL_NAME]
        model = models.VisionTransformer(num_classes=1000, **model_config)

        params = checkpoint.load(f"{MODEL_NAME}_imagenet2012.npz")
        params["pre_logits"] = {}  # Need to restore empty leaf for Flax.

        vit_apply_repl = jax.jit(
            lambda params_repl, inputs: model.apply(
                dict(params=params_repl), inputs, train=False
            )
        )

        noise_schedule = critical_images.noise.NoiseSchedule(steps=5000)
        noise_steps = [
            0,
            100,
            250,
            500,
            750,
            1000,
            1125,
            1250,
            1375,
            1500,
            1625,
            1750,
            2000,
            2500,
            3000,
            3500,
            4000,
            4999,
        ]

        noise_steps = [1000]

        mlflow.log_param("image_size", config.pp.crop)
        mlflow.log_param("batch_size", config.batch_eval)
        # mlflow.log_param("validation_size", VALIDATION_SIZE)
        mlflow.log_param("weights", "imagenet2012")
        mlflow.log_param("validation_data", dataset)
        mlflow.log_param("model", MODEL_NAME)

        for noise_step in noise_steps:

            noisify_inner = functools.partial(
                critical_images.noise.get_noisy_sample,
                step=noise_step,
                noise_schedule=noise_schedule,
            )

            def noisify(record):
                print("RECORD", record["image"])
                return {
                    "image": noisify_inner(record["image"]),
                    "label": record["label"],
                }

            # Apply noise
            ds_noise = ds_test.map(noisify)

            def preprocess_image(image: np.ndarray) -> np.ndarray:
                min_vals = tf.math.reduce_min(image, axis=[1, 2, 3], keepdims=True)
                max_vals = tf.math.reduce_max(image, axis=[1, 2, 3], keepdims=True)
                print(min_vals)
                return (image - min_vals) / (max_vals - min_vals) - 0.5

            ds_noise = ds_noise.map(
                lambda record: {
                    "image": preprocess_image(record["image"]),
                    "label": record["label"],
                }
            )

            mlflow.log_metric(
                "Accuracy",
                get_accuracy(params, ds_noise, apply_function=vit_apply_repl),
                step=noise_step,
            )
        print("Done")
    except TypeError:
        pass
