# Criticality in image classification

We use criticality analysis on image classification models on noisy data.

## Usage

- Install the project with `poetry install`.
- To download the weights, you need [`gsutil`](https://cloud.google.com/storage/docs/gsutil_install?hl=de#deb). Be aware that there is a differen `gsutil` package on apt.
  - View all available weights: `gsutil ls -lh gs://vit_models/imagenet*`
  - Download appropriate weights, e.g. `gsutil cp gs://vit_models/imagenet21k+imagenet2012/ViT-B_32.npz .`
- Download imagenet labels: `wget https://storage.googleapis.com/bit_models/ilsvrc2012_wordnet_lemmas.txt`
