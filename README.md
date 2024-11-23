# CIA: Controllable Image Augmentation Framework based on Stable Diffusion Synthesis

This is a data generation framework that uses [Stable Diffusion](https://huggingface.co/blog/stable_diffusion) with [ControlNet](https://huggingface.co/blog/train-your-controlnet), to do Data Augmentation for:
- Object Detection using [YOLOv8](https://github.com/ultralytics/ultralytics)
- FER (facial emotion recognition)

Models can be trained using a mix of real and generated data. They can also be logged and evaluated.
Quality metrics are available to enhance the training of models thereafter.

<img src="ciagen/docs/images/general_pipeline.png" />


## Installation

We recommend using either a virtual enviroment or a [docker container](#docker-installation).

### Docker
You need [docker compose](https://docs.docker.com/compose/) to run it. Simply do
```bash
./run_and_builder_docker_file.sh nvidia
```
and connect it in the command line with
```bash
docker exec -it ciagen zsh
```
or using your favorite editor.

### Virtual enviroment
Use `conda`, `virtualenv` or another. Do not forget to run
```bash
pip install -r requirements.txt
```

## How to use it:

The current pipe works by performing several tasks, the advised order is:

`prepare_data` &rarr; `gen` &rarr;`dtd` &rarr;`ptd` &rarr;`filtering` &rarr;`mix` &rarr;`train`

The `run.py` script works by means of a configuration file `ciagen/conf/config.yaml` that
can be udpated dynamically:
```bash
python run.py some-new-value=<my-new-value>
```

Thus calling a task on the framework is done by `python run.py task=<my-task>` or modyfing the configuration file directly.

### `prepare_data`

The `prepare_data` task should put your real data to be used as seed in the `data/real`
directory with labels and captions if needed in the following fashion:

```
└─ data/
    ├─ real/
    │   └─ dataset-name/ (e.g., coco)
    │       ├─ train/
    │       │   ├─ images/
    │       │   │   └─ image001.png
    │       │   ├─ labels/
    │       │   │   └─ image001.txt
    │       │   └─ captions/
    │       │       └─ image001.txt
    │       ├─ val/
    │       │   ├─ images/
    │       │   │   └─ image005.png
    │       │   ├─ labels/
    │       │   │   └─ image005.txt
    │       │   └─ captions/
    │       │       └─ image005.txt
    │       └─ test/
    │           ├─ images/
    │           │   └─ image006.png
    │           ├─ labels/
    │           │   └─ image006.txt
    │           └─ captions/
    │               └─ image006.txt
```

For the moment three datasets are already available to use: `coco, flickr30k, fer`. You
can of course add a new script to prepare your own data. Please follow the same schema for your data.

### `auto_caption`

If you have a dataset without any captions, you can use LLM vision solutions to auto caption. We provide two methods

1. server based : using openai (paid) (online)
> for openai you must create an api key, and top up your account with a minimum of 5 dollars here : https://platform.openai.com/

2. local execution : using ollama (free) (local)
> for ollama you must install it here : https://ollama.com/ then you must pull whatever model you want to use (ex: ollama pull llama3.2-vision). Use only multimodal vision models which can be found here : https://ollama.com/search?c=vision


### `gen`

The `gen` task will extract some condition from real data:

- by means of segmentation
- canny
- face features

or any other condition that [ControlNet](https://huggingface.co/blog/train-your-controlnet)
is able to learn.
The  new samples will be store in the corresponding directory:

```
└─ data/
    ├─ real/
    │   └─ ...
    ├─ generated/
    │   └─ dataset-name/ (e.g., coco)
    │       └─ controlnet-model-name/ (e.g., controlnet_segmentation)
    │           ├─ metadata.yaml
    │           ├─ image001_1.png
    │           └─ image001_2.png
    └─ mixed/
        └─ ...
```

The `metadata.yaml` file stores both the initial configuration used to generate the
dataset and will keep the value for future metrics and filtering tasks.

### `dtd`

(D)istribution (t)o (D)istribution metrics are thought to compare real and generated
empiric distributions, some examples are:
 - [Frechet Inception Distance](https://arxiv.org/abs/1706.08500)
 - [Inception Score](https://arxiv.org/abs/1606.03498)

You can specify the metrics that you want to use in the `config.yaml` file in the `metrics` section:
```yaml
metrics:
    dtd:
        - fid
        # - inception_score
        - ...
```

To use them of course run the script: `python run.py task=dtd`

### `ptd`

(P)oint (t)o (D)istribution are meant to function as outlier detectors. As for `dtd`
you can specify them in the `metrics` section of the `config.yaml` file.

### `filtering`

Will write to the `metadata.yaml` file the chosen samples to be kept and use during
training. Specify them in the `config.yaml` file:
```yaml
filtering:
    type: "top-k" #One of "threshold", "top-k" or "top-p"
    value: 4
```

works with better `k` copies, better `p` percent of copies or copies above or below
the specified `threshold`.

### `mix_dataset`

You can specify how you want to mix your initial real data and the generated one for
training or another ML task with the following parameters:

```yaml
ml:
    # number of samples for training, validation, and test
    val_nb: 1000
    test_nb: 1000
    train_nb: 2000

    augmentation_percent: 0.25
    with_filtering: false
    preferred_fe: vit
    filtering_metric: "mld"
    keep_training_size: true
```

The data will be put in the `mixed` directory:
```
└─ data/
    ├─ real/
    │   └─ ...
    ├─ generated/
    │   └─ ...
    └─ mixed/
        └─ dataset-name/ (e.g., coco)
            └─ train_nb/ (e.g., 250)
                └─ [controlnet-model-name]-[augmentation_percent]/ (e.g., controlnet_segmentation-0.1)
                    ├─ data.txt
                    ├─ train.txt
                    ├─ test.txt
                    └─ val.txt
```



### `train`

You can create your own train script to be called with the `train` task. Training scripts
exists for `yolo`  and `fer` classification models using the [Weigths and Bias](https://wandb.ai/) framework.

## Datasets

There are available datastes to test or use the framework:

- [COCO](https://cocodataset.org/#home) PEOPLE dataset :

```bash
python run.py task=prepare_data data.base=coco
```

- [Flickr30K Entities](https://bryanplummer.com/Flickr30kEntities/) PEOPLE dataset :

```bash
python run.py task=prepare_data data.base=flickr30k
```

- FER (facial emotion recognition) dataset:

```bash
python run.py task=prepare_data data.base=fer
```

- MOCS (Moving Objects in Construction Sites) dataset:

```bash
python run.py task=prepare_data data.base=mocs
```


Data will be downloaded and put in the respective files for images, labels and captions.

## Generate images

To generate some images, you can use

```bash
python run.py task=gen
```

See the `conf/config.yaml` file for all details and configuration options.

You can also configure directly on the command line :

```bash
python run.py task=gen model.cn_use=openpose prompt.base="Arnold" prompt.modifier="dancing"
```

If you use the `controlnet_segmentation` ControlNet, You will find your images in `data/generated/controlnet_segmentation` along with the base image and the feature extracted.

The configuration options work for all scripts available in the framework. For example, you can have different initial data sizes by controlling sample numbers :

```bash
python run.py task=prepare_data ml.train_nb=500
```

You can also launch multiple runs. Here's an example of a multi-run with 3 different generators :

```bash
python run.py task=gen model.cn_use=frankjoshua_openpose,fusing_openpose,lllyasviel_openpose
```

List of available models can be found in `conf/config.yaml`. We have 4 available extractors at the moment (Segmentation, OpenPose, Canny, MediaPipeFace), If you add another control-net model, make sure you add one of the following strings to its name to set the extractor to use :

- openpose
- canny
- segmentation
- mediapipe_face

Note that training your own pair of Stable Diffusion and ControlNet models that are
compliant with the [HuggingFace API](https://huggingface.co/) is possible and then
add it as possible condition.


## Create YOLO Dataset and Train :

We use wandb to track and visualize trainings.

```
wandb login
```

Create `train.txt`, `val.txt`, et `test.txt` :

```
./run.sh create_dataset
```

Launch the training !

```
./run.sh train
```

You can both create and launch at the same time to be able to execute multiple training with multiple augmentation percents on your server using hydra :

```
./run.sh create_n_train.py -m ml.augmentation_percent=0.1 ml.sampling.enable=True ml.sampling.metric=dbcnn,brisque ml.sampling.sample=best ml.epochs=15
```

## Download and test models

The download folder can be set in the config file. You'll have folders inside for each wandb project. each project folder contains :

- all models for the project
- summary file with parameters and results (map, precision ..etc)
- in case of running `test.py` : results.csv file containing the test results with map and other info.

```
./run.sh download.py ml.wandb.project=your-project ml.wandb.download.download=true ml.wandb.download.list_all=true

./run.sh test.py
```

**Note** Other scripts exist to execute different studies, like the usage of Active learning, which is still excremental, you can check the `src` folder for those scripts (This code is still not fully integrated into the framework, some path or configuration modifications might be necessary for correct execution).

## Runs Results Plots

Here are some plots for some of the many runs and studies that we performed :

### Coco Sampling

<img src="ciagen/docs/images/COCO_all_samplings_2.png" />

### Flickr Sampling

<img src="ciagen/docs/images/flickr_all_samplings.png" />

### Loss values for COCO

<img src="ciagen/docs/images/COCO_loss.png" />

### Random Sampling - Regular Runs

<img src="ciagen/docs/images/random_sampling.png" />

## Misc

### Kaggle API

To download the dataset related to Face Emotion Recognition (FER) you will need to use Kaggle. The easiest way is to download the dataset using Kaggle API. If you never used Kaggle API, first go to your account page on Kaggle, and go to the settings. Scroll down to the API section and create a new token. This will generate a jsonfile that you can then download on your computer. Move that file to `~/.kaggle/kaggle.json`. Make sure you have access to the dataset you are trying to download, as we are only using private kaggle datasets in the project.

### Data structuring

Please respect this structure when writing code :
images_dir
This project contains the following directory structure:

```
└─ data/
    ├─ real/
    │   └─ dataset-name/ (e.g., coco)
    │       ├─ train/
    │       │   ├─ images/
    │       │   │   └─ image001.png
    │       │   ├─ labels/
    │       │   │   └─ image001.txt
    │       │   └─ captions/
    │       │       └─ image001.txt
    │       ├─ val/
    │       │   ├─ images/
    │       │   │   └─ image005.png
    │       │   ├─ labels/
    │       │   │   └─ image005.txt
    │       │   └─ captions/
    │       │       └─ image005.txt
    │       └─ test/
    │           ├─ images/
    │           │   └─ image006.png
    │           ├─ labels/
    │           │   └─ image006.txt
    │           └─ captions/
    │               └─ image006.txt
    ├─ generated/
    │   └─ dataset-name/ (e.g., coco)
    │       └─ controlnet-model-name/ (e.g., controlnet_segmentation)
    │           ├─ image001_1.png
    │           └─ image001_2.png
    └─ mixed/
        └─ dataset-name/ (e.g., coco)
            └─ train_nb/ (e.g., 250)
                └─ [controlnet-model-name]-[augmentation_percent]/ (e.g., controlnet_segmentation-0.1)
                    ├─ data.txt
                    ├─ train.txt
                    ├─ test.txt
                    └─ val.txt
```

`train, val, test .txt` files contain a list of the images to use, here's an example :

```
/path/to/data/real/coco/images/000000368475.jpg
/path/to/data/real/coco/images/000000368488.jpg
```

### Docker installation:

- [install docker](https://docs.docker.com/engine/install/)
- [install docker compose plugin](https://docs.docker.com/compose/install/) **do not install the docker-desktop environment**
- if you want to use a gpu with docker you need to install the [docker runtime for nvidia](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

after that what I do is
- install [vscode](https://code.visualstudio.com/)
- install the [dev containers plugin](https://code.visualstudio.com/docs/devcontainers/containers)

run the docker script:
```bash
./run_and_build_docker_file.sh nvidia
```

put the `nvidia` argument at the end so the script build the container with the nvidia runtime.

Then using vscode and the dev container plugin connect to the container and code and run stuff from it as if it was your pc.
