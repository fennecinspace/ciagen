FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ARG GROUP_ID=1000
ARG USER_ID=1000

RUN addgroup --gid $GROUP_ID user
RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user

RUN apt-get update && apt -y install ffmpeg libsm6 libxext6 build-essential

# to build doc to pdf https://www.sphinx-doc.org/en/master/usage/builders/index.html#sphinx.builders.latex.LaTeXBuilder
RUN DEBIAN_FRONTEND=noninteractive apt-get install tzdata -y

# Install zsh and oh-my-zsh
RUN apt-get update && apt-get install -y curl zsh git wget
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"

ENV PATH="$PATH:/home/user/.local/bin"

WORKDIR gen_data

COPY requirements.txt requirements.txt

USER user

RUN pip install ninja
RUN pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
RUN pip install -r requirements.txt
