FROM continuumio/miniconda3

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update && apt-get install -y libgtk2.0-dev && \
    rm -rf /var/lib/apt/lists/*

RUN conda update -n base -c defaults conda && \
    conda install -c anaconda python=3.8 && \
    conda install -c hcc free-mujoco-py && \
    conda install -c hcc gym && \
    conda install -c huggingface transformers && \
    conda install -c itsandrewtruong wandb

# Create the environment:
COPY . .
RUN conda env create -f conda_env.yml
