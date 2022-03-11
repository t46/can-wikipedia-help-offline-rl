FROM python:3.8-slim-buster

RUN apt-get install sudo && \
    sudo apt-get install vim && \
    sudo apt-get install git && \
    sudo apt-get install libosmesa6-dev && \
    sudo apt-get install patchelf
RUN pip install mujoco-py && \
    pip install gym && \
    pip install transformers && \
    pip install wandb

# Create the environment:
COPY . .

ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin/
CMD ["bash"]
