FROM python:3.8-slim-buster

RUN pip3 install mujoco-py && \
    pip3 install gym && \
    pip3 install transformers && \
    pip3 install wandb

# Create the environment:
COPY . .
CMD ["bash"]
