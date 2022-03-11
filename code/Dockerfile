FROM python:3.8-slim-buster

RUN apt -y update && \
    apt -y install sudo && \
    sudo apt -y install vim && \
    sudo apt -y install git && \
    sudo apt -y install libosmesa6-dev && \
    sudo apt -y install patchelf
# Create the environment:
COPY . .

ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin/
CMD ["bash"]
