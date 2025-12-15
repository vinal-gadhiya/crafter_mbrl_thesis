FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-devel

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=America/San_Francisco
ENV PYTHONUNBUFFERED 1
ENV PIP_DISABLE_PIP_VERSION_CHECK 1
ENV PIP_NO_CACHE_DIR 1


RUN apt-get update && apt-get install -y software-properties-common

RUN add-apt-repository -y ppa:openjdk-r/ppa
RUN apt-get purge openjdk-*

RUN apt-get update && apt-get install -y \
    vim libglew2.1 libgl1-mesa-glx libosmesa6 \
    wget unrar cmake g++ libgl1-mesa-dev \
    libx11-6 openjdk-8-jdk x11-xserver-utils xvfb \
    python-opengl ffmpeg tmux\
    && apt-get clean

WORKDIR /workspace

RUN apt update
RUN pip3 install gym[all]==0.25.2
RUN pip3 install crafter
RUN pip3 install numpy==1.24.3
