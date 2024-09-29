# Use an official Python runtime as a parent image
FROM python:3.10
RUN pip install sagemaker-training
# Set the working directory in the container
WORKDIR /

# Copy the current directory contents into the container at /usr/src/app
##COPY . .

# Install any needed packages specified in requirements.txt
##RUN pip install --no-cache-dir -r requirements.txt

# Run app.py when the container launches
#CMD ["python", "./app.py"]

# Part of the implementation of this container is based on the Amazon SageMaker Apache MXNet container.
# https://github.com/aws/sagemaker-mxnet-container

#FROM ubuntu:16.04
#
#LABEL maintainer="Amazon AI"
#
# Defining some variables used at build time to install Python3
#ARG PYTHON=python3
#ARG PYTHON_PIP=python3-pip
#ARG PIP=pip3
#ARG PYTHON_VERSION=3.7.2



# Install some handful libraries like curl, wget, git, build-essential, zlib
#RUN apt-get update && apt-get install -y --no-install-recommends software-properties-common && \
#    add-apt-repository ppa:deadsnakes/ppa -y && \
#    apt-get update && apt-get install -y --no-install-recommends \
#        build-essential \
#        ca-certificates \
#        curl \
#        wget \
#        git \
#        libopencv-dev \
#        openssh-client \
#        openssh-server \
#        vim \
#        zlib1g-dev && \
#    rm -rf /var/lib/apt/lists/*

# Installing Python3
#RUN wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tgz && \
#        tar -xvf Python-$PYTHON_VERSION.tgz && cd Python-$PYTHON_VERSION && \
#        ./configure && make && make install && \
#        apt-get update && apt-get install -y --no-install-recommends libreadline-gplv2-dev libncursesw5-dev libssl-dev libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev && \
#        make && make install && rm -rf ../Python-$PYTHON_VERSION* && \
#        ln -s /usr/local/bin/pip3 /usr/bin/pip

# Upgrading pip and creating symbolic link for python3
#RUN ${PIP} --no-cache-dir install --upgrade pip
#RUN ln -s $(which ${PYTHON}) /usr/local/bin/python

WORKDIR /
COPY . .
# Installing numpy, pandas, scikit-learn, scipy
#RUN ${PIP} install --no-cache --upgrade \
#        numpy==1.14.5 \
#        pandas==0.24.1 \
#        scikit-learn==0.20.3 \
#        requests==2.21.0 \
#        scipy==1.2.1
        
RUN pip --no-cache-dir install -r requirements.txt

# Setting some environment variables.
#ENV PYTHONDONTWRITEBYTECODE=1 \
#    PYTHONUNBUFFERED=1 \
#    LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/lib" \
#    PYTHONIOENCODING=UTF-8 \
#    LANG=C.UTF-8 \
#    LC_ALL=C.UTF-8

#to resolve error: unrecognized arguments: train, follow https://docs.aws.amazon.com/sagemaker/latest/dg/adapt-training-container.html#:%7E:text=Step%202%3A%20Create%20and%20upload%20the%20Dockerfile%20and%20Python%20training%20scripts

ENV SAGEMAKER_PROGRAM app.py
#ENTRYPOINT ["python", "./app.py"]