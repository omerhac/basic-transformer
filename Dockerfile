FROM ubuntu:latest

RUN apt-get update -y

RUN apt-get install -y python3-pip

COPY . /app

WORKDIR /app

RUN pip3 install --upgrade pip

RUN pip3 install -r requirements.txt

RUN python3 -m pip --no-cache-dir install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow_cpu-2.1.0-cp37-cp37m-manylinux2010_x86_64.whl
