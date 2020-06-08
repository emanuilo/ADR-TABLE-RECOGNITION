# base image
FROM python:3.7-slim-buster

RUN apt-get update \
  && apt-get -y install tesseract-ocr \
  && apt-get -y install libtesseract-dev \
  && apt-get -y install libsm6 libxext6 libxrender-dev \
  && apt-get -y install gcc \
  && apt-get clean && rm -rf /var/lib/apt/lists/*

# set working directory
WORKDIR /usr/src/app

# add and install requirements
COPY ./requirements.txt /usr/src/app/requirements.txt
RUN pip install -r requirements.txt

COPY . /usr/src/app
RUN cd darkflow && pip install -e .
