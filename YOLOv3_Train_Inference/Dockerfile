#!/bin/sh
FROM python:3.8

RUN apt-get update && apt install net-tools
RUN pip3 install tqdm

WORKDIR /app
COPY ./*.py /app
RUN mkdir -p models/
RUN mkdir -p content/
RUN mkdir -p logs/
COPY ./models/*.pt /app/models
COPY ./requirement.txt /app/
RUN pip3 install -r /app/requirement.txt 

EXPOSE 9876