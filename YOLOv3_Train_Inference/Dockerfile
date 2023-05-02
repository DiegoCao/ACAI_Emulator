#!/bin/sh
FROM python:3.8

RUN apt-get update && apt install net-tools
RUN pip3 install tqdm

WORKDIR /app
COPY ./*.py /app
COPY ./*.pt /app 
COPY ./requirement.txt /app/
RUN pip3 install -r /app/requirement.txt 

EXPOSE 5001