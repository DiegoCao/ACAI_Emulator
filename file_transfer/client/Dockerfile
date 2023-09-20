#!/bin/sh
FROM python:3

RUN apt-get update && apt install net-tools

RUN mkdir /data
RUN mkdir /logs
RUN touch /logs/client-log.txt
WORKDIR /app
COPY ./client.py /app
