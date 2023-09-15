#!/bin/sh
FROM python:3

RUN apt-get update && apt install net-tools

RUN mkdir /data
RUN mkdir /logs
RUN touch /logs/log.txt
RUN touch /data/received.txt
WORKDIR /app
COPY ./server.py /app

EXPOSE 5001
