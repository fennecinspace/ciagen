#!/usr/bin/bash


RUNTIME=$1 USER_ID=$(id -u) GROUP_ID=$(id -g) docker compose up -d --build
