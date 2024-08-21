#!/usr/bin/bash


echo "HOME=$HOME RUNTIME=$1 USER_ID=$(id -u) GROUP_ID=$(id -g) docker compose up -d --build"
HOME=$HOME RUNTIME=$1 USER_ID=$(id -u) GROUP_ID=$(id -g) docker compose up -d --build

