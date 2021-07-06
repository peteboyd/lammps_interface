#!/bin/bash

git rev-list HEAD | wc -l | awk '{$1=$1;print}' > VERSION_INFO
git rev-parse --short HEAD >> VERSION_INFO 
