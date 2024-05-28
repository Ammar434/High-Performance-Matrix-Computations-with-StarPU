#!/bin/bash

WD=`pwd`
SCRIPT="bash ${WD}/setup/install.sh;"
HOST="sh10"
ssh ${HOST} "${SCRIPT}"
