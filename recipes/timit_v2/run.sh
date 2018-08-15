#!/bin/sh


# Load the configuration.
. ./setup.sh


echo =========================================================================
echo "                         Data Preparation                              "
echo =========================================================================
local/timit_data_prep.sh "$timit"  || exit 1
