#!/bin/bash

DEFAULT_SPICE_ROOT="."

if [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    echo "Install all files for running the java SPICE program in the SPICE_ROOT directory."
    echo "The default spice root path is \"${DEFAULT_SPICE_ROOT}\"."
    echo "Usage: $0 [SPICE_ROOT]"
    exit 0
fi

dpath_spice="$1"
if [ "$dpath_spice" = "" ]; then
	dpath_spice="${DEFAULT_SPICE_ROOT}"
fi

if [ ! -d "$dpath_spice" ]; then
    echo "Error: SPICE_ROOT \"$dpath_spice\" is not a directory."
    exit 1
fi

fname_zip="SPICE-1.0.zip"
fpath_zip="$dpath_spice/$fname_zip"
bn0=`basename $0`

echo "[$bn0] Start installation of SPICE metric java code in directory \"$dpath_spice\"..."

if [ ! -f "$fpath_zip" ]; then
    echo "[$bn0] Zip file not found, downloading from https://panderson.me..."
    wget https://panderson.me/images/SPICE-1.0.zip -P "$dpath_spice"
fi

dpath_unzip="$dpath_spice/SPICE-1.0"
if [ ! -d "$dpath_unzip" ]; then
    echo "[$bn0] Unzipping file $dpath_zip..."
    unzip $fpath_zip -d "$dpath_spice"

    echo "[$bn0] Downloading Stanford models..."
    bash $dpath_unzip/get_stanford_models.sh
fi

dpath_lib="$dpath_spice/lib"
if [ ! -d "$dpath_lib" ]; then
    echo "[$bn0] Moving lib directory to \"$dpath_spice\"..."
    mv "$dpath_unzip/lib" "$dpath_spice"
fi

fpath_jar="$dpath_spice/spice-1.0.jar"
if [ ! -f "$fpath_jar" ]; then
    echo "[$bn0] Moving spice-1.0.jar file to \"$dpath_spice\"..."
    mv "$dpath_unzip/spice-1.0.jar" "$dpath_spice"
fi

echo "[$bn0] SPICE metric Java code is installed."
exit 0
