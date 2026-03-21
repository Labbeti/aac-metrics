#!/bin/bash
# -*- coding: utf-8 -*-

pkg_name="aac_metrics"

docs_dpath=`dirname $0`
cd "$docs_dpath"

rm ${pkg_name}.*rst
uv run sphinx-apidoc -e -M -o . ../src/${pkg_name} && uv run make clean && uv run make html

exit 0
