#!/bin/bash
conda install pytorch torchvision cpuonly -c pytorch &&
pip install opencv-python &&
conda install numpy pandas matplotlib
