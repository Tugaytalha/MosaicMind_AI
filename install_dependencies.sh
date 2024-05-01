#!/bin/bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch &&
pip install opencv-python &&
conda install numpy pandas matplotlib