# Personal script for gvcallen to sync external data from AWR folder locally into this repo
import shutil
import os

path_simulated = '../../Data/Simulated/SMAMicrostripOpen'                   # path to simulated data
path_measured = '../../Data/Measured/SMAMicrostripOpen'                     # path to measured data
paths = [path_simulated, path_measured]

shutil.rmtree('data')
for path in paths:
    shutil.copytree(path, 'data', dirs_exist_ok=True)