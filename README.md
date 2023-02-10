# neural-image-segmentation
## Description
This project implements an interactive user interface that will apply our Neural Image segmentation model and Image post processing technique 
on the user input neuroblastoma images.

The main file for the UI is located at /neural-image-segmentation/imageSeg.py
## Dependencies
Python 3.8<br />
Pip packages: install them by running the install_dep.sh


## Inference Optimization
The current model is too compute and memory intensive. Pruning and quantisation or decreasing the number of convolutional blocks could ease this issue.

The script `neural-image-segmentation/bench_inference.py` loads, preprocesses and performs inference on a single high-resolution image (namely `Data_larged_images/images/Snap-775.tif`).

On a Intel Xeon IceLake-SP processors (Platinum 8360Y) with 72 cores and 256 GB RAM:
```
Profiling memory
Starting application...
Image loaded in: 0.166 seconds
Gamma correction in: 0.400 seconds
Prediction in: 5.515 seconds
Filename: neural-image-segmentation/bench_inference.py

Line #    Mem usage    Increment  Occurrences   Line Contents
=============================================================
    15  539.695 MiB  539.695 MiB           1   def main():
    16  539.695 MiB    0.000 MiB           1       warnings.simplefilter("ignore")
    17  539.695 MiB    0.000 MiB           1       print("Starting application...")
    18  539.695 MiB    0.000 MiB           1       start = time()
    19                                             # img = readImg('data/Axon2.png')
    20  570.781 MiB   31.086 MiB           1       img = readImg('data/Snap-775.tif')
    21  570.781 MiB    0.000 MiB           1       print(f"Image loaded in: {time() - start:.3f} seconds")
    22  570.781 MiB    0.000 MiB           1       start = time()
    23  571.039 MiB    0.258 MiB           1       gamma_image = gamma_correction(img)
    24  571.039 MiB    0.000 MiB           1       print(f"Gamma correction in: {time() - start:.3f} seconds")
    25  571.039 MiB    0.000 MiB           1       start = time()
    26 1018.918 MiB  447.879 MiB           1       seg_image = unet_predict(gamma_image)
    27 1018.918 MiB    0.000 MiB           1       print(f"Prediction in: {time() - start:.3f} seconds")
```
The memory count is misleading. Indeed, using mprof (sampling every 0.1s):

![data/memory_bench.png](data/memory_bench.png)