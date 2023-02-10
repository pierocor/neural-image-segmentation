"""
Baseline script performing inference on a High Resolution image.
The program performs 3 tasks:
- Load the image
- Perform gamma correction
- Inference (loading the model from checkpoint)
"""
from unet.unet_utils import gamma_correction, unet_predict
from postImgProc.utils import readImg
from time import time
import warnings
# import matplotlib.pyplot as plt


def main():
    warnings.simplefilter("ignore")
    print("Starting application...")
    start = time()
    # img = readImg('data/Axon2.png')
    img = readImg('data/Snap-775.tif')
    print(f"Image loaded in: {time() - start:.3f} seconds")
    start = time()
    gamma_image = gamma_correction(img)
    print(f"Gamma correction in: {time() - start:.3f} seconds")
    start = time()
    seg_image = unet_predict(gamma_image)
    print(f"Prediction in: {time() - start:.3f} seconds")
    # plt.imshow(seg_image)
    # plt.show()


if __name__ == '__main__':
    try:
        main = profile(main)
        print("Profiling memory")
    except:
        pass

    main()