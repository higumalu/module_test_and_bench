
import time
import os
import pydicom
import cv2
import skimage
import numpy as np

from scipy.ndimage import zoom
from skimage.transform import resize




dir_path = "ACCU_ANA\CT_3"

ds_list = [pydicom.dcmread(os.path.join(dir_path, dcm_path)) for dcm_path in os.listdir(dir_path)]

# image_volume = np.array([ds.pixel_array * ds.RescaleSlope + ds.RescaleIntercept for ds in ds_list])
# print("original volume shape: ", image_volume.shape)

shape = (128, 128, 128)
mask = np.zeros(shape, dtype=float)
center = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
radius = 30

for x in range(shape[2]):
    for y in range(shape[1]):
        for z in range(shape[0]):
            if (x - center[0])**2 + (y - center[1])**2 <= ((z - center[2]) * 2 * radius / shape[2])**2:
                mask[z, y, x] = 1

image_volume = mask.astype(float)



def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start} seconds")
        return result
    return wrapper


@timeit
def resize_volume_scipy(image_volume, target_shape: tuple[int, int, int]) -> np.ndarray:
    ori_shape = image_volume.shape
    resize_volume = zoom(image_volume,
                         (target_shape[0] / ori_shape[0], target_shape[1] / ori_shape[1], target_shape[2] / ori_shape[2]),
                         mode='nearest')
    return resize_volume


@timeit
def resize_volume_cv2(image_volume, target_shape: tuple[int, int, int]) -> np.ndarray:
    target_z, target_y, target_x = target_shape

    resize_xz_list = [cv2.resize(image_volume[:, index, :], (target_x, target_z), interpolation=cv2.INTER_LINEAR)
                      for index in range(image_volume.shape[1])]
    image_volume = np.array(resize_xz_list).transpose(1, 0, 2)

    resize_xy_list = [cv2.resize(image_volume[index, :, :], (target_x, target_y), interpolation=cv2.INTER_LINEAR)
                      for index in range(image_volume.shape[0])]
    image_volume = np.array(resize_xy_list)
    
    return image_volume


@timeit
def resize_volume_skimage(image_volume, target_shape: tuple[int, int, int]) -> np.ndarray:
    resize_volume = skimage.transform.resize(image_volume, target_shape)
    return resize_volume


resize_volume1 = resize_volume_scipy(image_volume, (408, 512, 512))
resize_volume2 = resize_volume_skimage(image_volume, (408, 512, 512))
resize_volume3 = resize_volume_cv2(image_volume, (408, 512, 512))



import matplotlib.pyplot as plt


plt.figure(figsize=(30, 10))
for i in range(0, 408, 10):

    plt.subplot(1, 3, 1)
    plt.imshow(resize_volume1[i, :, :])
    plt.title("scipy")

    plt.subplot(1, 3, 2)
    plt.imshow(resize_volume2[i, :, :])
    plt.title("skimg")

    plt.subplot(1, 3, 3)
    plt.imshow(resize_volume3[i, :, :])
    plt.title("cv2")

    plt.show()


# print("resize volume shape: ", resize_volume.shape)

