from .datahelper import rgb_array_to_jzazbz_array
import numpy as np
from PIL import Image
from kymatio import Scattering2D
import torch


def perform_scattering_transform(im, L=4, J=5):
    """Performs scattering transform on single channel  image array
	In: im (numpy ndarray (128x128))
	Out: scattering coefficient (list)"""

    scattering = Scattering2D(J=J, shape=im.shape, L=L, max_order=2)
    img_tensor = torch.tensor(im.astype(np.float32) / 255.0)
    scattering_coefficients = scattering(img_tensor)
    scattering_coefficients = np.array(scattering_coefficients)
    sc = []
    scattering_coefficients = (
        np.sum(scattering_coefficients, axis=(1, 2)) / scattering_coefficients.shape[1]
    )
    sc.append(scattering_coefficients[0])
    sc1 = scattering_coefficients[1 : J * L + 1]
    sc2 = scattering_coefficients[J * L + 1 :]
    for i in range(int(len(sc1) / L)):
        sc.append(np.mean(sc1[L * i : L * (i + 1)]))
    for i in range(int(len(sc2) / L)):
        sc.append(np.mean(sc2[L * i : L * (i + 1)]))
    return sc


def get_wavelet_embedding(im, mode="JzAzBz"):
    """Generates wavelet image embedding 
	In: im (PIL Image)
	    mode (str) = "JzAzBz", "RGB", "Grey" 
	Out: 
	    wavelet vector (numpy ndarray)"""

    im = im.resize((128, 128))
    print("WARNING: resizing image to 128x128")
    if mode == "Grey":
        im = im.convert("L")
        im = np.array(im)
        sc = np.array(perform_scattering_transform(im))
        return sc
    elif mode == "JzAzBz":
        sc = []
        im = np.array(im)[:, :, :3]
        im = rgb_array_to_jzazbz_array(im)
        for channel in [0, 1, 2]:
            im_channel = im[:, :, channel]
            sc.extend(perform_scattering_transform(im_channel))
        return np.array(sc)
    elif mode == "RGB":
        sc = []
        im = np.array(im)[:, :, :3]
        for channel in [0, 1, 2]:
            im_channel = im[:, :, channel]
            sc.extend(perform_scattering_transform(im_channel))
        return np.array(sc)
    else:
        sc = []
        print("WARNING: unknown mode, defaulting to JzAzBz")
        im = np.array(im)[:, :, :3]
        im = rgb_array_to_jzazbz_array(im)
        for channel in [0, 1, 2]:
            im_channel = im[:, :, channel]
            sc.extend(perform_scattering_transform(im_channel))
        return np.array(sc)
