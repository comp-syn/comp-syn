import numpy as np
import scipy.stats
import matplotlib.colors as mplcolors
from PIL import Image
from sklearn.cluster import KMeans
from collections import Counter

def kl_divergence(dist1, dist2, symmetrized=True):
    """
    Calculates Kullback-Leibler (KL) divergence between two distributions, with an option for symmetrization

    Args:
        dist1 (array): first distribution
        dist2 (array): second distribution
        symmetrized (Boolean): flag that defaults to symmetrized KL divergence, and returns non-symmetrized version if False

    Returns:
        kl (float): (symmetrized) KL divergence
    """
    if symmetrized == True:
        kl = (
            scipy.stats.entropy(dist1, dist2) + scipy.stats.entropy(dist2, dist1)
        ) / 2.0
        return kl
    else:
        kl = scipy.stats.entropy(dist1, dist2)
        return kl


def js_divergence(dist1, dist2):
    """
    Calculates Jensen-Shannon (JS) divergence between two distributions

    Args:
        dist1 (array): first distribution
        dist2 (array): second distribution

    Returns:
        js (float): JS divergence
    """
    mean_dist = (dist1 + dist2) / 2.0
    js = (
        scipy.stats.entropy(dist1, mean_dist) + scipy.stats.entropy(dist2, mean_dist)
    ) / 2.0
    return js


def rgb_array_to_jzazbz_array(rgb_array: np.ndarray) -> np.ndarray:
    """
    Converts rgb pixel values to JzAzBz pixel values
    ​
    Args:
        rgb_array (array): matrix of rgb pixel values
    ​
    Returns:
        jzazbz_array (array): matrix of JzAzBz pixel values
    """

    r = rgb_array[:, :, 0].reshape([-1])
    g = rgb_array[:, :, 1].reshape([-1])
    b = rgb_array[:, :, 2].reshape([-1])
    try:
        from .jzazbz import JZAZBZ_ARRAY_NPY
    except ImportError as exc:
        raise ImportError(
            f"This usually means that no jzazbz_array.npy file could be found at {os.getenv('COMPSYN_JZAZBZ_ARRAY')}"
        ) from exc

    jzazbz_vals = JZAZBZ_ARRAY_NPY[r, g, b]
    jzazbz_array = jzazbz_vals.reshape(list(rgb_array.shape[:3])).transpose([0, 1, 2])
    return jzazbz_array


def bin_img(
    img,
    num_bins,
    x_min,
    x_max,
    y_min,
    y_max,
    z_min,
    z_max
    num_channels=3):
    """
    Calculates the distribution of rgb or JzAzBz pixel values in an image
    ​
    Args:
        img (array): matrix of rgb or jzazbz pixel values
        num_bins (int): total number of colorspace subvolumes
        {xyz}_min, {xyz}_max (floats): minimum and maximum coordinates of each colorspace dimension
        num_channels (int): number of color channels
    ​
    Returns:
        dist (array): distribution of of rgb JzAzBz pixel values in specified bins
    """
    dist = np.histogramdd(
                        np.reshape(
                            img[:, :, :],
                            (
                                img.shape[0]
                                * img.shape[1],
                                num_channels,
                            ),
                        ),
                        bins=(
                            np.linspace(
                                x_min,
                                x_max,
                                1 + int(num_bins ** (1.0 / num_channels)),
                            ),
                            np.linspace(
                                y_min,
                                y_max,
                                1 + int(num_bins ** (1.0 / num_channels)),
                            ),
                            np.linspace(
                                z_min,
                                z_max,
                                1 + int(num_bins ** (1.0 / num_channels)),
                            ),
                        ),
                        density=True,
                    )[0]
    return dist


def bin_hsv(
    img_hsv,
    spacing,
    h_max=360):
    """
    Calculates the distribution of hue values in an image
    ​
    Args:
        img_hsv (array): array of hue values
        spacing (int): degrees of rotation subtended by each hsv bin
        h_max (int): maximum hue rotation angle
    ​
    Returns:
        dist (array): distribution of of rgb JzAzBz pixel values in specified bins
    """
    dist = np.histogram(
                    1.0 * h_max * np.ravel(img_hsv[:, :, 0]),
                    bins=np.arange(0, h_max + spacing, spacing),
                    density=True,
                )[0]
    return dist


def color_distribution(
    img,
    colorspace,
    spacing=36,
    num_bins=8,
    num_channels=3,
    Jz_min=0.0,
    Jz_max=0.167,
    Az_min=-0.1,
    Az_max=0.11,
    Bz_min=-0.156,
    Bz_max=0.115,
    h_max=360,
    rgb_max=255,
    ):
    """
    Calculates color distributions for each word in a dictionary
        
    Args:
        img (array): RGB image pixel values as loaded from PIL and compressed to (n,n,3)
        color_rep (string): colorspaces to calculate distributions in; "jzazbz", "hsv", or "rgb"
        spacing(int): hue spacing for HSV distribution (in degrees)
        num_bins(int): number of bins to calculate 3D distributions in
        num_channels(int): number of color channels
        *z_min (*z_max) (float): minimum (maximum) of JzAzBz coordinates
        h_max (int): maximum hue (in degrees)
        rgb_max (int): maximum value in RGB
        
    Returns:
        {}_dist (array): distribution of values (either jzazbz, hsv, or rgb)
    """

    if colorspace == "jzazbz":
        img_jzazbz = rgb_array_to_jzazbz_array(img)
        jzazbz_dist = bin_img(img_jzazbz,num_bins,Jz_min,Jz_max,Az_min,Az_max,Bz_min,Bz_max,num_channels)
        return jzazbz_dist

    elif colorspace == "hsv":
        img_hsv = mplcolors.rgb_to_hsv(img / (1.0 * rgb_max))
        hsv_dist = bin_hsv(img_hsv,spacing,h_max)
        return hsv_dist
 
    elif colorspace == "rgb":
        rgb_dist = bin_img(img,num_bins,0,rgb_max,0,rgb_max,0,rgb_max,num_channels)
        return rgb_dist


def avg_rgb(img):
    """
    Calculates the average rgb coordinates of an image
        
    Args:
        img (array): RGB image pixel values as loaded from PIL and compressed to (n,n,3)
        
    Returns:
        avg_rgb (array): 3x1 array containing [average r, average g, average b]
    """

    r = np.sum(np.ravel(img[:, :, 0]))
    g = np.sum(np.ravel(img[:, :, 1]))
    b = np.sum(np.ravel(img[:, :, 2]))
    tot = 1.0 * r + g + b
    avg_rgb = np.array([r / tot, g / tot, b / tot])
    return avg_rgb


def avg_hsv(img):
    """
    Returns the average hue, saturation, value for an image
        
    Args:
        img (array): RGB image pixel values as loaded from PIL and compressed to (n,n,3)
        
    Returns:
        h, s, v (floats): average hue, saturation, value
    """

    img_hsv = mplcolors.rgb_to_hsv(img / (1.0 * rgb_max))
    h = np.mean(np.ravel(img_hsv[:, :, 0]))
    s = np.mean(np.ravel(img_hsv[:, :, 1]))
    v = np.mean(np.ravel(img_hsv[:, :, 2]))
    return h, s, v

​
#helper function that converts RGB to HEX string 
def RGB2HEX(color):
    '''In: RGB color array
    Out: HEX string'''
​
    return "#{:02x}{:02x}{:02x}".format(int(color[0]), int(color[1]), int(color[2]))
​
​
#function that selects dominant color from image using kmeans 
def get_color(im):
    '''In: PIL Image object
    
    Out: Image color
        (string): HEX code 
        (list): RGB value'''
    #selects center portion of image 
    im = im.crop((100, 100, 200, 200))
    
    im = np.array(im)
    
    modified_image = im.reshape(im.shape[0]*im.shape[1], 3)
​
    #OPTIONAL: can be modified here to select more than one color
    clf = KMeans(n_clusters = 1)
    labels = clf.fit_predict(modified_image)
​
    counts = Counter(labels)
​
    center_colors = clf.cluster_centers_
    ordered_colors = [center_colors[i] for i in counts.keys()]
    hex_colors = [RGB2HEX(ordered_colors[i]) for i in counts.keys()]
    rgb_colors = [ordered_colors[i] for i in counts.keys()]
​
    return hex_colors[0], rgb_colors[0]