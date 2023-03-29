import numpy as np
import scipy
import cv2

# much faster version of https://github.com/davideverona/deep-crowd-counting_crowdnet/blob/master/dcc_crowdnet.ipynb
def gaussian_filter_density(pts, image_shape, k_neighbors=3):
    k_neighbors = k_neighbors+1
    density = np.zeros(image_shape, dtype=np.float32)
    gt_count = pts.shape[0]
    if gt_count == 0:
        return density
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=k_neighbors)

    for i, pt in enumerate(pts):
        if gt_count > 1:
            # sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
            sigma = np.mean(distances[i,1:]) * 0.3
        else:
            sigma = np.average(np.array(image_shape))/2./2. #case: 1 point

        min_distance_by_border = (np.minimum( np.minimum(pt[0], image_shape[1] - pt[0] - 1), np.minimum(pt[1], image_shape[0] - pt[1] - 1)) * 2) + 1

        # elements with a distance of more than 3 sigma of the center can be considered zero(see https://en.wikipedia.org/wiki/Gaussian_blur#Mathematics)
        kernel_size = int(np.int(3*2*sigma))
        kernel_size = kernel_size + (1-np.mod(kernel_size, 2) )
        kernel_size = np.minimum(kernel_size, min_distance_by_border)
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        kernel = np.multiply(kernel, kernel.T)
        indexes = np.meshgrid(np.arange(pt[1]-(kernel_size//2), pt[1]+(kernel_size//2)+1) ,np.arange(pt[0]-(kernel_size//2), pt[0]+(kernel_size//2)+1 ))
        density[indexes[0], indexes[1]] += kernel

    return density
