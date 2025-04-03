import matplotlib.pyplot as plt
import cv2
import numpy as np
from scipy.signal import convolve2d
imgL=cv2.imread(r"left.png",cv2.IMREAD_GRAYSCALE)
imgR=cv2.imread(r"right.png",cv2.IMREAD_GRAYSCALE)

def get_disparity(im1, im2, max_disp, win_size):
    h,w=im1.shape
    dispM=np.zeros((h, w), dtype=np.float32)
    min_ssd=np.full((h, w), np.inf)

    im1 = im1.astype(np.float32)
    im2 = im2.astype(np.float32)
    kernel = np.ones((win_size, win_size), dtype=np.float32)
    for d in range(max_disp + 1):
        # Shift im2 by d pixels to the right
        shifted_im2 = np.zeros_like(im2)
        if d > 0:
            shifted_im2[:, d:] = im2[:, :-d]
        else:
            shifted_im2 = im2.copy()
        
        ssd = (im1 - shifted_im2) ** 2 # Compute ssd and sum using convolution
        ssd_sum = convolve2d(ssd, kernel, mode='same', boundary='fill', fillvalue=0)

        # Updating disparity 
        for y in range(h):
            for x in range(w):
                if ssd_sum[y, x] < min_ssd[y, x]:
                    min_ssd[y, x] = ssd_sum[y, x]
                    dispM[y, x] = d
    return dispM

def create_depth_colormap(disparity_map):
    inverted = 255 - disparity_map
    colormap = plt.get_cmap('jet')
    depth_colored = colormap(inverted)
    depth_colored = (depth_colored[:, :, :3] * 255).astype(np.uint8)
    return depth_colored

plt.subplot(1,3,1)
plt.imshow(imgL,cmap="gray")
plt.title("Left")
plt.axis("off")
plt.subplot(1,3,2)
plt.imshow(imgR,cmap="gray")
plt.title("Right")
plt.axis("off")

disparity = get_disparity(imgL, imgR, 64, 25)
plt.subplot(1,3,3)
plt.imshow(disparity,cmap="gray")
plt.title("Disparity")
plt.axis("off")
plt.show()
disparity=cv2.normalize(disparity,None,alpha=0,beta=255,norm_type=cv2.NORM_MINMAX)
disparity=np.uint8(disparity)
depth_map = create_depth_colormap(disparity)
plt.imshow(depth_map)
plt.title("DEPTH MAP")
plt.show()

