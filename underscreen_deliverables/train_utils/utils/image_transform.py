from abc import ABC, abstractmethod


from scipy.ndimage import gaussian_filter
import numpy as np
import cv2
from skimage.transform import rescale, resize


def adjust_gamma(image, gamma=0.5):
    image = image ** gamma
    return image


def make_blurred(image, sigma):
    for channel in range(image.shape[-1]):
        image[:, :, channel] = gaussian_filter(image[:, :, channel], sigma=sigma)
    return image


def get_high_pass_filtered_image(image, sigma):
    blurred_image = make_blurred(image, sigma)
    filtered = image-blurred_image+0.5
    return filtered


def linear_blending(blend, target):    
    result = (blend>0.5) * (target + 2*(blend-0.5)) + (blend<=0.5) * (target + 2*blend-1)
    result = np.clip(result, 0, 1)
    return result


def align_images(target, source):
    # Convert to grayscale. 
    img1 = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY) 
    img2 = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY) 
    img1 = (img1*255).astype(np.uint8)
    img2 = (img2*255).astype(np.uint8)
    
    height, width = img2.shape     
    
    # Create ORB detector with 50000 features. 
    orb_detector = cv2.ORB_create(100000) 

    # Find keypoints and descriptors. 
    # The first arg is the image, second arg is the mask 
    #  (which is not reqiured in this case). 
    
    kp1, d1 = orb_detector.detectAndCompute(img1, None) 
    kp2, d2 = orb_detector.detectAndCompute(img2, None) 

    # Match features between the two images. 
    # We create a Brute Force matcher with  
    # Hamming distance as measurement mode. 
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True) 

    # Match the two sets of descriptors. 
    matches = matcher.match(d1, d2) 

    # Sort matches on the basis of their Hamming distance. 
    matches.sort(key = lambda x: x.distance) 

    # Take the top 90 % matches forward. 
    matches = matches[:int(len(matches)*90)] 
    no_of_matches = len(matches) 

    # Define empty matrices of shape no_of_matches * 2. 
    p1 = np.zeros((no_of_matches, 2)) 
    p2 = np.zeros((no_of_matches, 2)) 

    for i in range(len(matches)): 
        p1[i, :] = kp1[matches[i].queryIdx].pt 
        p2[i, :] = kp2[matches[i].trainIdx].pt 

    # Find the homography matrix. 
    homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC) 

    # Use this matrix to transform the 
    # colored image wrt the reference image. 
    transformed_img = cv2.warpPerspective(source, 
                        homography, (width, height)) 
    return transformed_img

class ImageTransform(ABC):
    """
    Parent class for making an arbitrary transform on given image
    """
    @abstractmethod
    def transform(self, input_image):
        pass


class ImageTransformGamma(ImageTransform):

    def __init__(self, gamma=0.5):
        self.gamma = gamma

    def transform(self, input_image):
        output_image = adjust_gamma(input_image, gamma=self.gamma)
        return output_image


class ImageTransformHPFilter(ImageTransform):

    def __init__(self, sigma=5):
        self.sigma = sigma

    def transform(self, input_image):
        output_image = get_high_pass_filtered_image(input_image, self.sigma)
        return output_image


class ImageTransformDownscale(ImageTransform):

    def __init__(self, downscale_ratio, anti_aliasing=False, order=3):
        self.downscale_ratio = downscale_ratio
        self.anti_aliasing = anti_aliasing
        self.order = order

    def transform(self, input_image):
        output_image = rescale(input_image,
                               order=self.order,
                               scale=1/self.downscale_ratio,
                               multichannel=True,
                               anti_aliasing=self.anti_aliasing)
        return output_image
