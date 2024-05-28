import glob
import skimage
from skimage import io
from skimage import filters
from skimage.filters import threshold_mean
from skimage.color import rgb2gray
import numpy as np
import os

#probably shouldn't use-- looking at both edge detected and thresholded images make them look like nightmares to work with

os.chdir("/Users/kericlamb/Desktop/Documents/Research/Stomata/Train/") #make training dataset annotations with makesense.ai
im_path = str("/Users/kericlamb/Desktop/Documents/Research/Stomata/jpegs/") #string form of image path
save_path = str("/Users/kericlamb/Desktop/Documents/Research/Stomata/Thresh/") #path for thresholded images

#grab all files ending in .jpg that are in directory of images
images = glob.glob('{0}/*.jpg'.format(im_path)) #assumes files end in '.jpg' 

for i in range(len(images)):
    image = skimage.io.imread('{0}'.format(images[i]), as_gray=True)
    image = rgb2gray(image)
    t_glob_mean = threshold_mean(image) #sets threshold for image with mean RGB value
    glob_mean = image >= t_glob_mean 
    edge_sobel = filters.sobel(glob_mean) #doesn't look as good as thresholding
    
    name = os.path.basename(images[i]).rstrip('.jpg')+str('_thresh')
    
    #io.imsave('{0}/{1}.jpg'.format(save_path, name), skimage.img_as_uint(glob_mean))
    io.imsave('{0}/{1}.jpg'.format(save_path, name), skimage.img_as_uint(edge_sobel)) #doesn't look as good as thresholding
