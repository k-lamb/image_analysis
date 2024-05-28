#convert tiff to jpeg

#libraries
import glob
from PIL import Image
import os

#define relevant paths
os.chdir("/Users/kericlamb/Desktop/Documents/Research/Stomata/") #home directory
im_path = str('/Users/kericlamb/Desktop/Documents/Research/Stomata/more_images/tifs') #path for tif images to be processed
pro_path = str('/Users/kericlamb/Desktop/Documents/Research/Stomata/more_images/jpegs') #path for jpegs image outtputs

#import all images in im_path 
images = glob.glob('{0}/*.tif'.format(im_path)) #assumes files end in '.jpg' 

for i in range(len(images)):
    im = Image.open(images[i])
    name = os.path.basename(images[i]).rstrip('.tif')
    im.save('{0}/{1}.jpg'.format(pro_path, name))
