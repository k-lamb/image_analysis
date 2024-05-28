#pipeline for trained model to process bulk samples

import torch
from detecto import core, utils, visualize
from detecto.utils import reverse_normalize
from detecto.visualize import show_labeled_image, plot_prediction_grid
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import math
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image
import glob
import time

startTime = time.time() #gets start time so we can see run time of script

#change to wherever images to process are being stored
os.chdir("/Users/kericlamb/Desktop/Documents/Research/Stomata/more_images/")

#Sets up image stuff. Can largely ignore
custom_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(900),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(saturation=0.2),
    transforms.ToTensor(),
    utils.normalize_transform(),
])

#load the model
#change path to where ever you have model actually saved and 'trichome' to whatever you have the label(s) as
#model = core.Model.load('/Users/kericlamb/Desktop/Documents/Research/Stomata/model_weights_epoch_thresh_many25_narrow.pth', ['stomate']) #narrow model
model = core.Model.load('/Users/kericlamb/Desktop/Documents/Research/Stomata/model_weights_epoch_thresh_many25_narrow_fullset.pth', ['stomate']) #wide model

#define function we'll use (modified version of show_labeled_image())
def save_labeled_image(image, boxes, path, name, labels=None):
    
    fig, ax = plt.subplots(1)
    # If the image is already a tensor, convert it back to a PILImage
    # and reverse normalize it
    if isinstance(image, torch.Tensor):
        image = reverse_normalize(image)
        image = transforms.ToPILImage()(image)
    ax.imshow(image)

    # Show a single box or multiple if provided
    if boxes.ndim == 1:
        boxes = boxes.view(1, 4)

    #if labels is not None and not _is_iterable(labels):
    #    labels = [labels]

    # Plot each box
    for i in range(boxes.shape[0]):
        box = boxes[i]
        width, height = (box[2] - box[0]).item(), (box[3] - box[1]).item()
        initial_pos = (box[0].item(), box[1].item())
        rect = patches.Rectangle(initial_pos,  width, height, linewidth=1,
                                 edgecolor='r', facecolor='none')
        if labels:
            ax.text(box[0] + 5, box[1] - 5, '{}'.format(labels[i]), color='red')

        ax.add_patch(rect)

    #plt.show()
    plt.savefig('{0}/{1}.jpg'.format(path, name))

#define relevant paths
im_path = str('/Users/kericlamb/Desktop/Documents/Research/Stomata/more_images/jpegs') #path for images to be processed
pro_path = str('/Users/kericlamb/Desktop/Documents/Research/Stomata/more_images/processed') #path for processed images to be saved and dimension csv
throw_path = str('/Users/kericlamb/Desktop/') #pathway for throwaway csv in calculating box dimensions

#import all images in im_path 
images = glob.glob('{0}/*.jpg'.format(im_path)) #assumes files end in '.jpg' 

#set sensitivity threshold for keeping labels/bounding boxes
thresh = 0.5 #oriented images, thresholded with maximal annotations

#for loop will run through all images in 
for i in range(len(images)):
    time_loop_start = time.time()
    image = utils.read_image('{0}'.format(images[i])) #reads in ith image
    predictions = model.predict(image) #makes model predictions
    labels, boxes, scores = predictions #ascribes labels, boxes, and threshold scores to predictions
    filtered_indices=np.where(scores>thresh) #makes an np index of where score > threshold
    filtered_scores=scores[filtered_indices]
    filtered_boxes=boxes[filtered_indices]
    num_list = filtered_indices[0].tolist() #retain only filtered scores that exceed threshold confidence
    
    name = os.path.basename(images[i]).rstrip('.jpg')+str('_processed') #names processed image off of base image path

    #getting dimensions of each box
    x = [filtered_boxes[n,2] - filtered_boxes[n,0] for n in num_list] #get x-length
    y = [filtered_boxes[n,3] - filtered_boxes[n,1] for n in num_list] #get y-length
    z = [name for n in num_list] #set up third column with the name of images

    xy = {'X-coord':x,'Y-coord':y, 'name':z} #sets up dictionary 
    xy = pd.DataFrame(xy) #convert to df

    #dealing with error in coverting dictionary elements to strings
    xy.to_csv('{0}/xy_throwaway.csv'.format(throw_path)) #throwaway csv path
    xy = pd.read_csv('{0}/xy_throwaway.csv'.format(throw_path))

    #remove exttra bits we don't want for each x- and y-length entry
    xy['X-coord'] = xy['X-coord'].str.replace('tensor', '')
    xy['Y-coord'] = xy['Y-coord'].str.replace('tensor', '')
    xy = xy.replace(to_replace='\(', value="", regex=True)
    xy = xy.replace(to_replace='\)', value="", regex=True)

    #convert to numerics
    xy['X-coord'] = pd.to_numeric(xy['X-coord'])
    xy['Y-coord'] = pd.to_numeric(xy['Y-coord'])

    #calculate hypotenus of box
    xy['C2'] = xy['X-coord']**2 + xy['Y-coord']**2 #gives c^2
    xy['C2'] = pd.to_numeric(xy['C2'])
    xy['C2'] = xy['C2'].apply(lambda x: float(x)) #converts c^2 to a float
    xy['C'] = xy['C2']**(1/2) #takes the square root of c^2

    xy = xy[['X-coord', 'Y-coord', 'C', 'name']]
    xy.to_csv('{0}/xy_final.csv'.format(pro_path), mode = 'a', index= False, header = False)

    #retaining label functionality here in case we want to build in multiple classes of labels
    filtered_labels = [labels[n] for n in num_list] #can get rid of since we don't really care about labels
    
    save_labeled_image(image, filtered_boxes, pro_path, name) #function that actually saves processed image

    #calculating time per image taken
    time_loop_end = time.time()
    image_time = pd.to_numeric(time_loop_end - time_loop_start)
    time_df = pd.DataFrame()
    time_df['time'] = [image_time]
    time_df.to_csv('{0}/time.csv'.format(pro_path), mode = 'a', index= False, header = False)

executionTime = (time.time() - startTime) #calculate run time
print("Script finished. Run in: %s minutes" % (executionTime/60) )
    