#pipeline to train the model 

from turtle import st
import torch
from detecto import core, utils, visualize
from detecto.visualize import show_labeled_image, plot_prediction_grid
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import math
import time

startTime = time.time() #gets start time so we can see run time of script
print(startTime)

#change to directory above Test and Train directories
#the model and loss graph will also be saved to this location
os.chdir("/Users/kericlamb/Desktop/Documents/Research/Stomata/")

#image formatting-- ignore
custom_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(900),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(saturation=0.2),
    transforms.ToTensor(),
    utils.normalize_transform(),
])

#need to divide labeled images to train model in two sets: Train and Test
#make sure both are decently large (>200 images)
Train_dataset = core.Dataset('Train/',transform=custom_transforms) #path to train dataset
Test_dataset = core.Dataset('Test/') #path to test dataset
loader = core.DataLoader(Train_dataset, batch_size=2, shuffle=True) #batch size of 2 will make it seem like you have half as many images as you do when running
model = core.Model(['stomate']) #name and write model with labels we want to detect

#trains model on train set then attempts to validate with test set
epoch = 25 #sets number of epochs to run. check loss figure to figure out sweet spot between detection and wasting time
losses = model.fit(loader, Test_dataset, epochs=epoch, lr_step_size=5, learning_rate=0.001, verbose=True) #adjust epochs up to increase number of iterations of model learning 

#loss plot
plt.plot(losses) #might cause some issues on rivanna
plt.savefig('loss_fig_{0}_narrow.jpg'.format(epoch)) 

#save the model
model.save('model_weights_epoch_thresh_many{0}_narrow_fullset.pth'.format(epoch))
#model = core.Model.load('model_weights_epoch_thresh{0}.pth'.format(epoch), ['trichome'])

#helps us tell run time of script
executionTime = (time.time() - startTime)
print("Script finished: run in %s hours" % ((executionTime/60)/60))
