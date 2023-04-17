from math import pi
import random
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import numpy as np
import cv2

from collections import namedtuple

Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'trainId'     , # Feel free to modify these IDs as suitable for your method. Then create
                    # ground truth images with train IDs, using the tools provided in the
                    # 'preparation' folder. However, make sure to validate or submit results
                    # to our evaluation server using the regular IDs above!
                    # For trainIds, multiple labels might have the same ID. Then, these labels
                    # are mapped to the same class in the ground truth images. For the inverse
                    # mapping, we use the label that is defined first in the list below.
                    # For example, mapping all void-type classes to the same ID in training,
                    # might make sense for some approaches.
                    # Max value is 255!

    'category'    , # The name of the category that this label belongs to

    'categoryId'  , # The ID of this category. Used to create ground truth images
                    # on category level.

    'hasInstances', # Whether this label distinguishes between single instances or not

    'ignoreInEval', # Whether pixels having this class as ground truth label are ignored
                    # during evaluations or not

    'color'       , # The color of this label
    ] )


class ClassificationVisualizer():

    def __init__(self, title):
       
        # Initial parameters
        self.handles = {} # dictionary of handles per layer
        self.title = title 
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
        self.tensor_to_pil_image = transforms.Compose([ 
            transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist()),
            transforms.ToPILImage(),
        ])

        self.labels = [
            #       name                     id    trainId   category            catId     hasInstances   ignoreInEval   color
            Label(  'background'            ,  0 ,      0 , 'void'            , 0       , False        , True         , (  0,  0,  0) ),
            Label(  'crosswalk'             ,  1 ,      1 , 'void'            , 0       , False        , True         , (255,255,255) ),
            Label(  'driveable'             ,  2 ,      2 , 'void'            , 0       , False        , True         , (128, 64,128) ),
            Label(  'driveable_alt'         ,  3 ,      3 , 'void'            , 0       , False        , True         , (244, 35,232) ),
            Label(  'parking'               ,  4 ,      4 , 'void'            , 0       , False        , True         , (250,170,160) ),
        ]

        self.mask = np.array([label.color for label in self.labels if label.trainId >= -1 and label.trainId <= 19] ,dtype='uint8')
        self.mask = np.concatenate((self.mask, np.array([[0,0,0]])), axis=0)

    def draw(self, inputs, masks, masks_predicted):

        # Setup figure
        self.figure = plt.figure(self.title)
        plt.axis('off')
        self.figure.canvas.manager.set_window_title(self.title)
        self.figure.set_size_inches(10,7)
        plt.suptitle(self.title)

        inputs = inputs
        batch_size,_,_,_ = list(inputs.shape)

        if batch_size < 25:
            random_idxs = random.sample(list(range(batch_size)), k=batch_size)
        else:
            random_idxs = random.sample(list(range(batch_size)), k=5*5)
        plt.clf()
        
        for plot_idx, image_idx in enumerate(random_idxs, start=1):
            image_t = inputs[image_idx,:,:,:].detach().cpu()
            image_pil = self.tensor_to_pil_image(image_t)
            mask_predicted_pil = masks_predicted[image_idx,:,:,:]
            mask_predicted_pil = mask_predicted_pil.argmax(0)
            mask_predicted_pil = mask_predicted_pil.byte().cpu().numpy()
            mask= masks[image_idx].detach().cpu()
            mask = mask.numpy()
            ax = self.figure.add_subplot(5,15,plot_idx) # define a 5 x 5 subplot matrix
            plt.imshow(np.asarray(image_pil))
            ax = self.figure.add_subplot(5,15,plot_idx+len(random_idxs)) # define a 5 x 5 subplot matrix
            plt.imshow(self.mask[mask_predicted_pil])
            ax = self.figure.add_subplot(5,15,plot_idx+len(random_idxs)+len(random_idxs)) # define a 5 x 5 subplot matrix
            plt.imshow(self.mask[mask])

        plt.draw()
        key = plt.waitforbuttonpress(0.05)
        if not plt.fignum_exists(1):
            print('Terminating')
            exit(0)
            

class DataVisualizer():

    def __init__(self, title):
       
        # Initial parameters
        self.handles = {} # dictionary of handles per layer
        self.title = title
         
        # Setup figure
        self.figure = plt.figure(title)
        self.figure.canvas.manager.set_window_title(title)
        self.figure.set_size_inches(4,3)
        plt.suptitle(title)
        plt.legend(loc='best')
        plt.waitforbuttonpress(0.1)

    def draw(self,xs,ys, layer='default', marker='.', markersize=1, color=[0.5,0.5,0.5], alpha=1, label='', x_label='', y_label=''):

        xs,ys = self.toNP(xs,ys) # make sure we have np arrays
        plt.figure(self.title)


        if not layer in self.handles: # first time drawing this layer
            self.handles[layer] = plt.plot(xs, ys, marker, markersize=markersize, 
                                        color=color, alpha=alpha, label=label)
            plt.legend(loc='best')

        else: # use set to edit plot
            plt.setp(self.handles[layer], data=(xs, ys))  # update lm

        plt.xlabel(x_label)    
        plt.ylabel(y_label)    
        plt.draw()

        key = plt.waitforbuttonpress(0.01)
        if not plt.fignum_exists(1):
            print('Terminating')
            exit(0)

    def toNP(self, xs, ys):
        if torch.is_tensor(xs):
            xs = xs.cpu().detach().numpy()

        if torch.is_tensor(ys):
            ys = ys.cpu().detach().numpy()

        return xs,ys


    def recomputeAxesRanges(self):

        plt.figure(self.title)
        ax = plt.gca()
        ax.relim()
        ax.autoscale_view()
        plt.draw()
