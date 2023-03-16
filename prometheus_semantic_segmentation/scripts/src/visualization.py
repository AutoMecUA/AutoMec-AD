from math import pi
import random
import matplotlib.pyplot as plt
from torchvision import transforms
import torch
import numpy as np
import cv2

class ClassificationVisualizer():

    def __init__(self, title):
       
        # Initial parameters
        self.handles = {} # dictionary of handles per layer
        self.title = title
        self.tensor_to_pil_image = transforms.ToPILImage()

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
            image_t = inputs[image_idx,:,:,:]
            mask_predicted_t = masks_predicted[image_idx,:,:,:]
            image_pil = self.tensor_to_pil_image(image_t)
            mask_predicted_pil = self.tensor_to_pil_image(mask_predicted_t)

            ax = self.figure.add_subplot(5,10,plot_idx) # define a 5 x 5 subplot matrix
            plt.imshow(cv2.cvtColor(np.asarray(image_pil), cv2.COLOR_BGR2RGB))
            ax = self.figure.add_subplot(5,10,plot_idx+len(random_idxs)) # define a 5 x 5 subplot matrix
            plt.imshow(cv2.cvtColor(np.asarray(mask_predicted_pil), cv2.COLOR_GRAY2BGR))
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.xaxis.set_ticks([])
            ax.yaxis.set_ticks([])

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
