#!/usr/bin/python3

# Imports 
import argparse
import os
import glob
import random
from statistics import mean
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from colorama import Fore, Style
import torch

#  custom imports
from src.dataset import Dataset
from models.cnn_nvidia import Model
from src.utils import SaveModel, SaveGraph
from src.visualization import DataVisualizer, ClassificationVisualizer

# Main code
def main():
    ########################################
    # Initialization                       #
    ########################################
    parser = argparse.ArgumentParser(description='Data Collector')
    parser.add_argument('-v', '--visualize', action='store_true',
                        help='Visualize the loss')
    parser.add_argument('-fn', '--folder_name', type=str, required=True, help='folder name')
    parser.add_argument('-mn', '--model_name', type=str, required=True, help='model name')
    parser.add_argument('-n_epochs', '--max_epoch', default=50, type=int,
                        help='Maximum number of epochs')
    parser.add_argument('-batch_size', '--batch_size', default=256, type=int,
                        help='Batch size')
    parser.add_argument('-c', '--cuda', default=0, type=int,
                        help='Number of cuda device')
    parser.add_argument('-loss', '--loss_threshold', default=0.01, type=float,
                        help='Loss threshold criteria for when to stop')
    parser.add_argument('-lr', '--learning_rate', default=0.01, type=float,
                        help='Learning rate')

    arglist = [x for x in sys.argv[1:] if not x.startswith('__')]
    args = vars(parser.parse_args(args=arglist))

    # General Path
    files_path=f'/home/andre/catkin_ws/src/AutoMec-AD/prometheus_driving/data/'
    # Image dataset paths
    dataset_path = files_path + 'set10/'
    columns = ['img_name','steering', 'velocity'] 
    df = pd.read_csv(os.path.join(dataset_path, 'driving_log.csv'), names = columns)

    del df["velocity"] # not in use, currently
    df.head()

    print(f'{Fore.BLUE}The dataset has {len(df)} images{Style.RESET_ALL}')

    model_path = files_path + f'/models/{args["folder_name"]}/{args["model_name"]}.pkl'
    folder_path =files_path + f'/models/{args["folder_name"]}'
    # Checks if the models folder exists if not create
    if not os.path.exists(f'{files_path}/models'):
        os.makedirs(f'{files_path}/models') # Creates the folder

    device = f'cuda:{args["cuda"]}' if torch.cuda.is_available() else 'cpu' # cuda: 0 index of gpu

    model = Model() # Instantiate model

    # Define hyper parameters
    learning_rate = args['learning_rate']
    maximum_num_epochs = args['max_epoch'] 
    termination_loss_threshold =  args['loss_threshold']
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    ########################################
    # Dataset                              #
    ########################################

    # Sample ony a few images for develop
    #image_filenames = random.sample(image_filenames,k=700)
    train_dataset,test_dataset = train_test_split(df,test_size=0.2)
    # Creates the train dataset
    dataset_train = Dataset(train_dataset)
    # Creates the batch size that suits the amount of memory the graphics can handle
    loader_train = torch.utils.data.DataLoader(dataset=dataset_train,batch_size=args['batch_size'],shuffle=True)
    # Goes through al the images and displays them

    dataset_test = Dataset(test_dataset)
    loader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=args['batch_size'], shuffle=True)

    ########################################
    # Training                             #
    ########################################
    # Init visualization of loss
    if args['visualize']: # Checks if the user wants to visualize the loss
        loss_visualizer = DataVisualizer('Loss')
        loss_visualizer.draw([0,maximum_num_epochs], [termination_loss_threshold, termination_loss_threshold], layer='threshold', marker='--', markersize=1, color=[0.5,0.5,0.5], alpha=1, label='threshold', x_label='Epochs', y_label='Loss')
        test_visualizer = ClassificationVisualizer('Test Images')
    # Resume training
    if os.path.exists(folder_path): # Checks to see if the model exists
        print(Fore.YELLOW + f'Folder already exists! Do you want to resume training?' + Style.RESET_ALL)
        ans = input("YES/no")
        if ans.lower() in ['', 'yes','y']:
            checkpoint = torch.load(model_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device) # move the model variable to the gpu if one exists
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            loader_train = checkpoint['loader_train']
            loader_test = checkpoint['loader_test']
            idx_epoch = checkpoint['epoch']
            epoch_train_losses = checkpoint['train_losses']
            stored_train_loss=epoch_train_losses[-1]
            epoch_test_losses = checkpoint['test_losses']
        else:
            print(f'{Fore.RED} Terminating training... {Fore.RESET}')
            exit(0)
    else:
        print(Fore.YELLOW + f'Model Folder not found: {args["folder_name"]}. Starting from sratch.' + Style.RESET_ALL)
        os.makedirs(folder_path)
        idx_epoch = 0
        epoch_train_losses = []
        epoch_test_losses = []
        stored_train_loss=1e2
        model.to(device) # move the model variable to the gpu if one exists
    # -----------

    while True:
        # Train batch by batch -----------------------------------------------
        train_losses = []
        for batch_idx, (image_t, label_t) in tqdm(enumerate(loader_train), total=len(loader_train), desc=Fore.GREEN + 'Training batches for Epoch ' + str(idx_epoch) +  Style.RESET_ALL):

            image_t = image_t.to(device=device, dtype=torch.float)
            label_t = label_t.to(device=device, dtype=torch.float).unsqueeze(1)

            # Apply the network to get the predicted ys
            label_t_predicted = model.forward(image_t)
 
            # Compute the error based on the predictions
            loss = loss_function(label_t_predicted, label_t)

            # Update the model, i.e. the neural network's weights 
            optimizer.zero_grad() # resets the weights to make sure we are not accumulating
            loss.backward() # propagates the loss error into each neuron
            optimizer.step() # update the weights


            train_losses.append(loss.data.item())

        # Compute the loss for the epoch
        epoch_train_loss = mean(train_losses)
        epoch_train_losses.append(epoch_train_loss)

        # Run test in batches ---------------------------------------
        # TODO dropout
        test_losses = []
        for batch_idx, (image_t, label_t) in tqdm(enumerate(loader_test), total=len(loader_test), desc=Fore.GREEN + 'Testing batches for Epoch ' + str(idx_epoch) +  Style.RESET_ALL):

            image_t = image_t.to(device=device, dtype=torch.float)
            label_t = label_t.to(device=device, dtype=torch.float).unsqueeze(1)

            # Apply the network to get the predicted ys
            label_t_predicted = model.forward(image_t)
            # Compute the error based on the predictions
            loss = loss_function(label_t_predicted, label_t)

            test_losses.append(loss.data.item())

            if args['visualize']:
                test_visualizer.draw(image_t, label_t, label_t_predicted)


        # Compute the loss for the epoch
        epoch_test_loss = mean(test_losses)
        epoch_test_losses.append(epoch_test_loss)


        ########################################
        # Visualization                        #
        ########################################
        if args['visualize']:
            loss_visualizer.draw(list(range(0, len(epoch_train_losses))), epoch_train_losses, layer='train loss', marker='-', markersize=1, color=[0,0,0.7], alpha=1, label='Train Loss', x_label='Epochs', y_label='Loss')

            loss_visualizer.draw(list(range(0, len(epoch_test_losses))), epoch_test_losses, layer='test loss', marker='-', markersize=1, color=[1,0,0.7], alpha=1, label='Test Loss', x_label='Epochs', y_label='Loss')

            loss_visualizer.recomputeAxesRanges()

        ########################################
        # Termination criteria                 #
        ########################################
        if idx_epoch >= maximum_num_epochs:
            print(Fore.CYAN + 'Finished training. Reached maximum number of epochs. Comparing to previously stored model' + Style.RESET_ALL)
            if epoch_train_loss < stored_train_loss:
                print(Fore.BLUE + 'Saving model at Epoch ' + str(idx_epoch) + ' Loss ' + str(epoch_train_loss) + Style.RESET_ALL)
                SaveModel(model,idx_epoch,optimizer,loader_train,loader_test,epoch_train_losses,epoch_test_losses,model_path,device) # Saves the model
                SaveGraph(epoch_train_losses,epoch_test_losses,folder_path)
            else:
                print(Fore.BLUE + 'Not saved, current loos '+ str(epoch_train_loss) + '. Previous model is better, previous loss ' + str(stored_train_loss) + '.' + Style.RESET_ALL)
            break
        elif epoch_train_loss <= termination_loss_threshold:
            print(Fore.CYAN + 'Finished training. Reached target loss. Comparing to previously stored model' + Style.RESET_ALL)
            if epoch_train_loss < stored_train_loss:
                print(Fore.BLUE + 'Saving model at Epoch ' + str(idx_epoch) + ' Loss ' + str(epoch_train_loss) + Style.RESET_ALL)
                SaveModel(model,idx_epoch,optimizer,loader_train,loader_test,epoch_train_losses,epoch_test_losses,model_path,device) # Saves the model
                SaveGraph(epoch_train_losses,epoch_test_losses,folder_path)
            else:
                print(Fore.BLUE + 'Not saved, current loos '+ str(epoch_train_loss) + '. Previous model is better, previous loss ' + str(stored_train_loss) + '.' + Style.RESET_ALL)
            break

        ########################################
        # Checkpoint                           #
        ########################################
        if idx_epoch%10==0:
            print(Fore.CYAN + 'Verifying if the new model is better than the previous one stored' + Style.RESET_ALL)
            if epoch_train_loss < stored_train_loss: # checks if the previous model is better than the new one
                print(Fore.BLUE + 'Saving model at Epoch ' + str(idx_epoch) + ' Loss ' + str(epoch_train_loss) + Style.RESET_ALL)
                # Save checkpoint
                SaveModel(model,idx_epoch,optimizer,loader_train,loader_test,epoch_train_losses,epoch_test_losses,model_path,device) # Saves the model
                SaveGraph(epoch_train_losses,epoch_test_losses,folder_path)
                stored_train_loss=epoch_train_loss
                
            else: 
                print(Fore.BLUE + 'Not saved, current loos '+ str(epoch_train_loss) + '. Previous model is better, previous loss ' + str(stored_train_loss) + '.' + Style.RESET_ALL)


        idx_epoch += 1 # go to next epoch

if __name__ == '__main__':
    main()