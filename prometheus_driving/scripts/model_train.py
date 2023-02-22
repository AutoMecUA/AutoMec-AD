#!/usr/bin/python3

# Imports 
import argparse
import os
from statistics import mean
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from colorama import Fore, Style
import torch
from torch.nn import MSELoss
import yaml

# Custom imports
from src.dataset import Dataset
from models.cnn_nvidia import Nvidia_Model
from models.cnn_rota import Rota_Model
from models.mobilenetv2 import MobileNetV2
from models.inceptionV3 import InceptionV3
from models.vgg import MyVGG
from models.resnet import ResNet
from models.lstm import LSTM
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
    parser.add_argument('-d', '--dataset_name', type=str, required=True,
                        help='folder name of the dataset')
    parser.add_argument('-fn', '--folder_name', type=str, required=True,
                        help='folder name where the model is stored')
    parser.add_argument('-n_epochs', '--max_epoch', default=50, type=int,
                        help='Maximum number of epochs')
    parser.add_argument('-batch_size', '--batch_size', default=256, type=int,
                        help='Batch size')
    parser.add_argument('-c', '--cuda', default=0, type=int,
                        help='Number of cuda device')
    parser.add_argument('-loss_t', '--loss_threshold', default=0.001, type=float,
                        help='Loss threshold criteria for when to stop')
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float,
                        help='Learning rate')
    parser.add_argument('-lr_step_size', '--lr_step_size', type=int, default=20,
                        help='Step size of the learning rate decay')
    parser.add_argument('-lr_gamma', '--lr_gamma', type=float, default=0.5,
                        help='Decay of the learning rate after step size')
    parser.add_argument('-wd', '--weight_decay', type=float, default=0, help='L2 regularizer')
    parser.add_argument('-nw', '--num_workers', type=int, default=4, 
                        help='How many subprocesses to use for data loading. 0 means that the data will be loaded in the main process.')
    parser.add_argument('-pff', '--prefetch_factor', type=int, default=2, 
                        help='Number of batches loaded in advance by each worker')
    parser.add_argument('-m', '--model', default='Nvidia_Model()', type=str,
                        help='Model to use [Nvidia_Model(), Rota_Model(), MobileNetV2(), InceptionV3(), MyVGG(), ResNet()]')
    parser.add_argument('-loss_f', '--loss_function', type=str, default='MSELoss()',
                        help='Type of loss function. [MSELoss()]')

    arglist = [x for x in sys.argv[1:] if not x.startswith('__')]
    args = vars(parser.parse_args(args=arglist))

    # General Path
    files_path=os.environ.get('AUTOMEC_DATASETS')
    # Image dataset paths
    dataset_path = f'{files_path}/datasets/{args["dataset_name"]}/'
    columns = ['img_name','steering', 'velocity'] 
    df = pd.read_csv(os.path.join(dataset_path, 'driving_log.csv'), names = columns)

    del df["velocity"] # not in use, currently
    df.head()

    # Read YAML file
    with open(dataset_path + "info.yaml", 'r') as stream:
        data_loaded = yaml.safe_load(stream)

    print(f'{Fore.BLUE}The dataset has {len(df)} images{Style.RESET_ALL}')

    model_name = args["folder_name"]
    model_path = files_path + f'/models/{args["folder_name"]}/{args["folder_name"]}.pkl'
    folder_path =files_path + f'/models/{args["folder_name"]}'
    # Checks if the models folder exists if not create
    if not os.path.exists(f'{files_path}/models'):
        os.makedirs(f'{files_path}/models') # Creates the folder

    device = f'cuda:{args["cuda"]}' if torch.cuda.is_available() else 'cpu' # cuda: 0 index of gpu
    model = eval(args['model']) # Instantiate model
    print(f'You are training using the device: ' + Fore.YELLOW + f'{device}' + Style.RESET_ALL)


    # Define hyper parameters
    learning_rate = args['learning_rate']
    maximum_num_epochs = args['max_epoch'] 
    termination_loss_threshold =  args['loss_threshold']
    loss_function = eval(args['loss_function']) # Instantiate loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate , weight_decay=args['weight_decay'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args['lr_step_size'], gamma=args['lr_gamma'])

    ########################################
    # Dataset                              #
    ########################################
    if args['model'] == 'LSTM()':
        shuffle = False
    else:
        shuffle = True
    # Sample ony a few images for development
    #df = df[0:700]
    train_dataset,test_dataset = train_test_split(df,test_size=0.2)
    # Creates the train dataset
    dataset_train = Dataset(train_dataset,dataset_path)
    # Creates the batch size that suits the amount of memory the graphics can handle
    loader_train = torch.utils.data.DataLoader(dataset=dataset_train,batch_size=args['batch_size'],shuffle=shuffle , num_workers=args['num_workers'] , prefetch_factor=args['prefetch_factor'])
    # Creates the test dataset
    dataset_test = Dataset(test_dataset , dataset_path , augmentation=False)
    # Creates the batch size that suits the amount of memory the graphics can handle
    loader_test = torch.utils.data.DataLoader(dataset=dataset_test, batch_size=args['batch_size'], shuffle=shuffle , num_workers=args['num_workers'] , prefetch_factor=args['prefetch_factor'])

    ########################################
    # Training                             #
    ########################################
    # Init visualization of loss
    if args['visualize']: # Checks if the user wants to visualize the loss
        loss_visualizer = DataVisualizer('Loss')
        loss_visualizer.draw([0,maximum_num_epochs], [termination_loss_threshold, termination_loss_threshold], layer='threshold', marker='--', markersize=1, color=[0.5,0.5,0.5], alpha=1, label='threshold', x_label='Epochs', y_label='Loss')
        test_visualizer = ClassificationVisualizer('Test Images')
    # Resume training
    ans = ''
    if os.path.exists(folder_path): # Checks to see if the model exists
        print(Fore.YELLOW + f'Model already exists! Do you want to resume training?' + Style.RESET_ALL)
        ans = input(Fore.YELLOW + "Y" + Style.RESET_ALL + "ES/" + Fore.YELLOW + "n" + Style.RESET_ALL + "o/"+ Fore.YELLOW + "o" + Style.RESET_ALL + "verwrite: ") # Asks the user if they want to resume training
        if ans.lower() in ['', 'yes','y']: # If the user wants to resume training
            try:
                checkpoint = torch.load(model_path)
            except FileNotFoundError:
                print(Fore.YELLOW + 'Folder empty' + Style.RESET_ALL)
                ans = 'o'
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(device) # move the model variable to the gpu if one exists
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                loader_train = checkpoint['loader_train']
                loader_test = checkpoint['loader_test']
                idx_epoch = checkpoint['epoch'] + 1
                epoch_train_losses = checkpoint['train_losses']
                stored_train_loss=epoch_train_losses[-1]
                epoch_test_losses = checkpoint['test_losses']
        if ans.lower() in ['overwrite', 'o']:
            print(Fore.YELLOW + 'Overwriting.' + Style.RESET_ALL)
            os.system('rm -rf ' + folder_path)
        if ans.lower() not in ['', 'yes', 'y', 'overwrite', 'o']: # If the user does not want to resume training
            print(f'{Fore.RED} Terminating training... {Fore.RESET}')
            exit(0)
    # If the model does not exist or the user wants to overwrite
    if not os.path.exists(folder_path) or ans.lower() in ['overwrite', 'o']: # If the model does not exist
        print(Fore.YELLOW + f'Starting from scratch.' + Style.RESET_ALL)
        os.makedirs(folder_path)
        idx_epoch = 0
        epoch_train_losses = []
        epoch_test_losses = []
        stored_train_loss=1e2
        model.to(device) # move the model variable to the gpu if one exists
    # Training loop for each epoch
    while True:
        # Train batch by batch
        train_losses = []
        for batch_idx, (image_t, label_t) in tqdm(enumerate(loader_train), total=len(loader_train), desc=Fore.GREEN + 'Training batches for Epoch ' + str(idx_epoch) +  Style.RESET_ALL):
            # Move the data to the GPU if one exists
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
            # Store the loss for the batch
            train_losses.append(loss.data.item())

        # Compute the loss for the epoch
        epoch_train_loss = mean(train_losses)
        epoch_train_losses.append(epoch_train_loss)

        # Run test in batches 
        test_losses = []
        for batch_idx, (image_t, label_t) in tqdm(enumerate(loader_test), total=len(loader_test), desc=Fore.GREEN + 'Testing batches for Epoch ' + str(idx_epoch) +  Style.RESET_ALL):
            # Move the data to the gpu if one exists
            image_t = image_t.to(device=device, dtype=torch.float)
            label_t = label_t.to(device=device, dtype=torch.float).unsqueeze(1)

            # Apply the network to get the predicted ys
            label_t_predicted = model.forward(image_t)
            # Compute the error based on the predictions
            loss = loss_function(label_t_predicted, label_t)
            # Store the loss for the batch
            test_losses.append(loss.data.item())
            # Visualize the test images with the labeled data and predicted data
            if args['visualize']:
                test_visualizer.draw(image_t, label_t, label_t_predicted)


        # Compute the loss for the epoch
        epoch_test_loss = mean(test_losses)
        epoch_test_losses.append(epoch_test_loss)

        # Steps the learning rate scheduler
        scheduler.step()


        ########################################
        # Visualization                        #
        ########################################
        if args['visualize']:
            loss_visualizer.draw(list(range(0, len(epoch_train_losses))), epoch_train_losses, layer='train loss', marker='-', markersize=1, color=[0,0,0.7], alpha=1, label='Train Loss', x_label='Epochs', y_label='Loss')

            loss_visualizer.draw(list(range(0, len(epoch_test_losses))), epoch_test_losses, layer='test loss', marker='-', markersize=1, color=[1,0,0.7], alpha=1, label='Test Loss', x_label='Epochs', y_label='Loss')

            loss_visualizer.recomputeAxesRanges()

        print(f'{Fore.LIGHTBLUE_EX}Epoch {str(idx_epoch)} Train Loss: {str(epoch_train_loss)} Test Loss: {str(epoch_test_loss)} With learning rate: {str(scheduler.get_lr())} {Style.RESET_ALL}')

        ########################################
        # Termination criteria                 #
        ########################################
        if idx_epoch >= maximum_num_epochs:
            print(Fore.CYAN + 'Finished training. Reached maximum number of epochs. Comparing to previously stored model' + Style.RESET_ALL)
            if epoch_train_loss < stored_train_loss:
                print(Fore.BLUE + 'Saving model at Epoch ' + str(idx_epoch) + ' Loss ' + str(epoch_train_loss) + Style.RESET_ALL)
                SaveModel(model,idx_epoch,optimizer,loader_train,loader_test,epoch_train_losses,epoch_test_losses,folder_path,device,model_name,args['model'],idx_epoch,args['batch_size'],train_losses[-1],test_losses[-1],args['loss_function'],data_loaded) # Saves the model
                SaveGraph(epoch_train_losses,epoch_test_losses,folder_path)
            else:
                print(Fore.BLUE + 'Not saved, current loos '+ str(epoch_train_loss) + '. Previous model is better, previous loss ' + str(stored_train_loss) + '.' + Style.RESET_ALL)
            break
        elif epoch_train_loss <= termination_loss_threshold:
            print(Fore.CYAN + 'Finished training. Reached target loss. Comparing to previously stored model' + Style.RESET_ALL)
            if epoch_train_loss < stored_train_loss:
                print(Fore.BLUE + 'Saving model at Epoch ' + str(idx_epoch) + ' Loss ' + str(epoch_train_loss) + Style.RESET_ALL)
                SaveModel(model,idx_epoch,optimizer,loader_train,loader_test,epoch_train_losses,epoch_test_losses,folder_path,device,model_name,args['model'],idx_epoch,args['batch_size'],train_losses[-1],test_losses[-1],args['loss_function'],data_loaded) # Saves the model
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
                SaveModel(model,idx_epoch,optimizer,loader_train,loader_test,epoch_train_losses,epoch_test_losses,folder_path,device,model_name,args['model'],idx_epoch,args['batch_size'],train_losses[-1],test_losses[-1],args['loss_function'],data_loaded) # Saves the model
                SaveGraph(epoch_train_losses,epoch_test_losses,folder_path)
                stored_train_loss=epoch_train_loss
                
            else: 
                print(Fore.BLUE + 'Not saved, current loss '+ str(epoch_train_loss) + '. Previous model is better, previous loss ' + str(stored_train_loss) + '.' + Style.RESET_ALL)


        idx_epoch += 1 # go to next epoch

if __name__ == '__main__':
    main()