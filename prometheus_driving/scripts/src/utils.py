import torch
import matplotlib.pyplot as plt
import os
import glob


def SaveModel(model,idx_epoch,optimizer,training_loader,testing_loader,epoch_train_losses,epoch_test_losses,model_path,device):
    model.to('cpu')
    torch.save({
        'epoch': idx_epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loader_train': training_loader,
        'loader_test': testing_loader,
        'train_losses': epoch_train_losses,
        'test_losses': epoch_test_losses,
        }, model_path)
    model.to(device)

def SaveGraph(train_losses,test_losses,folder_name):
    plt.figure()
    plt.plot(train_losses, label='train loss')
    plt.plot(test_losses, label='test loss')
    plt.xlabel("Epoch")    
    plt.ylabel("Loss")   
    plt.legend()
    plt.savefig(f'{folder_name}/losses.png')