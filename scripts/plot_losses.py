import matplotlib.pyplot as plt
import numpy as np
from models import *
from params import *

# This script creates loss plot as a function of epoch num(both train + validation)


def get_train_loss(model, i):
    if isinstance(model, Vanilla) or isinstance(model, MultiLayer):
        train_epoch_losses_file = open(model.path_dir + "\\train_loss_block" + str(i) + ".txt", 'r')
    else:
        train_epoch_losses_file = open(model.path_dir + "\\train_loss.txt", 'r')
    lines = train_epoch_losses_file.readlines()
    epochs = []
    losses = []
    for line in lines:
        epochs.append(float(line.split(',')[0]))
        losses.append(float(line.split(',')[1]))

    return epochs, losses


def get_validation_loss(model, i):
    if isinstance(model, Vanilla) or isinstance(model, MultiLayer):
        validation_epoch_losses_file = open(model.path_dir + "\\validation_loss_block" + str(i) + ".txt" ,'r')
    else:
        validation_epoch_losses_file = open(model.path_dir + "\\validation_loss.txt", 'r')
    lines = validation_epoch_losses_file.readlines()
    epochs = []
    losses = []
    for line in lines:
        epochs.append(float(line.split(',')[0]))
        losses.append(float(line.split(',')[1]))

    return epochs, losses

def plot_losses(model):
    epochs, train_losses = get_train_loss(model, None)
    _ , validation_losses = get_validation_loss(model, None)
    plt.plot(epochs, train_losses, label="train")
    plt.plot(epochs, validation_losses, label="validation")
    plt.xlabel("epochs")
    plt.ylabel("avg loss")
    plt.title("MSE loss as function of number of epochs")
    plt.legend()
    plt.show()

def plot_losses_for_blocks_models(model):
    train_losses = [[] for _ in range(NUM_OF_BLOCKS)]
    validation_losses = [[] for _ in range(NUM_OF_BLOCKS)]
    for i in range(NUM_OF_BLOCKS):
        epochs, train_losses[i] = get_train_loss(model,i)
        _, validation_losses[i] = get_validation_loss(model,i)

    train_losses_merged = [0.0 for _ in range(len(epochs))]
    validation_losses_merged = [0.0 for _ in range(len(epochs))]

    for i in range(len(epochs)):
        tmp_train_loss = 0.0
        tmp_validation_loss = 0.0
        for j in range(NUM_OF_BLOCKS):
            tmp_train_loss += train_losses[j][i]
            tmp_validation_loss += validation_losses[j][i]
        train_losses_merged[i] = tmp_train_loss
        validation_losses_merged[i] = tmp_validation_loss

    plt.plot(epochs, train_losses_merged, label="train")
    plt.plot(epochs, validation_losses_merged, label="validation")
    plt.xlabel("epochs")
    plt.ylabel("avg loss")
    plt.title("MSE loss as function of number of epochs")
    plt.legend()
    plt.show()

# The way to use this script:
# model = VAE()
# plot_losses(model)
