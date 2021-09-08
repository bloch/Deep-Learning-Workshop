import sys
import time
from dataset import *
from random import randrange
from models import *
import torch.optim as optim
import torch.nn as nn


def calculate_validation_loss(model, criterion):
    if isinstance(model, ConvAutoencoder4) or isinstance(model, ConvAutoencoder5) or isinstance(model, VAE):
        dataset = Dataset(0, VALIDATION_SET_DIR, False)
    else:  # blocks
        dataset = Dataset(0, VALIDATION_SET_BLOCKS_DIR + "\\block" + str(model.model_index), True)

    data_loader = torch.utils.data.DataLoader(dataset, **train_loader_params)
    running_loss = 0.0
    for i, data in enumerate(data_loader, 0):
        inputs, labels = data

        if not isinstance(model, VAE):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        else:
            outputs, mu, logvar = model(inputs)
            loss, bce, kld = loss_fn(outputs, inputs, mu, logvar)

        running_loss += loss.item()

    return running_loss / dataset.length

def train(model, training_loader, criterion, optimizer, epochs, N):
    train_start_time = time.time()

    train_loss_path = model.path_dir + "\\train_loss.txt"
    validation_loss_path = model.path_dir + "\\validation_loss.txt"
    if isinstance(model, Vanilla) or isinstance(model, MultiLayer):
        train_loss_path = model.path_dir + "\\train_loss_block" + str(model.model_index) + ".txt"
        validation_loss_path = model.path_dir + "\\validation_loss_block" + str(model.model_index) + ".txt"

    train_loss_file = open(train_loss_path, "w")
    train_loss_file.close()
    validation_loss_file = open(validation_loss_path, "w")
    validation_loss_file.close()

    for epoch in range(1, epochs + 1):
        train_loss_file = open(train_loss_path, "a")
        validation_loss_file = open(validation_loss_path, "a")
        running_loss = 0.0
        epoch_running_loss = 0.0
        for i, data in enumerate(training_loader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            if not isinstance(model, VAE):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            else:
                outputs, mu, logvar = model(inputs)
                loss, bce, kld = loss_fn(outputs, inputs, mu, logvar)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_running_loss += loss.item()
            if (i+1) % 10 == 0:  # print loss every 10 mini-batches
                print('[%d, %8d] loss: %f' % (epoch, i + 1, running_loss / 10))
                running_loss = 0.0

        torch.save(model.state_dict(), model.path)
        train_loss_file.write(str(epoch) + "," + str(epoch_running_loss / N) + "\n")
        train_loss_file.close()
        validation_loss_file.write(str(epoch) + "," + str(calculate_validation_loss(model, criterion)) + "\n")
        validation_loss_file.close()

        # save the model every some epochs to a checkpoints dir.
        if isinstance(model, Vanilla) or isinstance(model, MultiLayer):
            if epoch % 10 == 0:
                torch.save(model.state_dict(), model.checkpoints_path + "\\epoch" + str(epoch) + "_block" + str(model.model_index) + ".pt")
        else:
            if isinstance(model, VAE):
                if epoch % 10 == 0:
                    torch.save(model.state_dict(), model.checkpoints_path + "\\epoch" + str(epoch) + ".pt")
            else:
                torch.save(model.state_dict(), model.checkpoints_path + "\\epoch" + str(epoch) + ".pt")

        print("Finished epoch " + str(epoch) + " , total time so far: " + str((time.time() - train_start_time)/3600) + " hours.")


def init_model(type):
    if type == "vanilla":
        for i in range(NUM_OF_BLOCKS):
            model = Vanilla(np.prod(BLOCK_DIMS), VANILLA_HIDDEN_LAYER_SIZE, i)
            torch.save(model.state_dict(), model.path)
    if type == "multilayer":
        for i in range(NUM_OF_BLOCKS):
            model = MultiLayer(np.prod(BLOCK_DIMS), MULTILAYER_HIDDEN1_SIZE, MULTILAYER_HIDDEN2_SIZE, MULTILAYER_HIDDEN_LAYER_SIZE, i)
            torch.save(model.state_dict(), model.path)
    if type == "conv4":
        model = ConvAutoencoder4()
        torch.save(model.state_dict(), model.path)
    if type == "conv5":
        model = ConvAutoencoder5()
        torch.save(model.state_dict(), model.path)
    if type == "vae":
        model = VAE()
        torch.save(model.state_dict(), model.path)
    print("model is initialized.")


def train_model(model_type, epochs):
    if model_type == "conv4" or model_type == "conv5" or model_type == "vae":
        dataset = Dataset(0, TRAINING_SET_DIR, False)
        train_loader = torch.utils.data.DataLoader(dataset, **train_loader_params)
        criterion = nn.MSELoss()
        if model_type == "conv4":
            model = ConvAutoencoder4()
        if model_type == "conv5":
            model = ConvAutoencoder5()
        if model_type == "vae":
            model = VAE()

        model.load_state_dict(torch.load(model.path))
        optimizer = optim.Adam(model.parameters())
        train(model, train_loader, criterion, optimizer, epochs, dataset.length)
    if model_type == "vanilla" or model_type == "multilayer":
        datasets = [Dataset(0, TRAINING_SET_BLOCKS_DIR + "\\block" + str(i), True) for i in range(NUM_OF_BLOCKS)]
        train_loaders = [torch.utils.data.DataLoader(datasets[i], **train_loader_params) for i in range(NUM_OF_BLOCKS)]
        criterion = nn.MSELoss()
        for i in range(NUM_OF_BLOCKS):      # train each block
            if model_type == "vanilla":
                model = Vanilla(np.prod(BLOCK_DIMS), VANILLA_HIDDEN_LAYER_SIZE, i)
            elif model_type == "multilayer":
                model = MultiLayer(np.prod(BLOCK_DIMS), MULTILAYER_HIDDEN1_SIZE, MULTILAYER_HIDDEN2_SIZE, MULTILAYER_HIDDEN_LAYER_SIZE, i)
            model.load_state_dict(torch.load(model.path))
            optimizer = optim.Adam(model.parameters())
            train(model, train_loaders[i], criterion, optimizer, epochs, datasets[0].length)


if len(sys.argv) == 4:
    model_name, num_of_epochs, to_init = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    if to_init == 1:
        init_model(model_name)
    train_model(model_name, num_of_epochs)
