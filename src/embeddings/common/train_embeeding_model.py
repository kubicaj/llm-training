import numpy


def train_the_model(model, device, dataloader, optimizer, loss_function, num_epochs=25):
    """
    Train the model for a set number of epochs

    Args:
        model: the model to train
        device: the device to use (cpu or cuda)
        dataloader: the dataloader to use for training
        optimizer: the optimizer to use
        loss_function: the loss function to use
        num_epochs: the number of epochs to train for

    Return:
        model: the trained model
        total_loss: the loss at each epoch
    """
    # initialize losses
    total_loss = numpy.zeros(num_epochs)

    for epoch in range(num_epochs):

        # initialize
        epoch_loss = 0

        # loop over batches in the data loader
        for X, y in dataloader:
            # move data to GPU
            X, y = X.to(device), y.to(device)

            # clear previous gradients
            model.zero_grad()

            # forward pass
            log_probs = model(X)

            # calculate the losses from the final target word
            loss = loss_function(log_probs, y[:, -1])

            # backprop
            loss.backward()
            optimizer.step()

            # sum the per-epoch losses
            epoch_loss += loss.item()

        # scale by the number of tokens in this dataloader
        total_loss[epoch] = epoch_loss / len(dataloader.dataset)

        # update our progress :)
        print(f'  Finished epoch {epoch + 1} with loss {epoch_loss / len(dataloader.dataset):.4f}')

    # output the model and the losses
    return model, total_loss
