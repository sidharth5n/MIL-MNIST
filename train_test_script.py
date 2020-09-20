import torch
from torch import optim
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score

def calculate_classification_error(Y, Y_hat):
    """
    Calculates the classification error.

    Parameters
    ----------
    Y : GT labels
    Y_hat : Predicted labels

    Returns
    -------
    error : classification error
    Y_hat : predicted labels
    """

    Y = Y.float()
    error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

    return error

def calculate_objective(Y, Y_prob):
    """
    Calculates the loss.

    Parameters
    ----------
    Y : GT labels
    Y_prob : Predicted probabilities

    Returns
    -------
    neg_log_likelihood : The negative log-likelihood
    """

    # Convert labels to float
    Y = Y.float()
    # Clip the predicted probabilities
    Y_prob = torch.clamp(Y_prob, min = 1e-5, max = 1. - 1e-5)
    # Binary cross entropy
    neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))

    return neg_log_likelihood

def train(model, train_loader, loss_fn, optimizer, device, epochs):
    """
    Trains the given model with given loss function and optimizer

    Parameters
    ----------
    model        : Model to be trained
    train_loader : Data loader
    loss_fn      : Loss function
    optimizer    : Optimizer
    device       : cuda or cpu
    epochs       : No. of iterations to be trained
    """
    model.train()
    for epoch in range(epochs):
        train_loss = 0.
        train_error = 0.
        for X, Y in train_loader:
            X, Y = X.to(device), Y.to(device)
            X, Y = Variable(X), Variable(Y)

            # reset gradients
            optimizer.zero_grad()
            # forward pass
            Y_prob, Y_hat = model(X)
            # calculate loss
            loss = loss_fn(Y_prob, Y.float())
            # update total loss
            train_loss += loss.item()
            error = calculate_classification_error(Y, Y_hat)
            # update total error
            train_error += error
            # backward pass
            loss.backward()
            # update the parameters
            optimizer.step()

        # calculate average loss and error
        train_loss /= len(train_loader)
        train_error /= len(train_loader)

        print('Epoch {}/{} : Loss = {:.4f}, Error = {:.4f}'.format(epoch+1, epochs, train_loss, train_error))


def test(model, test_loader, loss_fn, device):
    """
    Tests the model

    Parameters
    ----------
    model       : The model to be tested
    test_loader : Data loader
    loss_fn     : The loss function
    device      : cuda or cpu
    """
    model.eval()
    test_loss = 0.
    test_error = 0.
    for X, Y in test_loader:
        X, Y = X.to(device), Y.to(device)
        X, Y = Variable(X), Variable(Y)
        # forward pass
        Y_prob, Y_hat = model(X)
        # compute loss
        loss = loss_fn(Y_prob, Y.float())
        # update total loss
        test_loss += loss.item()
        # compute classification error
        error = calculate_classification_error(Y, Y_hat)
        # update total error
        test_error += error

    test_error /= len(test_loader)
    test_loss /= len(test_loader)

    print('Loss = {:.4f}, Error = {:.4f}'.format(test_loss, test_error))
