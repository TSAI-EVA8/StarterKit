import torch
import torch
import torch.nn.functional as F
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchsummary import summary
from torch_lr_finder import LRFinder



def eval(model, loader, device, criterion, losses, accuracies, correct_samples, incorrect_samples, sample_count=25, last_epoch=False):
    """Test the model.
    Args:
        model: Model instance.
        loader: Validation data loader.
        device: Device where the data will be loaded.
        criterion: Loss function.
        losses: List containing the change in loss.
        accuracies: List containing the change in accuracy.
        correct_samples: List containing correctly predicted samples.
        incorrect_samples: List containing incorrectly predicted samples.
        sample_count: Total number of predictions to store from each correct
            and incorrect samples.
    """

    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            img_batch = data  # This is done to keep data in CPU
            data, target = data.to(device), target.to(device)  # Get samples
            output = model(data)  # Get trained model output
            val_loss += criterion(output, target).item()  # Sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # Get the index of the max log-probability
            result = pred.eq(target.view_as(pred))

            # Save correct and incorrect samples
            if last_epoch:
                for i in range(len(list(result))):
                    if not list(result)[i] and len(incorrect_samples) < sample_count:
                        incorrect_samples.append({
                            'prediction': list(pred)[i],
                            'label': list(target.view_as(pred))[i],
                            'image': img_batch[i]
                        })
                    elif list(result)[i] and len(correct_samples) < sample_count:
                        correct_samples.append({
                            'prediction': list(pred)[i],
                            'label': list(target.view_as(pred))[i],
                            'image': img_batch[i]
                        })

            correct += result.sum().item()

    val_loss /= len(loader.dataset)
    losses.append(val_loss)
    accuracies.append(100. * correct / len(loader.dataset))

    print(
        f'Test set: Average loss: {val_loss:.4f}, Accuracy: {correct}/{len(loader.dataset)} ({accuracies[-1]:.2f}%)\n'
    )



def train(model, loader, device, optimizer, losses,accuracies,criterion, l1_factor=0.0):
    """Train the model.
    Args:
        model: Model instance.
        device: Device where the data will be loaded.
        loader: Training data loader.
        optimizer: Optimizer for the model.
        criterion: Loss Function.
        l1_factor: L1 regularization factor.
    """
    train_loss = 0
    model.train()
    pbar = tqdm(loader)
    correct = 0
    processed = 0
    for batch_idx, (data, target) in enumerate(pbar, 0):
        # Get samples
        data, target = data.to(device), target.to(device) # move to the GPU

        # Set gradients to zero before starting backpropagation
        optimizer.zero_grad()

        # Predict output
        y_pred = model(data)

        # Calculate loss
        #loss = l1(model, criterion(y_pred, target), l1_factor)
        loss = F.nll_loss(y_pred, target)
        # Perform backpropagation
        loss.backward()
        optimizer.step()

        # Update Progress Bar
        pred = y_pred.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)
        pbar.set_description(
            desc=f'Loss={loss.item():0.2f} Batch_ID={batch_idx} Train_Accuracy={(100 * correct / processed):.2f}'
        )
        train_loss+=loss.item()
    
    train_loss /= len(loader.dataset)
        
    losses.append(train_loss)
    accuracies.append((100 * correct / processed))


def l1(model, loss, Lambda):
    """Apply L1 regularization.
    Args:
        model: Model instance.
        loss: Loss function value.
        factor: Factor for applying L1 regularization
    
    Returns:
        Regularized loss value.
    """

    if Lambda > 0:
        criteria = nn.L1Loss(size_average=False)
        regularizer_loss = 0
        for parameter in model.parameters():
            regularizer_loss += criteria(parameter, torch.zeros_like(parameter))
        loss += Lambda * regularizer_loss
    return loss


def cross_entropy_loss():
    """Create Cross Entropy Loss
    Returns:
        Cross entroy loss function
    """
    return nn.CrossEntropyLoss()


def sgd_optimizer(model, learning_rate, momentum, l2_factor=0.0):
    """Create optimizer.
    Args:
        model: Model instance.
        learning_rate: Learning rate for the optimizer.
        momentum: Momentum of optimizer.
        l2_factor: Factor for L2 regularization.
    
    Returns:
        SGD optimizer.
    """
    return optim.SGD(
        model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=l2_factor
    )


def find_lr(net, optimizer, criterion, train_loader):
    """Find learning rate for using One Cyclic LRFinder
    Args:
        net (instace): torch instace of defined model
        optimizer (instance): optimizer to be used
        criterion (instance): criterion to be used for calculating loss
        train_loader (instance): torch dataloader instace for trainig set
    """
    lr_finder = LRFinder(net, optimizer, criterion, device="cuda")
    lr_finder.range_test(train_loader, end_lr=10, num_iter=200, step_mode="exp")
    lr_finder.plot()
    min_loss = min(lr_finder.history['loss'])
    ler_rate = lr_finder.history['lr'][np.argmin(lr_finder.history['loss'], axis=0)]
    print("Max LR is {}".format(ler_rate))

    lr_finder.reset()
    return format(ler_rate)


def class_level_accuracy(model, loader, device, classes):
    """Print test accuracy for each class in dataset.
    Args:
        model: Model instance.
        loader: Data loader.
        device: Device where data will be loaded.
        classes: List of classes in the dataset.
    """

    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))

    with torch.no_grad():
        for _, (images, labels) in enumerate(loader, 0):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()

            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1


    for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))

