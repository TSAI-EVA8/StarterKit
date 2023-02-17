import torch
import torch
import torch.nn.functional as F
import torch.nn.functional as F
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torchvision

def set_seed(seed, cuda):
    """ Setting the seed makes the results reproducible. """
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)


def initialize_cuda(seed):
    """ Check if GPU is availabe and set seed. """

    # Check CUDA availability
    cuda = torch.cuda.is_available()
    print('GPU Available?', cuda)

    # Initialize seed
    set_seed(seed, cuda)

    # Set device
    device = torch.device("cuda" if cuda else "cpu")

    return cuda,device 



def plot_metric(values, metric):
    '''Plot the individual metric curves'''
    # Initialize a figure
    fig = plt.figure(figsize=(7, 5))

    # Plot values
    plt.plot(values)

    # Set plot title
    plt.title(f'Validation {metric}')

    # Label axes
    plt.xlabel('Epoch')
    plt.ylabel(metric)

    # Set legend
    location = 'upper' if metric == 'Loss' else 'lower'

    # Save plot
    fig.savefig(f'{metric.lower()}_change.png')


def plotLossAccuracyCurve(train_losses,train_accuracies,test_losses,test_accuracies):
    '''Plots the combined plot for the training Loss, testing Loss, Training Accuracy and test accuracy
    Args:
        train_losses: list of the losses per epoch for the training.
        train_accuracies: accuracy per epoch
        test_losses: test loss per epoch.
        test_accuracies: test accuracy per epoch.
    '''
    
    sns.set()

    fig, axs = plt.subplots(2,2,figsize=(15,10))
    #ax = plt.GridSpec(2, 2)
    fig.tight_layout(pad=5.0)

    x = range(0,len(train_losses))
    labels=[str(i+1) for i in x]

    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Metric:Training Loss")
    axs[0, 0].set_xticks(x, labels)
    axs[0, 0].set_xlabel("Epoch")

    axs[1, 0].plot(train_accuracies)
    axs[1, 0].set_title("Metric:Training Accuracy")
    axs[1, 0].set_xticks(x, labels)
    axs[1, 0].set_xlabel("Epoch")

    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Metric:Test Loss")
    axs[0, 1].set_xticks(x, labels)
    axs[0, 1].set_xlabel("Epoch")

    axs[1, 1].plot(test_accuracies)
    axs[1, 1].set_title("Metric:Test Accuracy")
    axs[1, 1].set_xticks(x, labels)
    axs[1, 1].set_xlabel("Epoch")

    plt.show()



def save_and_show_result(data, classes):
    """Display 25 misclassified images.
    Args:
        data: Contains model predictions and labels.
    """

    # # Create directories for saving data
    # path = os.path.join(
    #     os.path.dirname(os.path.abspath(__file__)), 'predictions'
    # )
    # if not os.path.exists(path):
    #     os.makedirs(path)

    # Initialize plot
    row_count = -1
    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    fig.tight_layout()

    for idx, result in enumerate(data):

        # If 25 samples have been stored, break out of loop
        if idx > 24:
            break
        
        rgb_image = np.transpose(result['image'], (1, 2, 0)) / 2 + 0.5
        label = result['label'].item()
        prediction = result['prediction'].item()

        # Plot image
        if idx % 5 == 0:
            row_count += 1
        axs[row_count][idx % 5].axis('off')
        axs[row_count][idx % 5].set_title(f'Label: {classes[label]}\nPrediction: {classes[prediction]}')
        axs[row_count][idx % 5].imshow(rgb_image)
    
def displaySamples(images,labels,classes):
    # functions to show an image
    row_count = -1
    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    fig.tight_layout()
    for idx in range(25):
        # #print(labels[idx])
        #     # If 25 samples have been stored, break out of loop
        # if idx > 24:
        #     break
        
        rgb_image = np.transpose(images[idx], (1, 2, 0)) / 2 + 0.5
        label = labels[idx]
        #prediction = result['prediction'].item()

        # Plot image
        if idx % 5 == 0:
            row_count += 1
        axs[row_count][idx % 5].axis('off')
        axs[row_count][idx % 5].set_title(f'Label: {classes[label]}')
        axs[row_count][idx % 5].imshow(rgb_image)


def displayImageGridFromLoader(loader):
    dataiter = iter(loader)
    images, labels = next(dataiter)

    # show images..
    ## get the grid of the image
    # imageGrid=torchvision.utils.make_grid(images[0:25])
    
    # ## logic to unnormalize
    # imageGrid = imageGrid / 2 + 0.5     # unnormalize
    # npimg = imageGrid.numpy()
    
    # ## ste the figure size
    # plt.subplots(figsize=(10, 10))
    # plt.axis('off')
    # ## plot the grid
    # plt.imshow(np.transpose(npimg, (1, 2, 0)))
    
    row_count = -1
    fig, axs = plt.subplots(5, 5, figsize=(10, 10))
    fig.tight_layout()
    for idx in range(25):
        # #print(labels[idx])
        #     # If 25 samples have been stored, break out of loop
        # if idx > 24:
        #     break
        
        rgb_image = np.transpose(images[idx], (1, 2, 0)) / 2 + 0.5
        label = labels[idx]
        #prediction = result['prediction'].item()

        # Plot image
        if idx % 5 == 0:
            row_count += 1
        axs[row_count][idx % 5].axis('off')
        #axs[row_count][idx % 5].set_title(f'Label: {classes[label]}')
        axs[row_count][idx % 5].imshow(rgb_image)


def plot_metric(data, metric):
    """Plot accuracy graph or loss graph.
    Args:
        data (list or dict): If only single plot then this is a list, else
            for multiple plots this is a dict with keys containing.
            the plot name and values being a list of points to plot
        metric (str): Metric name which is to be plotted. Can be either
            loss or accuracy.
    """

    single_plot = True
    if type(data) == dict:
        single_plot = False
    
    # Initialize a figure
    fig = plt.figure(figsize=(7, 5))

    # Plot data
    if single_plot:
        plt.plot(data)
    else:
        plots = []
        for value in data.values():
            plots.append(plt.plot(value)[0])

    # Set plot title
    plt.title(f'{metric} Change')

    # Label axes
    plt.xlabel('Epoch')
    plt.ylabel(metric)

    if not single_plot: # Set legend
        location = 'upper' if metric == 'Loss' else 'lower'
        plt.legend(
            tuple(plots), tuple(data.keys()),
            loc=f'{location} right',
            shadow=True,
            prop={'size': 15}
        )

    # Save plot
    fig.savefig(f'{metric.lower()}_change.png')




class CutOut(object):
    """Randomly mask out one or more patches from an image.

    Args:
        n_holes (int): Number of patches to cut out of each image.
        length (int): The length (in pixels) of each square patch.
    """
    def __init__(self, n_holes, length,p):
        self.n_holes = n_holes
        self.length = length
        self.p=p

    def __call__(self, img):
        """
        Args:
            img (Tensor): Tensor image of size (C, H, W).
        Returns:
            Tensor: Image with n_holes of dimension length x length cut out of it.
        """
        if torch.rand(1) > self.p:
            return img
        h = img.size(1)
        w = img.size(2)

        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - self.length // 2, 0, h)
            y2 = np.clip(y + self.length // 2, 0, h)
            x1 = np.clip(x - self.length // 2, 0, w)
            x2 = np.clip(x + self.length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img