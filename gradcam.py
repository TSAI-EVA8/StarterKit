import torch
import torch.nn.functional as F
import numpy as np
import cv2

import torch
import matplotlib.pyplot as plt
class GradCAM:
    """Calculate GradCAM salinecy map.

    Args:
        input: Input image with shape of (1, 3, H, W)
        class_idx: Class index for calculating GradCAM.
            If not specified, the class index that makes the highest model prediction score will be used.

    Returns:
        mask: Saliency map of the same spatial dimension with input
        logit: Model output
    """

    def __init__(self, model, layer_name):
        self.model = model
        self.layer_name = layer_name
        self._target_layer()

        self.gradients = dict()
        self.activations = dict()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['value'] = grad_output[0]

        def forward_hook(module, input, output):
            self.activations['value'] = output

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def _target_layer(self):
        layer_num = int(self.layer_name.lstrip('layer'))
        if layer_num == 1:
            self.target_layer = self.model.layer1
        elif layer_num == 2:
            self.target_layer = self.model.layer2
        elif layer_num == 3:
            self.target_layer = self.model.layer3
        elif layer_num == 4:
            self.target_layer = self.model.layer4
      
    def saliency_map_size(self, *input_size):
        device = next(self.model.parameters()).device
        self.model(torch.zeros(1, 3, *input_size, device=device))
        return self.activations['value'].shape[2:]

    def forward(self, input, class_idx=None, retain_graph=False):
        b, c, h, w = input.size()

        logit = self.model(input)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        self.model.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']
        activations = self.activations['value']
        b, k, u, v = gradients.size()

        alpha = gradients.view(b, k, -1).mean(2)
        # alpha = F.relu(gradients.view(b, k, -1)).mean(2)
        weights = alpha.view(b, k, 1, 1)

        saliency_map = (weights*activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map - saliency_map_min).div(saliency_map_max - saliency_map_min).data

        return saliency_map, logit

    def __call__(self, input, class_idx=None, retain_graph=False):
        return self.forward(input, class_idx, retain_graph)
    

class GradCAMPP(GradCAM):
    """Calculate GradCAM++ salinecy map.

    Args:
        input: Input image with shape of (1, 3, H, W)
        class_idx: Class index for calculating GradCAM.
            If not specified, the class index that makes the highest model prediction score will be used.

    Returns:
        mask: saliency map of the same spatial dimension with input
        logit: model output
    """

    def forward(self, input, class_idx=None, retain_graph=False):
        b, c, h, w = input.size()

        logit = self.model(input)
        if class_idx is None:
            score = logit[:, logit.max(1)[-1]].squeeze()
        else:
            score = logit[:, class_idx].squeeze()

        self.model.zero_grad()
        score.backward(retain_graph=retain_graph)
        gradients = self.gradients['value']  # dS/dA
        activations = self.activations['value']  # A
        b, k, u, v = gradients.size()

        alpha_num = gradients.pow(2)
        alpha_denom = alpha_num.mul(2) + activations.mul(gradients.pow(3)).view(b, k, u*v).sum(-1).view(b, k, 1, 1)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))

        alpha = alpha_num.div(alpha_denom+1e-7)
        positive_gradients = F.relu(score.exp() * gradients)  # ReLU(dY/dA) == ReLU(exp(S)*dS/dA))
        weights = (alpha * positive_gradients).view(b, k, u*v).sum(-1).view(b, k, 1, 1)

        saliency_map = (weights * activations).sum(1, keepdim=True)
        saliency_map = F.relu(saliency_map)
        saliency_map = F.upsample(saliency_map, size=(h, w), mode='bilinear', align_corners=False)
        saliency_map_min, saliency_map_max = saliency_map.min(), saliency_map.max()
        saliency_map = (saliency_map-saliency_map_min).div(saliency_map_max-saliency_map_min).data

        return saliency_map, logit




def unnormalize(image, mean, std, out_type='array'):
    """Un-normalize a given image.
    
    Args:
        image: A 3-D ndarray or 3-D tensor.
            If tensor, it should be in CPU.
        mean: Mean value. It can be a single value or
            a tuple with 3 values (one for each channel).
        std: Standard deviation value. It can be a single value or
            a tuple with 3 values (one for each channel).
        out_type: Out type of the normalized image.
            If `array` then ndarray is returned else if
            `tensor` then torch tensor is returned.
    """

    if type(image) == torch.Tensor:
        image = np.transpose(image.clone().numpy(), (1, 2, 0))
    
    normal_image = image * std + mean
    if out_type == 'tensor':
        return torch.Tensor(np.transpose(normal_image, (2, 0, 1)))
    elif out_type == 'array':
        return normal_image
    return None  # No valid value given


def to_numpy(tensor):
    """Convert 3-D torch tensor to a 3-D numpy array.
    Args:
        tensor: Tensor to be converted.
    """
    return np.transpose(tensor.clone().numpy(), (1, 2, 0))


def to_tensor(ndarray):
    """Convert 3-D numpy array to 3-D torch tensor.
    Args:
        ndarray: Array to be converted.
    """
    return torch.Tensor(np.transpose(ndarray, (2, 0, 1)))

def visualize_cam(mask, img, alpha=1.0):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.

    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]
    Returns:

        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """

    heatmap = (255 * mask.squeeze()).type(torch.uint8).cpu().numpy()
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b]) * alpha

    result = heatmap + img.cpu()
    result = result.div(result.max()).squeeze()

    return heatmap, result


class GradCAMView:

    def __init__(self, model, layers, device, mean, std):
        """Instantiate GradCAM and GradCAM++.

        Args:
            model: Trained model.
            layers: List of layers to show GradCAM on.
            device: GPU or CPU.
            mean: Mean of the dataset.
            std: Standard Deviation of the dataset.
        """
        self.model = model
        self.layers = layers
        self.device = device
        self.mean = mean
        self.std = std

        self._gradcam()
        self._gradcam_pp()

        #print('Mode set to GradCAM.')
        self.grad = self.gradcam.copy()

        self.views = []

    def _gradcam(self):
        """ Initialize GradCAM instance. """
        self.gradcam = {}
        for layer in self.layers:
            self.gradcam[layer] = GradCAM(self.model, layer)
    
    def _gradcam_pp(self):
        """ Initialize GradCAM++ instance. """
        self.gradcam_pp = {}
        for layer in self.layers:
            self.gradcam_pp[layer] = GradCAMPP(self.model, layer)
    
    def switch_mode(self):
        if self.grad == self.gradcam:
            print('Mode switched to GradCAM++.')
            self.grad = self.gradcam_pp.copy()
        else:
            print('Mode switched to GradCAM.')
            self.grad = self.gradcam.copy()
    
    def _cam_image(self, norm_image):
        """Get CAM for an image.

        Args:
            norm_image: Normalized image. Should be of type
                torch.Tensor
        
        Returns:
            Dictionary containing unnormalized image, heatmap and CAM result.
        """
        norm_image_cuda = norm_image['image'].clone().unsqueeze_(0).to(self.device)
        heatmap, result = {}, {}
        for layer, gc in self.gradcam.items():
            mask, _ = gc(norm_image_cuda)
            cam_heatmap, cam_result = visualize_cam(
                mask,
                unnormalize(norm_image['image'], self.mean, self.std, out_type='tensor').clone().unsqueeze_(0).to(self.device)
            )
            heatmap[layer], result[layer] = to_numpy(cam_heatmap), to_numpy(cam_result)
        return {
            'image': unnormalize(norm_image['image'], self.mean, self.std),
            'label':norm_image['labelClass'],
            'prediction':norm_image['predictionClass'],
            'heatmap': heatmap,
            'result': result
        }
    
    # def _plot_view(self, view, fig, row_num, ncols, metric):
    #     """Plot a CAM view.

    #     Args:
    #         view: Dictionary containing image, heatmap and result.
    #         fig: Matplotlib figure instance.
    #         row_num: Row number of the subplot.
    #         ncols: Total number of columns in the subplot.
    #         metric: Can be one of ['heatmap', 'result'].
    #     """
    #     sub = fig.add_subplot(row_num, ncols, 1)
    #     sub.axis('off')
    #     plt.imshow(view['image'])
    #     label=str(view['label'])
    #     prediction=str(view['prediction'])
    #     sub.set_title(f'Label: {label}\nPrediction: {prediction}')
    #     for idx, layer in enumerate(self.layers):
    #         sub = fig.add_subplot(row_num, ncols, idx + 2)
    #         sub.axis('off')
    #         plt.imshow(view[metric][layer])
    #         sub.set_title(layer)
    
    def cam(self, norm_image_list):
        """Get CAM for a list of images.

        Args:
            norm_image_list: List of normalized images. Each image
                should be of type torch.Tensor
        """
        for norm_image in norm_image_list:
            self.views.append(self._cam_image(norm_image))
    
    # def plot(self, plot_path):
    #     """Plot heatmap and CAM result.

    #     Args:
    #         plot_path: Path to save the plot.
    #     """

    #     for idx, view in enumerate(self.views):
    #         # Initialize plot
    #         fig = plt.figure(figsize=(10, 10))

    #         # Plot view
    #         #self._plot_view(view, fig, 1, len(self.layers) + 1, 'heatmap')
    #         self._plot_view(view, fig, 2, len(self.layers) + 1, 'result')
            
    #         # Set spacing and display
    #         fig.tight_layout()
    #         plt.show()

    #         # Save image
    #         fig.savefig(f'{plot_path}_{idx + 1}.png', bbox_inches='tight')

    #         # Clear cache
    #         plt.clf()
    
    def plot(self):
        fig, axs = plt.subplots(len(self.views), 5,figsize=(32,32))
        row_count=-1
        #plt.rcParams['figure.figsize'] = [10, 10]
        for idx, view in enumerate(self.views):
            label = view['label']
            prediction = view['prediction']
            row_count += 1
            axs[row_count][0].imshow(view['image']/2)
            axs[row_count][0].set_title(f'Label: {label}\nPrediction: {prediction}')
            axs[row_count][0].title.set_size(20)
            axs[row_count][0].axis('off')
            for j in range(1,5):
                layer="layer"+str(j)
                axs[row_count][j].imshow(view['result'][layer])
                axs[row_count][j].set_title(layer)
                axs[row_count][j].title.set_size(20)
                axs[row_count][j].axis('off')
            #plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=None, wspace=None, hspace=None)

            
        # Set spacing and display
        fig.tight_layout(pad=0.5)
        plt.show()
        plt.clf()
   
    def __call__(self, norm_image_list, plot_path):
        """Get GradCAM for a list of images.

        Args:
            norm_image_list: List of normalized images. Each image
                should be of type torch.Tensor
        """
        self.cam(norm_image_list)
        self.plot()