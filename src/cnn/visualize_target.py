import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt

def visualize(model, target_layer, image, class_labels):
    model.eval()
    
    features = image.unsqueeze(0)
    features.requires_grad_()
    output = model(features)
    _, predicted = torch.max(output, 1)
    predicted_class = predicted.item()

    # Compute gradients of the predicted class score with respect to the target layer's output
    one_hot_output = torch.zeros_like(output)
    one_hot_output[0][predicted_class] = 1
    model.zero_grad()
    output.backward(gradient=one_hot_output)

    # Get the gradients and feature maps of the target layer
    gradients = features.grad[0].cpu().numpy()
    activations = target_layer(features).detach().squeeze(0).cpu().numpy()

    # Compute the weights using global average pooling (GAP) of the gradients
    weights = np.transpose(gradients, (1, 2, 0))
    weights = np.maximum(weights, 0)
    
    print(f'Predicted: {class_labels[predicted_class]}')
    
    # plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    image = image.permute(1, 2, 0).cpu().numpy()
    ax1.imshow(image)
    ax1.axis('off')
    ax1.set_title('Input Image')
    ax2.imshow(weights)
    ax2.axis('off')
    ax2.set_title('Weights')
