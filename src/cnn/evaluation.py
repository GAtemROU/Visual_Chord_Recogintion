import torch
import matplotlib.pyplot as plt
import numpy as np


def evaluate(model, test_loader, device, class_labels):
    
    
    model.eval()

    misclassified_images = []
    misclassified_labels = []
    misclassified_predictions = []
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            misclassified_mask = predicted != labels
            misclassified_images.extend(inputs[misclassified_mask])
            misclassified_labels.extend(labels[misclassified_mask])
            misclassified_predictions.extend(predicted[misclassified_mask])
    # Convert the misclassified images, labels, and predictions to numpy arrays
    misclassified_images = [x.cpu().numpy() for x in misclassified_images]
    misclassified_labels = [x.cpu().numpy() for x in misclassified_labels]
    misclassified_predictions = [x.cpu().numpy() for x in misclassified_predictions]
    accuracy = (total - len(misclassified_labels))/total
    print(f'Test accuracy: {accuracy}')
    # Plot the misclassified images along with their predicted and ground truth labels
    print('Here are the wrong predicitons:')
    num_samples = len(misclassified_images)
    num_cols = 5
    num_rows = int(np.ceil(num_samples / num_cols))

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 12))

    for i, ax in enumerate(axes.flatten()):
        if i < num_samples:
            image = np.transpose(misclassified_images[i], (1,2,0))
            label = misclassified_labels[i]
            prediction = misclassified_predictions[i]

            ax.imshow(image)
            ax.set_title(f"True: {class_labels[label]}, Predicted: {class_labels[prediction]}")
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()
    
    return accuracy