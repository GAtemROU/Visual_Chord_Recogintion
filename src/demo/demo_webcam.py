import cv2
import torch
import torchvision.transforms as transforms
from cnn import main_model
import numpy as np


model = torch.load('model/bestmodel.pth')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)
model.eval()
transform = transforms.Compose([
    transforms.ToTensor() 
])


cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 0.5
color = (255, 0, 0)
thickness = 2
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

while True:
    ret, img = cap.read()
    image = cv2.resize(img, (80, 80))
#     image = cv2.flip(image, 1)
    image = transform(image)
    image = image.to(device)
    results = model(image.unsqueeze(0))
    results = results.cpu().detach().numpy().squeeze()
    sorted_results = sorted(results, reverse=True)
    for i, res in enumerate(sorted_results):
        cv2.putText(img, f'{labels[np.where(results==res)[0][0]]} --- {"{:.2f}".format(res)}', [10, 10+20*i], font, fontScale, color, thickness)
    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

