# Description: This file is used to load the trained model and make predictions on the test data.

import torch
import torch.nn as nn
import cv2
import os
import numpy as np
from torchvision.models import vgg16, VGG16_Weights
import warnings
warnings.filterwarnings("ignore")


def inference():
    predict = ["forge", "genuine"]
    checkpoint_path = "my_checkpoint"
    image_path = "test_Signature/genuine/NFI-03005030.PNG"
    SIZE = 224
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([0.229, 0.224, 0.225])
    # Run on GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # MODEL
    model = vgg16(VGG16_Weights.IMAGENET1K_V1)
    model.classifier[-1] = nn.Linear(4096, 2)
    model.to(device)
    # Load model
    checkpoint = os.path.join(checkpoint_path, "best.pt")
    saved_data = torch.load(checkpoint)
    model.load_state_dict(saved_data["model"])
    model.eval()
    # preprocessing images
    original_image = cv2.imread(image_path)
    image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (SIZE, SIZE))
    image = image/255
    image = (image-mean)/std
    image = np.transpose(image, (2, 0, 1))
    image = image[None, :, :, :]
    image = torch.from_numpy(image).float().to(device)
    # softmax for accuracy
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        output = model(image).to(device)
        predicted_index = torch.argmax(output, dim=1).item()  # Get the predicted class index as an integer
        predict_class = predict[predicted_index]
        # Apply softmax and extract the probability for the predicted class:
        prob = softmax(output)[0, predicted_index].item()

        print("Predict: ", predict_class)
        print("Probability: {: 0.2f}%".format(prob * 100))
        cv2.imshow("image", original_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
if __name__ == "__main__":
    inference()