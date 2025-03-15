# Description: This file is used to train the model using the dataset.

from dataset import SignatureDataset
from torchvision.models import vgg16, VGG16_Weights
from torchvision.transforms import ToTensor, Resize, Compose, Normalize
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def plot_confusion_matrix(writer, cm, class_names, epoch):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """

    figure = plt.figure(figsize=(20, 20))
    # color map: https://matplotlib.org/stable/gallery/color/colormap_reference.html
    plt.imshow(cm, interpolation='nearest', cmap="Wistia")
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            color = "white" if cm[i, j] > threshold else "black"
            plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    writer.add_figure('confusion_matrix', figure, epoch)


def Train():
    resume_training = False
    batch_size = 64
    SIZE = 224
    num_epoches = 20
    learning_rate = 1e-4 
    num_classes = 2
    best_accuracy = -1
    tensorboard_path = "my_tensorboard"
    checkpoint_path = "my_checkpoint"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # preprocessing
    transform = Compose([
        Resize((SIZE, SIZE)),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # get train data
    train_dataset = SignatureDataset(train=True, transform=transform)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )
    # get validation data
    valid_dataset = SignatureDataset(train=False, transform=transform)
    valid_dataloader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )
    # MODEL: VGG16
    model = vgg16(VGG16_Weights.IMAGENET1K_V1)
    model.classifier[-1] = nn.Linear(4096, num_classes, bias=True)
    model.to(device)
    # OPTIMIZER: ADAM
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # LOSS: crossEntropyloss
    criterion = torch.nn.CrossEntropyLoss()
    writer = SummaryWriter(tensorboard_path)
    num_iters = len(train_dataloader)
    # resume training
    if resume_training:
        checkpoint = os.path.join(checkpoint_path, "last.pt")
        saved_data = torch.load(checkpoint)
        model.load_state_dict(saved_data["model"])
        optimizer.load_state_dict(saved_data["optimizer"])
        best_accuracy = saved_data["best_accuracy"]
        start_epoch = saved_data["epoch"]
    else:
        start_epoch = 0
        best_accuracy = -1
    if not os.path.isdir(tensorboard_path):
        os.makedirs(tensorboard_path)
    if not os.path.isdir(checkpoint_path):
        os.makedirs(checkpoint_path)
    for epoch in range(start_epoch, num_epoches):
        # cho model vào training mode
        model.train()
        progress_bar = tqdm(train_dataloader, colour="cyan")
        total_losses = []
        for iter, (images, labels) in enumerate(progress_bar):
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            loss = criterion(output, labels)
            total_losses.append(loss.item())
            avg_loss = np.mean(total_losses)
            progress_bar.set_description("epoch {}/{}, loss {: .4f}".format(epoch+1, num_epoches, loss))
            writer.add_scalar("Train/Loss", avg_loss, global_step=num_iters*epoch+iter)
            # backward
            optimizer.zero_grad() # clear gradients
            loss.backward() # update parameters
            optimizer.step() # update weights
        with torch.no_grad():
            all_predictions = []
            all_labels = []
            total_losses = []
            model.eval()
            progress_bar = tqdm(valid_dataloader, colour="magenta")
            for iter, (images, labels) in enumerate(progress_bar):
                images = images.to(device)
                all_labels.extend(labels)
                labels = labels.to(device)
                output = model(images)
                prediction = torch.argmax(output, dim=1).tolist()
                all_predictions.extend(prediction)
                loss = criterion(output, labels)
                total_losses.append(loss.item())
        avg_loss = np.mean(total_losses)
        accuracy = accuracy_score(all_labels, all_predictions)
        progress_bar.set_description("loss: {: 0.4f}, accuracy: {}".format(avg_loss, accuracy))
        writer.add_scalar("Valid/Loss", avg_loss, global_step=epoch)
        writer.add_scalar("Valid/Accuracy", accuracy, global_step=epoch)
        plot_confusion_matrix(writer, confusion_matrix(all_labels, all_predictions), [0, 1], epoch)
        # save model
        saved_data = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_accuracy": best_accuracy,
            "epoch": epoch+1,
        }
        checkpoint = os.path.join(checkpoint_path, "last.pt")   # used for resume training
        torch.save(saved_data, checkpoint)
        if accuracy > best_accuracy:
            checkpoint = os.path.join(checkpoint_path, "best.pt")   # used for save best accuracy
            torch.save(saved_data, checkpoint)
            best_accuracy = accuracy