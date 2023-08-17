# License: BSD
# Author: Sasank Chilamkurthy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# DEVICE = torch.device("cpu")
class ResNet():
    def main(self):
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []
        cudnn.benchmark = True
        plt.ion()   # interactive mode

        # Data augmentation and normalization for training
        # Just normalization for validation
        self.data_transforms = {
            'train': transforms.Compose([
                # transforms.RandomResizedCrop(224),
                # transforms.RandomHorizontalFlip(),
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
            'val': transforms.Compose([
                transforms.Resize(224),
                # transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        }

        data_dir = 'vision_project'
        self.image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                                self.data_transforms[x])
                        for x in ['train', 'val']}
        self.dataloaders = {x: torch.utils.data.DataLoader(self.image_datasets[x], batch_size=4,
                                                    shuffle=True, num_workers=4)
                    for x in ['train', 'val']}
        self.dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train', 'val']}
        self.class_names = self.image_datasets['train'].classes

        self.device = torch.device(DEVICE)




        # Get a batch of training data
        inputs, classes = next(iter(self.dataloaders['train']))

        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs)

        self.imshow(out, title=[self.class_names[x] for x in classes])






        self.model_ft = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = self.model_ft.fc.in_features
        # Here the size of each output sample is set to 2.
        # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
        
        # for param in model_ft.parameters():
        #   param.requires_grad = False
        # for param in model_ft.layer4.parameters():
        #     param.requires_grad = True
        self.model_ft.fc = nn.Linear(num_ftrs, 7)

        self.model_ft = self.model_ft.to(self.device)



        criterion = nn.CrossEntropyLoss()

        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(self.model_ft.parameters(), lr=0.001, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        self.model_ft = self.train_model(criterion, optimizer_ft, exp_lr_scheduler,
                            num_epochs=10)

        self.visualize_model(self.model_ft)

    def visualize_model(self, model, num_images=6):
        was_training = model.training
        model.eval()
        images_so_far = 0
        fig = plt.figure()

        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.dataloaders['val']):
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                for j in range(inputs.size()[0]):
                    images_so_far += 1
                    ax = plt.subplot(num_images//2, 2, images_so_far)
                    ax.axis('off')
                    ax.set_title(f'predicted: {self.class_names[preds[j]]}')
                    self.imshow(inputs.cpu().data[j])

                    if images_so_far == num_images:
                        model.train(mode=was_training)
                        return
            model.train(mode=was_training)

    def train_model(self, criterion, optimizer, scheduler, num_epochs=25):
        since = time.time()
        # Create a temporary directory to save training checkpoints
        with TemporaryDirectory() as tempdir:
            best_model_params_path = os.path.join(tempdir, 'best_model_params.pt')

            torch.save(self.model_ft.state_dict(), best_model_params_path)
            best_acc = 0.0

            for epoch in range(num_epochs):
                print(f'Epoch {epoch}/{num_epochs - 1}')
                print('-' * 10)

                # Each epoch has a training and validation phase
                for phase in ['val', 'train']:
                    if phase == 'train':
                        self.model_ft.train()  # Set model to training mode
                    else:
                        self.model_ft.eval()   # Set model to evaluate mode

                    running_loss = 0.0
                    running_corrects = 0

                    # Iterate over data.
                    for inputs, labels in self.dataloaders[phase]:
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward
                        # track history if only in train
                        with torch.set_grad_enabled(phase == 'train'):
                            outputs = self.model_ft(inputs)
                            _, preds = torch.max(outputs, 1)
                            loss = criterion(outputs, labels)

                            # backward + optimize only if in training phase
                            if phase == 'train':
                                loss.backward()
                                optimizer.step()

                        # statistics
                        running_loss += loss.item() * inputs.size(0)
                        running_corrects += torch.sum(preds == labels.data)
                    if phase == 'train':
                        scheduler.step()


                    epoch_loss = running_loss / self.dataset_sizes[phase]
                    epoch_acc = float(running_corrects) / self.dataset_sizes[phase]

                    print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                    # deep copy the model
                    if phase == 'train':
                        self.train_loss.append(epoch_loss)
                        self.train_acc.append(epoch_acc)
                        print(self.val_loss.append(epoch_loss))
                    if phase == 'val':
                        self.val_loss.append(epoch_loss)
                        self.val_acc.append(epoch_acc)
                        print(self.val_loss.append(epoch_loss))
                    if phase == 'val' and epoch_acc > best_acc:
                        best_acc = epoch_acc
                        torch.save(self.model_ft.state_dict(), best_model_params_path)
                    
                print()

            time_elapsed = time.time() - since
            print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
            print(f'Best val Acc: {best_acc:4f}')

            # load best model weights
            self.model_ft.load_state_dict(torch.load(best_model_params_path))
        return self.model_ft
    
    def imshow(self, inp, title=None):
        """Display image for Tensor."""
        inp = inp.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        plt.imshow(inp)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated


if __name__ == '__main__':
    obj = ResNet()
    obj.main()
    PATH = '/vision_project/mymodel1.pth'
    torch.save(obj.model_ft.state_dict(), PATH)

    ## Plot loss, accuracy graphs
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(8, 10))

    # Plot data on the first subplot
    axes[0].plot(obj.val_loss, label='Validation loss')
    axes[0].plot(obj.train_loss, label='Train loss')
    axes[0].set_title('Loss')
    axes[0].legend()

    # Plot data on the second subplot
    axes[1].plot(obj.val_acc, label='Val acc')
    axes[1].plot(obj.train_acc, label='Train acc')
    axes[1].set_title('Acc')
    axes[1].legend()

    # Adjust layout
    plt.tight_layout()

    # Show the plots
    plt.show()

    ## Get test set accuracy

    data_transforms = {
        'test': transforms.Compose([
            # transforms.RandomResizedCrop(224),
            # transforms.RandomHorizontalFlip(),
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    data_dir = '/content/gdrive/My Drive'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=255,
                                                shuffle=True, num_workers=4)
                for x in ['test']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['test']}
    class_names = image_datasets['test'].classes
    obj.model_ft.eval()
    phase = 'test'
    running_corrects = 0
    epoch_acc = 0
    for inputs, labels in dataloaders[phase]:
        inputs = inputs.to('cuda')
        labels = labels.to('cuda')



        # forward
        # track history if only in train
        with torch.set_grad_enabled(phase == 'test'):
            outputs = obj.model_ft(inputs)
            _, preds = torch.max(outputs, 1)
            print(preds, labels.data)
            running_corrects += torch.sum(preds == labels.data)
            if (preds == labels.data).all:
                print('correct')

    epoch_acc = running_corrects.double() / dataset_sizes[phase]

    print(epoch_acc)

    ## Plot tsne graph for trained model
    from torchvision.models.feature_extraction import create_feature_extractor
    from torchvision.io import read_image
    from sklearn.manifold import TSNE
    import pandas as pd
    import seaborn as sns


    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    return_nodes = {'flatten': 'flatten'}
    feature_extractor = create_feature_extractor(obj.model_ft, return_nodes=return_nodes)

    # Step 4: Load the image(s) and apply inference preprocessing transf
    inputs, classes = next(iter(dataloaders['test']))
    feature_list = []
    for image in inputs:
        features = feature_extractor(image.to('cuda').unsqueeze(0))
        flatten_fts = features["flatten"].squeeze()
        feature_list.append(flatten_fts.to('cpu').detach().numpy())

    print(np.shape(feature_list))
    z = tsne.fit_transform(torch.tensor(feature_list))
    df = pd.DataFrame()
    df["y"] = classes
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]

    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", 7),
                    data=df).set(title="Iris data T-SNE projection")


    ## Plot tsne graph for resnet model without training
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT).to('cuda')
    tsne = TSNE(n_components=2, verbose=1, random_state=123)
    return_nodes = {'flatten': 'flatten'}
    feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)

    # Step 4: Load the image(s) and apply inference preprocessing transf
    inputs, classes = next(iter(dataloaders['vision_project']))
    feature_list = []
    for image in inputs:
        features = feature_extractor(image.to('cuda').unsqueeze(0))
        flatten_fts = features["flatten"].squeeze()
        feature_list.append(flatten_fts.to('cpu').detach().numpy())

    print(np.shape(feature_list))
    z = tsne.fit_transform(torch.tensor(feature_list))
    df = pd.DataFrame()
    df["y"] = classes
    df["comp-1"] = z[:,0]
    df["comp-2"] = z[:,1]

    sns.scatterplot(x="comp-1", y="comp-2", hue=df.y.tolist(),
                    palette=sns.color_palette("hls", 7),
                    data=df).set(title="Iris data T-SNE projection")
