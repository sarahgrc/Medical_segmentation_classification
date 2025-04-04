import torch
from patsy.test_highlevel import test_env_transform
from torchvision import transforms
from models3D import Clf3D
from models2D import *
from dataset.dataset import Dataset2D
from torch.utils.data import DataLoader
from models2D import ResNet
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.optim as optim



data_root = './archive'


train_transforms = transforms.Compose([
    transforms.Resize((256,256)),
    # data augmentation
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomRotation(degrees=(-0.2, +0.2)),

    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.PILToTensor(),
    transforms.ConvertImageDtype(torch.float),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])




def train(model, train_loader, val_loader, num_epochs=25, lr=1e-4, device='cpu', tolerance = 4, name_wtg =''):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    train_losses = []
    val_losses = []

    tolerance_c = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, leave=True)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images).squeeze(1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss == min(val_losses):
            torch.save(model.state_dict(), f"best_model_{name_wtg}.pth")

        # early stop
        if val_loss > val_losses[-1] :
            tolerance_c += 1
            if tolerance_c == tolerance :
                print(f'** EARLY STOPPING ** at epoch : {tolerance_c}')
                break


    # Visualisation
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss", marker="o")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Learning Curves")
    plt.grid()
    plt.show()



if __name__ == '__main__' :
    # data loader
    train_dataset = Dataset2D(data_root=data_root,
                              transforms=train_transforms,
                              mode='train')

    val_dataset = Dataset2D(data_root=data_root,
                            transforms=train_transforms,
                            mode='val')

    test_dataset = Dataset2D(data_root=data_root,
                             transforms=test_transforms,
                             mode='test')
    # display data info
    train_dataset.get_info()
    test_dataset.get_info()

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # models
    resnet50 = ResNet(num_class=4, model_name='ResNet50', trainable_layers=['layer4', 'fc'])
    resnet34 = ResNet(num_class=4, model_name='ResNet34', trainable_layers=['layer4', 'fc'])

    # train
    train(resnet34, train_loader, val_loader, name_wtg='resnet34_10epoch')
    train(resnet50, train_loader, val_loader, name_wtg='resnet50_10epoch')
    train(resnet34, train_loader, val_loader, name_wtg='resnet34_10epoch')