import argparse
from torchvision import transforms
from models2D import *
import seaborn as sns
from dataset.dataset import Dataset2D
from torch.utils.data import DataLoader
from models2D import ResNet
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix


# writer = SummaryWriter('../../runs/train/')
# writer.add_scalar(name, tensor, iteration)
# writer.add_histogram(name, tensor, iteration)


# train( , , train_su if itere %10 ==0 else None,)
# dans les fc on rajouter aussi writer
# dans fontion du train
# def train_one_iter(model, opt, im, label, writer, iter)
# if writer is not None,:
#    writer.add_scalar('loss', loss, iter)


def parser():
    """ Parser """
    parser = argparse.ArgumentParser(description='Parser of train.py')
    parser.add_argument('--data_path',
                        type=str,
                        help='Absolute path to the data folder containing train & validation data',
                        required=True)
    parser.add_argument('--num_class',
                        type=int,
                        help='number of classes for the classification',
                        required=True)
    parser.add_argument('--model_name',
                        type=str,
                        help='name between : [ResNet50, ResNet34, ResNet18, ResNet101] or effb0, effv2s',
                        required = True)
    parser.add_argument('--trainable_layers',
                        type = str,
                        nargs= '+',
                        help = 'Liste of trainable layers',
                        required = True
                        )
    return parser


def get_transformations():
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        # data augmentation
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=(-0.2, +0.2)),

        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.PILToTensor(),
        transforms.ConvertImageDtype(torch.float),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return train_transforms, test_transforms


def train(model, train_loader: DataLoader, val_loader, num_epochs: int = 25, lr: float = 1e-4, device: str = 'cpu',
          tolerance: int = 5, delta_tol: float = 0.02,
          name_wtg: str = '') -> None:
    """
    Train the given model using CrossEntropyLoss and Adam optimizer.
    :param model: model to train
    :param train_loader: trainning data loader
    :param val_loader: validation data loader
    :param num_epochs: number of epochs to train on
    :param lr: learning rate
    :param device: device on witch perform the training
    :param tolerance: (int) : number of epoch to wait before early stopping
    :param delta_tol: (int)
    :param name_wtg: name to add to the name of the wgt file
    :return: None, best wgths are saved and learning curves are ploted
    """

    # define parameters
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
            loop.set_postfix(loss=loss.item())  # show loss on tqdm progression bar

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
            best_model = model.state_dict()

        # early stop
        if (epoch > 2) & (val_loss >= val_losses[-2] - delta_tol):
            tolerance_c += 1
            if tolerance_c == tolerance:
                print(f'** EARLY STOPPING ** at epoch : {epoch}')
                num_epochs = epoch + 1  # reset num epochs for the visualisation part
                torch.save(model.state_dict(), f"best_model_{name_wtg}_epoch{epoch}_early_stop.pth")
                break
        else:
            tolerance_c = 0  # reset tolerance counter

    torch.save(best_model, f"best_model_{name_wtg}_epoch{epoch}.pth")

    # Visualisation
    plt.figure(figsize=(8, 6))
    print('** epochs : ', num_epochs, '** len train losses ', len(train_losses))
    plt.plot(range(1, num_epochs + 1), train_losses, label="Train Loss", marker="o")
    plt.plot(range(1, num_epochs + 1), val_losses, label="Validation Loss", marker="o")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Learning Curves")
    plt.grid()
    plt.show()


def evaluation(model, test_loader, classes_names: list, device: str = 'cpu'):
    """
    Evaluation of the model
    :param model: model with loaded wgths
    :param test_loader: test data loader
    :param classes_names:(list) list of class names
    :return: None, plot confusion matrix & display classification repport
    """

    preds = []
    labels = []

    model.eval()
    with torch.no_grad():
        for im, label in test_loader:
            # im = tensor ( batch_size, chanel, im_w, im_h)
            # label = tensor([ ....] ) <-- labels of the batch size
            im, label = im.to(device), label.to(device)
            out = model(im)
            pred = torch.argmax(out, dim=1)  # output the index of max element following dim axis

            if device == 'cuda':
                pred = pred.cpu()
            pred = pred.numpy()  # torch.tensor --> numpy

            # store in list
            preds.extend(pred)  # add the batch prediction
            labels.extend(label)  # add labels of the batch

    # Metrics
    print(classification_report(labels, preds))

    cm = confusion_matrix(labels, preds)
    plt.figure()
    sns.heatmap(cm, annot=True, xticklabels=classes_names, yticklabels=classes_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Matrice de Confusion')
    plt.show()


if __name__ == '__main__':
    data_root = 'C:/Users/julie/OneDrive/Bureau/Sarah/cours 5A/projet M2/propre/data_clf_2d/archive'

    # Get parsed arguments
    args = parser().parse_args()
    data_root = args.data_path
    model_name = args.model_name
    num_classes = args.num_class
    trainable_layers = args.trainable_layers

    # Usual parameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # transformation
    train_transforms, test_transforms = get_transformations()

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

    classes_dico = test_dataset.dataset.class_to_idx
    classes_names = list(classes_dico.keys())

    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    # models ResNet
    resnet50 = ResNet(num_class=4, model_name='ResNet50', trainable_layers=['layer4', 'fc'])
    resnet34 = ResNet(num_class=4, model_name='ResNet34', trainable_layers=['layer4', 'fc'])
    resnet34_l = ResNet(num_class=4, model_name='ResNet34', trainable_layers=['fc'])
    resnet18 = ResNet(num_class=4, model_name='ResNet18', trainable_layers=['layer4', 'fc'])

    # models EfficientNet
    # effv2 = EfficienceNet(num_class=4, model_name='effv2s', )

    # train
    # train(resnet34, train_loader, val_loader, num_epochs = 10, name_wtg='resnet34',device)
    # train(resnet34_l, train_loader, val_loader, num_epochs=20, name_wtg='resnet34_l', device) # upp num_epochs ?
    # train(resnet50, train_loader, val_loader, name_wtg='resnet50_10epoch', device)
    # train(resnet18, train_loader, val_loader, name_wtg='resnet18', device)

    # Evaluation
    # model1 =  ResNet(num_class=4, model_name='ResNet34', trainable_layers=['layer4', 'fc'])
    whg_early_s = 'C:/Users/julie/OneDrive/Bureau/Sarah/Projets_Python/Medical_segmentation_classification/model/clf/best_model_resnet34_epoch6_early_stop.pth'
    resnet34.load_state_dict(torch.load(whg_early_s))
    evaluation(resnet34, test_loader, classes_dico, device)

    whg1 = 'C:/Users/julie/OneDrive/Bureau/Sarah/Projets_Python/Medical_segmentation_classification/model/clf/best_model_resnet34_25epoch.pth'
    resnet34.load_state_dict(torch.load(whg1))
    print('/n', '****** RESNET 34 ******')
    evaluation(resnet34, test_loader, classes_dico, device)

    whg2 = 'C:/Users/julie/OneDrive/Bureau/Sarah/Projets_Python/Medical_segmentation_classification/model/clf/best_model_resnet50_25epoch.pth'
    resnet50.load_state_dict(torch.load(whg2))
    print('/n', '****** RESNET 50 ******')
    evaluation(resnet50, test_loader, classes_dico, device)

    # whg3 = 'C:/Users/julie/OneDrive/Bureau/Sarah/Projets_Python/Medical_segmentation_classification/model/clf/best_model_resnet34_25epoch.pth'
    # resnet18.load_state_dict(torch.load(whg3))
    # print('/n', '****** RESNET 18 ******')
    # evaluation(resnet18, test_loader)
