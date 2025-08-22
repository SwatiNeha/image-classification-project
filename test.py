import torch
import torch.nn as nn
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import warnings
warnings.filterwarnings("ignore")


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)

model.load_state_dict(torch.load('model.pth', map_location=device))

model = model.to(device)

criterion = nn.CrossEntropyLoss()

test_transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5473, 0.4103, 0.3312], std=[0.2913, 0.2981, 0.2977]) # Normalization values used for pre-trained models
])

test_dataset = datasets.ImageFolder(root='./testdata', transform=test_transform)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

class_names = ['cherry', 'tomato', 'strawberry']

def test_model(model, test_loader, criterion, class_names):
    model.eval() 
    test_loss = 0.0
    correct_test = 0
    total_test = 0

    class_correct = [0 for _ in range(len(class_names))]
    class_total = [0 for _ in range(len(class_names))]

    with torch.no_grad(): 
        for images, labels in test_loader:

            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item() * images.size(0)

            _, predicted = torch.max(outputs, 1)

            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()

            for i in range(labels.size(0)):
                label = labels[i].item()
                pred = predicted[i].item()
                if pred == label:
                    class_correct[label] += 1
                class_total[label] += 1

    avg_test_loss = test_loss / total_test
    test_accuracy = correct_test / total_test

    print(f"Test Loss: {avg_test_loss:.4f}, Overall Test Accuracy: {test_accuracy:.4f}")

    for i, class_name in enumerate(class_names):
        if class_total[i] > 0:  # Avoid division by zero
            class_accuracy = class_correct[i] / class_total[i]
            print(f'Accuracy of {class_name}: {class_accuracy:.4f} ({class_correct[i]}/{class_total[i]})')

    return avg_test_loss, test_accuracy

avg_test_loss, test_accuracy = test_model(model, test_loader, criterion, class_names)