import torch.nn as nn
import torch
import timm
from tqdm import tqdm

class BasicTrainer:
    def __init__(self, model="resnet18", pretrained=False, criterion=nn.CrossEntropyLoss(), num_classes=10, lr=0.001, epochs=100, optimizer=torch.optim.Adam, weight_decay=0.0, trainloader=None, testloader=None, resume_path=None, save_path="./"):
        if isinstance(model, str):
            self.model = timm.create_model(model_name=model, pretrained=pretrained, num_classes=num_classes)
        elif isinstance(model, nn.Module):
            self.model = model
        else:
            raise Exception("Non valid model")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not isinstance(criterion, nn.modules.loss._Loss):
            raise Exception("Non valid criterion")
        self.criterion = criterion
        if not isinstance(optimizer, type) or not issubclass(optimizer, torch.optim.Optimizer):
            raise Exception("Non valid optimizer")
        self.optimizer = optimizer(params=self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.save_path = save_path
        self.epochs = epochs
        self.epoch=0
        self.loss = 0
        if not isinstance(trainloader, torch.utils.data.DataLoader):
            raise Exception("Not valid train loader")
        if not isinstance(testloader, torch.utils.data.DataLoader):
            raise Exception("Not valid train loader")
        self.trainloader = trainloader
        self.testloader = testloader
        self.model.to(self.device)
        print(f"Model created on device {self.device}")
        if resume_path is not None:
            checkpoint = torch.load(resume_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model resumed: Epoch {checkpoint['epoch']}")
            self.epoch = checkpoint['epoch']+1
    def eval(self):
        self.model.eval()
        accuracy = 0.0
        total = 0.0
        running_loss = 0.0
        with tqdm(self.testloader, unit="batch") as tepoch:
            for data in tepoch:
                tepoch.set_description(f"Val")
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                accuracy += (predicted == labels).sum().item()
                running_loss += loss.item()
                tepoch.set_postfix(loss=(running_loss/total), accuracy=(100 * accuracy/total))
        return (running_loss/total)
    def get_model(self):
        return self.model
    def __call__(self):
        self.model.train()
        for i in range(self.epoch, self.epochs):
            running_loss = 0.0
            accuracy = 0.0
            total = 0
            with tqdm(self.trainloader, unit="batch") as tepoch:
                for X, y in tepoch:
                    tepoch.set_description(f"Epoch {i}")
                    X = X.to(self.device)
                    y = y.to(self.device)
                    self.optimizer.zero_grad()
                    outputs = self.model(X)
                    loss = self.criterion(outputs, y)
                    predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
                    total += y.size(0)
                    accuracy += (predictions == y).sum().item()
                    running_loss += loss.item()
                    loss.backward()
                    self.optimizer.step()
                    tepoch.set_postfix(loss=(running_loss/total), accuracy=(100 * accuracy/total))
            val_loss = self.eval()
            torch.save({
                    'epoch': i,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': loss.item(),
                    }, f"{self.save_path}/checkpoint{i}.pt")
        