import torch.nn as nn
import torch
import timm
from ..attack import Attacker, AttackSteps
from tqdm import tqdm
from torchvision import transforms
from torch.utils import data
import functools

optimizers = {
    "Adam": functools.partial(torch.optim.Adam),
    "SGD": functools.partial(torch.optim.SGD, momentum=0.9)
}
schedulers = {
    "CosineAnnealingLR": functools.partial(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=200),
    "ExponentialLR": functools.partial(torch.optim.lr_scheduler.ExponentialLR, gamma=0.9),
    "StepLR": functools.partial(torch.optim.lr_scheduler.StepLR, step_size=20, gamma=0.1)
}

class AdversarialTrainer:
    def __init__(self, model="resnet18", pretrained=False, criterion=nn.CrossEntropyLoss(), num_classes=10, lr=0.001, epochs=100, adv_step=AttackSteps.L2Step, adv_iter=20, adv_epsilon=0.5, optimizer="Adam", lr_scheduler=None, weight_decay=0.0, trainset=None, testset=None, batch_size=20, transforms_train=transforms.Compose([transforms.ToTensor()]), transforms_test=transforms.Compose([transforms.ToTensor()]), input_normalizer=None, resume_path=None, save_path="./"):
        if isinstance(model, str):
            self.model = timm.create_model(model_name=model, pretrained=pretrained, num_classes=num_classes)
        elif isinstance(model, nn.Module):
            self.model = model
        else:
            raise Exception("Non valid model")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        if not isinstance(criterion, nn.modules.loss._Loss):
            raise Exception("Non valid criterion")
        self.criterion = criterion
        if not isinstance(optimizer, str) or optimizer not in optimizers:
            raise Exception("Non valid optimizer")
        self.optimizer = optimizers[optimizer](params=self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = None
        if lr_scheduler is not None:
            if not isinstance(lr_scheduler, str) or lr_scheduler not in schedulers:
                raise Exception("Non valid learning rate scheduler")
            self.scheduler = schedulers[lr_scheduler](optimizer=self.optimizer)
        self.save_path = save_path
        self.epochs = epochs
        self.epoch=0
        self.loss = 0
        self.attacker = Attacker(model=self.model, epsilon=adv_epsilon, attack_step=adv_step, num_iter=adv_iter, tqdm=False)
        if not isinstance(trainset, data.Dataset):
            raise Exception("Not valid train loader")
        if not isinstance(testset, data.Dataset):
            raise Exception("Not valid train loader")
        if not isinstance(transforms_train, transforms.Compose):
            raise Exception("Not valid data transforms")
        if not isinstance(transforms_test, transforms.Compose):
            raise Exception("Not valid data transforms")
        trainset.transform = transforms_train
        testset.transform = transforms_test
        self.trainloader = data.DataLoader(dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        self.testloader = data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=True, num_workers=2)
        if input_normalizer is not None:
            if not isinstance(input_normalizer, transforms.Normalize):
                raise Exception("Non valid input normalizer")
            else:
                self.attacker.normalizer = input_normalizer
        self.normalizer = input_normalizer
        print(f"Model created on device {self.device}")
        if resume_path is not None:
            checkpoint = torch.load(resume_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model resumed: Epoch {checkpoint['epoch']}")
            self.epoch = checkpoint['epoch']+1
    def eval_nat(self):
        self.model.eval()
        accuracy = 0.0
        total = 0.0
        running_loss = 0.0
        with torch.no_grad():
            with tqdm(self.testloader, unit="batch") as tepoch:
                for b, data in enumerate(tepoch):
                    tepoch.set_description(f"Nat Val")
                    images, labels = data
                    images = images.to(self.device)
                    if self.normalizer is not None:
                        images = self.normalizer(images)
                    labels = labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    accuracy += (predicted == labels).sum().item()
                    running_loss += loss.item()
                    tepoch.set_postfix(loss=(running_loss/(b+1)), accuracy=(100 * accuracy/total))
            return (running_loss/total)
    def eval_adv(self):
        self.model.eval()
        accuracy = 0.0
        total = 0.0
        running_loss = 0.0
        with tqdm(self.testloader, unit="batch") as tepoch:
            for b, data in enumerate(tepoch):
                tepoch.set_description(f"Adv Val")
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                adv_ex = self.attacker(images, labels)
                if self.normalizer is not None:
                    adv_ex = self.normalizer(adv_ex)
                outputs = self.model(adv_ex)
                loss = self.criterion(outputs, labels)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                accuracy += (predicted == labels).sum().item()
                running_loss += loss.item()
                tepoch.set_postfix(loss=(running_loss/(b+1)), accuracy=(100 * accuracy/total))
        return (running_loss/total)
    def get_model(self):
        return self.model
    def __call__(self):
        best_acc = -10
        for i in range(self.epoch, self.epochs):
            self.model.train()
            running_loss = 0.0
            accuracy = 0.0
            total = 0
            with tqdm(self.trainloader, unit="batch") as tepoch:
                for b, (X, y) in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {i}")
                    X = X.to(self.device)
                    y = y.to(self.device)
                    self.optimizer.zero_grad()
                    adv_ex = self.attacker(X, y)
                    if self.normalizer is not None:
                        adv_ex = self.normalizer(adv_ex)
                    outputs = self.model(adv_ex)
                    loss = self.criterion(outputs, y)
                    predictions = outputs.argmax(dim=1, keepdim=True).squeeze()
                    total += y.size(0)
                    accuracy += (predictions == y).sum().item()
                    running_loss += loss.item()
                    loss.backward()
                    self.optimizer.step()
                    tepoch.set_postfix(loss=(running_loss/(b+1)), accuracy=(100 * accuracy/total))
            acc1 = self.eval_nat()
            acc2 = self.eval_adv()
            acc = (acc1 + acc2)/2
            if self.scheduler is not None :
                self.scheduler.step()
            if acc > best_acc:
                best_acc = acc
                torch.save({
                        'epoch': i,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'loss': running_loss,
                        }, f"{self.save_path}/best.pt")
        