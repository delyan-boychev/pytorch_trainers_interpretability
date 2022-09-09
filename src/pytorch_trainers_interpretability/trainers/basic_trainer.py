import torch.nn as nn
import torch
import functools
import timm
from ..attack import Attacker, AttackSteps
from tqdm import tqdm
from torchvision import transforms
from torch.utils import data
from .tools import Backward, SaveInfo, AccEvaluator
import os


optimizers = {
    "Adam": functools.partial(torch.optim.Adam),
    "SGD": functools.partial(torch.optim.SGD, momentum=0.9)
}
schedulers = {
    "CosineAnnealingLR": functools.partial(torch.optim.lr_scheduler.CosineAnnealingLR, T_max=200),
    "ExponentialLR": functools.partial(torch.optim.lr_scheduler.ExponentialLR, gamma=0.9),
    "StepLR": functools.partial(torch.optim.lr_scheduler.StepLR, step_size=20, gamma=0.1),
    "OneCycleLR": functools.partial(torch.optim.lr_scheduler.OneCycleLR, max_lr=0.01)
}
class BasicTrainer:
    def __init__(self, model="resnet18", pretrained=False, criterion=nn.CrossEntropyLoss(),
    num_classes=10, lr=0.001, epochs=100,
    adv_step=AttackSteps.L2Step, adv_iter=20, adv_epsilon=0.5,
    optimizer="Adam", lr_scheduler=None, weight_decay=0.0,
    trainset=None, testset=None, batch_size=20,
    transforms_train=transforms.Compose([transforms.ToTensor()]), transforms_test=transforms.Compose([transforms.ToTensor()]), input_normalizer=None,
    resume_path=None, save_plot=True, save_path="./"):
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
        if not os.path.exists(save_path):
            raise Exception("Save path not exists")
        self.save_path = save_path
        self.epochs = epochs
        self.epoch=0
        self.loss = 0
        self.save_plot = save_plot
        self.save_info = SaveInfo(self.save_path, adv_train=False)
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
        self.testloader = data.DataLoader(dataset=testset, batch_size=batch_size, shuffle=False, num_workers=2)
        if lr_scheduler is not None:
            if not isinstance(lr_scheduler, str) or lr_scheduler not in schedulers:
                raise Exception("Non valid learning rate scheduler")
            if lr_scheduler is "OneCycleLR":
                self.scheduler = schedulers[lr_scheduler](optimizer=self.optimizer, steps_per_epoch=len(self.trainloader), epochs=epochs)
            else:
                self.scheduler = schedulers[lr_scheduler](optimizer=self.optimizer)
        if input_normalizer is not None:
            if not isinstance(input_normalizer, transforms.Normalize):
                raise Exception("Non valid input normalizer")
        self.normalizer = input_normalizer
        self.acc_eval = AccEvaluator(model=self.model, criterion=self.criterion, device=self.device, testloader=self.testloader, adv_step=adv_step, adv_iter=adv_iter, adv_eps=adv_epsilon, normalizer=self.normalizer)
        self.backward = Backward(self.model, self.criterion, self.optimizer)
        print(f"Model created on device {self.device}")
        if resume_path is not None:
            checkpoint = torch.load(resume_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if "optimizer_state_dics" in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model resumed: Epoch {checkpoint['epoch']}")
            self.epoch = checkpoint['epoch']+1
    def eval(self):
        return self.acc_eval.eval_nat()
    def eval_adv(self):
        return self.acc_eval.eval_adv()
    def get_model(self):
        return self.model
    def __call__(self):
        for i in range(self.epoch, self.epochs):
            self.model.train()
            running_loss = 0.0
            accuracy = 0.0
            total = 0
            with tqdm(self.trainloader, unit="batch") as tepoch:
                tepoch.set_description(f"Epoch {i}")
                for b, (X, y) in enumerate(tepoch):
                    X = X.to(self.device)
                    if self.normalizer is not None:
                        X = self.normalizer(X)
                    y = y.to(self.device)
                    curr_loss, curr_acc, length = self.backward(X, y)
                    running_loss += curr_loss
                    accuracy += curr_acc
                    total += length
                    tepoch.set_postfix(loss=(running_loss/(b+1)), accuracy=(100 * accuracy/total))
                    if self.scheduler is not None:
                        if isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                            self.scheduler.step()
            self.save_info.append_test(self.eval())
            self.save_info.append_train((100 * accuracy/total), (running_loss/(b+1)))
            if self.scheduler is not None:
                if not isinstance(self.scheduler, torch.optim.lr_scheduler.OneCycleLR):
                    self.scheduler.step()
            if self.save_info.to_save_model:
                self.save_info.save_model((self.model.state_dict(), i, (running_loss/(b+1)), self.optimizer.state_dict()))
            self.save_info.save_train_info()    
        if self.save_plot  is True:
               self.save_info.save_acc_plot()
               self.save_info.save_loss_plot()
        