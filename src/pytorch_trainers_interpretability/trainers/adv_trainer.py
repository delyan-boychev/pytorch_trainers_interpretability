import torch.nn as nn
import torch
import timm
from ..attack import Attacker, AttackSteps
from tqdm import tqdm
from torchvision import transforms
from torch.utils import data
import functools
from .tools import SaveInfo, Backward, AccEvaluator
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


class AdversarialTrainer:
    def __init__(self, model="resnet18", pretrained=False, criterion=nn.CrossEntropyLoss(),
                 num_classes=10, lr=0.001, epochs=100,
                 adv_step=AttackSteps.L2Step, adv_iter=20, adv_epsilon=0.5, adv_lr=0.01,
                 optimizer="Adam", lr_scheduler=None, weight_decay=0.0,
                 trainset=None, testset=None, batch_size=20,
                 transforms_train=transforms.Compose([transforms.ToTensor()]), transforms_test=transforms.Compose([transforms.ToTensor()]),
                 input_normalizer=lambda x: x,
                 resume_path=None, save_plot=True, save_path="./"):
        if isinstance(model, str):
            self.model = timm.create_model(
                model_name=model, pretrained=pretrained, num_classes=num_classes)
        elif isinstance(model, nn.Module):
            self.model = model
        else:
            raise Exception("Invalid model")
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        if not isinstance(criterion, nn.modules.loss._Loss):
            raise Exception("Invalid criterion")
        self.criterion = criterion
        if not isinstance(optimizer, str) or optimizer not in optimizers:
            raise Exception("Invalid optimizer")
        self.optimizer = optimizers[optimizer](
            params=self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = None
        if lr_scheduler is not None:
            if not isinstance(lr_scheduler, str) or lr_scheduler not in schedulers:
                raise Exception("Invalid learning rate scheduler")
            self.scheduler = schedulers[lr_scheduler](optimizer=self.optimizer)
        if not os.path.exists(save_path):
            raise Exception("Save path not exists")
        self.save_path = save_path
        self.epochs = epochs
        self.epoch = 0
        self.loss = 0
        self.save_plot = save_plot
        self.save_info = SaveInfo(
            save_path=self.save_path, resume_path=resume_path, adv_train=True)
        self.attacker = Attacker(model=self.model, epsilon=adv_epsilon,
                                 attack_step=adv_step, num_iter=adv_iter, lr=adv_lr, tqdm=False)
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
        self.trainloader = data.DataLoader(
            dataset=trainset, batch_size=batch_size, shuffle=True, num_workers=2)
        self.testloader = data.DataLoader(
            dataset=testset, batch_size=batch_size, shuffle=True, num_workers=2)
        self.attacker.normalizer = input_normalizer
        self.normalizer = input_normalizer
        self.acc_eval = AccEvaluator(model=self.model, criterion=self.criterion, device=self.device, testloader=self.testloader,
                                     adv_step=adv_step, adv_lr=adv_lr, adv_iter=adv_iter, adv_eps=adv_epsilon, normalizer=self.normalizer)
        self.backward = Backward(self.model, self.criterion, self.optimizer)
        print(f"Model created on device {self.device}")
        if resume_path is not None:
            checkpoint = torch.load(os.path.join(
                resume_path, "checkpoint.pt"), map_location=self.device)
            self.save_info.load_train_info(checkpoint['epoch'])
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if "lr_scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(
                    checkpoint["lr_scheduler_state_dict"])
            print(f"Model resumed: Epoch {checkpoint['epoch']}")
            self.epoch = checkpoint['epoch']+1

    def eval_nat(self):
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
                    y = y.to(self.device)
                    adv_ex = self.attacker(X, y)
                    adv_ex = self.normalizer(adv_ex)
                    curr_loss, curr_acc, length = self.backward(adv_ex, y)
                    running_loss += curr_loss
                    accuracy += curr_acc
                    total += length
                    tepoch.set_postfix(
                        loss=(running_loss/(b+1)), accuracy=(100 * accuracy/total))
            acc_test, loss_test = self.eval_nat()
            acc_test_adv, loss_test_adv = self.eval_adv()
            self.save_info.append_test(
                acc_test, loss_test, acc_test_adv, loss_test_adv)
            self.save_info.append_train(
                (100 * accuracy/total), (running_loss/(b+1)))
            if self.scheduler is not None:
                self.scheduler.step()
            lr_scheduler_state_dict = self.scheduler.state_dict(
            ) if self.scheduler is not None else None
            self.save_info.save_model(self.model.state_dict(), i, (running_loss/(
                b+1)), self.optimizer.state_dict(), lr_scheduler_state_dict=lr_scheduler_state_dict, best=False)
            if self.save_info.to_save_model:
                self.save_info.save_model(self.model.state_dict(), i, (running_loss/(
                    b+1)), self.optimizer.state_dict(), lr_scheduler_state_dict=lr_scheduler_state_dict, best=True)
            self.save_info.save_train_info()
            if self.save_plot is True:
                self.save_info.save_acc_plot()
                self.save_info.save_loss_plot()
