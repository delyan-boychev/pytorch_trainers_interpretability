import torch
from tqdm import tqdm
from ...attack import Attacker

class AccEvaluator:
    def __init__(self, model, criterion, testloader, device, adv_step, adv_iter, adv_eps, normalizer=lambda x: x):
        self.model = model
        self.criterion = criterion
        self.testloader = testloader
        self.attacker = Attacker(self.model, num_iter=adv_iter, epsilon=adv_eps, attack_step=adv_step, tqdm=False)
        self.normalizer = normalizer
        self.attacker.normalizer = self.normalizer
        self.device = device
    def eval_nat(self):
        self.model.eval()
        accuracy = 0.0
        total = 0.0
        running_loss = 0.0
        with torch.no_grad():
            with tqdm(self.testloader, unit="batch") as tepoch:
                for b, data in enumerate(tepoch):
                    tepoch.set_description(f"Val")
                    X, y = data
                    X = X.to(self.device)
                    X = self.normalizer(X)
                    y = y.to(self.device)
                    outputs = self.model(X)
                    loss = self.criterion(outputs, y)
                    _, predicted = torch.max(outputs.data, 1)
                    total += y.size(0)
                    accuracy += (predicted == y).sum().item()
                    running_loss += loss.item()
                    tepoch.set_postfix(loss=(running_loss/(b+1)), accuracy=(100 * accuracy/total))
        return (100 * accuracy/total), (running_loss/(b+1))
    def eval_adv(self):
        self.model.eval()
        accuracy = 0.0
        total = 0.0
        running_loss = 0.0
        with tqdm(self.testloader, unit="batch") as tepoch:
            tepoch.set_description(f"Adv Val")
            for b, data in enumerate(tepoch):
                X, y = data
                X = X.to(self.device)
                y = y.to(self.device)
                adv_ex = self.attacker(X, y)
                adv_ex = self.normalizer(adv_ex)
                outputs = self.model(adv_ex)
                loss = self.criterion(outputs, y)
                _, predicted = torch.max(outputs.data, 1)
                total += y.size(0)
                accuracy += (predicted == y).sum().item()
                running_loss += loss.item()
                tepoch.set_postfix(loss=(running_loss/(b+1)), accuracy=(100 * accuracy/total))
        return (100 * accuracy/total), (running_loss/(b+1))
    
