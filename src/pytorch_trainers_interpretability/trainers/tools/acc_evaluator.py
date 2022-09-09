import torch
from tqdm import tqdm
from ...attack import Attacker

class AccEvaluator:
    def __init__(self, model, criterion, testloader, adv_step, adv_iter, adv_eps, normalizer=None):
        self.model = model
        self.criterion = criterion
        self.testloader = testloader
        self.attacker = Attacker(self.model, num_iter=adv_iter, epsilon=adv_eps, attack_step=adv_step)
        self.normalizer = normalizer
    def eval_nat(self):
        self.model.eval()
        accuracy = 0.0
        total = 0.0
        running_loss = 0.0
        with torch.no_grad():
            with tqdm(self.testloader, unit="batch") as tepoch:
                for b, data in enumerate(tepoch):
                    tepoch.set_description(f"Val")
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
        return (100 * accuracy/total), (running_loss/(b+1))
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
        return (100 * accuracy/total), (running_loss/(b+1))
    
