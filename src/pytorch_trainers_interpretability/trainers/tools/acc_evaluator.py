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
    def accuracy(self, output, target, topk=(1,)):
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)
            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = (pred == target.unsqueeze(dim=0)).expand_as(pred)

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                res.append(correct_k.mul_(100 / batch_size).item())
            return res
    def eval_nat(self):
        self.model.eval()
        accuracy_top1 = 0.0
        accuracy_top2 = 0.0
        accuracy_top5 = 0.0
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
                    accs = self.accuracy(outputs, y, topk=(1, 2, 5))
                    accuracy_top1 += accs[0]
                    accuracy_top2 += accs[1]
                    accuracy_top5 += accs[2]
                    running_loss += loss.item()
                    tepoch.set_postfix(loss=(running_loss/(b+1)), accuracy_top1=accuracy_top1/(b+1), accuracy_top2=accuracy_top2/(b+1), accuracy_top5=accuracy_top5/(b+1))
        return (accuracy_top1/(b+1)), (running_loss/(b+1))
    def eval_adv(self):
        self.model.eval()
        accuracy_top1 = 0.0
        accuracy_top2 = 0.0
        accuracy_top5 = 0.0
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
                accs = self.accuracy(outputs, y, topk=(1, 2, 5))
                accuracy_top1 += accs[0]
                accuracy_top2 += accs[1]
                accuracy_top5 += accs[2]
                running_loss += loss.item()
                tepoch.set_postfix(loss=(running_loss/(b+1)), accuracy_top1=accuracy_top1/(b+1), accuracy_top2=accuracy_top2/(b+1), accuracy_top5=accuracy_top5/(b+1))
        return (accuracy_top1/(b+1)), (running_loss/(b+1))
    
