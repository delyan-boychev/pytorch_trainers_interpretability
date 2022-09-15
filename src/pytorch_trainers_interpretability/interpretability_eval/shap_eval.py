import shap
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import torch
from ..attack import L2Step, Attacker
import matplotlib.pyplot as plt

class ShapEval:
    def __init__(self, model=None, testset=None, classes=None, transform=transforms.Compose([transforms.ToTensor()]), normalizer=None):
        if model is None or not isinstance(model, nn.Module):
            raise Exception("Invalid model")
        if testset is None or not isinstance(testset, data.Dataset):
            raise Exception("Invalid dataset")
        if classes is None or not isinstance(classes, list):
            raise Exception("Invalid classes")
        if not isinstance(transform, transforms.Compose):
            raise Exception("Invalid transform")
        if normalizer is not None:
            if not isinstance(normalizer, transforms.Normalize):
                raise Exception("Invalid normalizer")
        self.classes = classes
        self.dataset = testset
        self.transform = transform
        self.dataset.transforms = transform
        self.normalizer = normalizer
        self.dataloader = data.DataLoader(self.dataset, 200, shuffle=True)
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def _shap_deep_explain(self, background, X):
        ex = shap.DeepExplainer(self.model, background)
        return ex.shap_values(X)
    def nat_deep_exp(self, num_images):
        batch = next(iter(self.dataloader))
        X, y = batch
        background = X[:100].to(self.device)
        test_X = X[100:100+num_images]
        X_viz = test_X
        if self.normalizer is not None:
            test_X = self.normalizer(test_X)
        pr = (-self.model(test_X.to(self.device))).argsort(1).cpu().numpy()
        pr_labels = []
        for i in range(pr.shape[0]):
            pr_labels.append([])
            for j in range(pr.shape[1]):
                pr_labels[i].append(self.classes[pr[i][j]])
        shap_values = self._shap_deep_explain(background, test_X)
        shap_numpy = [np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values[:4]]
        test_numpy = (np.swapaxes(np.swapaxes(X_viz.numpy(), 1, -1), 1, 2)  * 255).astype(np.uint8)
        shap.plots.image(shap_numpy, test_numpy, true_labels=[self.classes[i] for i in y[100:100+num_images].cpu().numpy()], labels=np.array(pr_labels))
    def adv_deep_exp(self, num_images, attack_step=L2Step, num_iter=20, epsilon=0.5):
        attacker = Attacker(self.model, num_iter=num_iter, epsilon=epsilon, attack_step=attack_step, normalizer=self.normalizer)
        batch = next(iter(self.dataloader))
        X, y = batch
        background = attacker(X[:100], y[:100]).to(self.device)
        test_X = attacker(X[100:100+num_images], y[100:100+num_images]).cpu()
        X_viz = test_X
        if self.normalizer is not None:
            test_X = self.normalizer(test_X)
        pr = (-self.model(test_X.to(self.device))).argsort(1).cpu().numpy()
        pr_labels = []
        for i in range(pr.shape[0]):
            pr_labels.append([])
            for j in range(pr.shape[1]):
                pr_labels[i].append(self.classes[pr[i][j]])
        shap_values = self._shap_deep_explain(background, test_X)
        shap_numpy = np.abs([np.swapaxes(np.swapaxes(s, 1, -1), 1, 2) for s in shap_values[:4]]).mean(0)
        test_numpy = (np.swapaxes(np.swapaxes(X_viz.numpy(), 1, -1), 1, 2)  * 255).astype(np.uint8)

        shap.plots.image(shap_values, test_numpy, true_labels=[self.classes[i] for i in y[100:100+num_images].cpu().numpy()], labels=np.array(pr_labels))
