import shap
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import torch
from ..attack import L2Step, Attacker
import matplotlib.pyplot as plt

class ShapEval:
    def __init__(self, model=None, classes=None, normalizer=lambda x: x):
        if model is None or not isinstance(model, nn.Module):
            raise Exception("Invalid model")
        if classes is None or not isinstance(classes, list):
            raise Exception("Invalid classes")
        self.classes = classes
        self.normalizer = normalizer
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        transform= [
        transforms.Lambda(self.nhwc_to_nchw),
        transforms.Lambda(self.normalizer),
        transforms.Lambda(self.nchw_to_nhwc),
        ]
        inv_transform= [
            transforms.Lambda(self.nhwc_to_nchw),
            transforms.Normalize(
                mean = (-1 * np.array(self.normalizer.mean) / np.array(self.normalizer.std)).tolist(),
                std = (1 / np.array(self.normalizer.std)).tolist()
            ),
            transforms.Lambda(self.nchw_to_nhwc),
        ]
        self.transform = transforms.Compose(transform)
        self.inv_transform = transforms.Compose(inv_transform)
        self.model.to(self.device)
    def nhwc_to_nchw(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x if x.shape[1] == 3 else x.permute(0, 3, 1, 2)
        elif x.dim() == 3:
            x = x if x.shape[0] == 3 else x.permute(2, 0, 1)
        return x

    def nchw_to_nhwc(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            x = x if x.shape[3] == 3 else x.permute(0, 2, 3, 1)
        elif x.dim() == 3:
            x = x if x.shape[2] == 3 else x.permute(1, 2, 0)
        return x
    def _shap_part_explain(self, X):
        def predict(img: np.ndarray) -> torch.Tensor:
            img = self.nhwc_to_nchw(torch.Tensor(img))
            img = img.to(self.device)
            output = self.model(img)
            return output
        print(X[0].shape)
        masker = shap.maskers.Image("blur(128, 128)", X[0].shape)
        explainer = shap.Explainer(predict, masker, output_names=self.classes)
        return explainer(X, max_evals=1000, batch_size=50, outputs=shap.Explanation.argsort.flip[:4])
    def nat_part_exp(self, images, labels):
        images = self.transform(images)
        shap_values = self._shap_part_explain(images)
        shap_values.data = self.inv_transform(shap_values.data).cpu().numpy()
        shap_values.values = [val for val in np.moveaxis(shap_values.values,-1, 0)]
        shap.image_plot(shap_values=shap_values.values,
                pixel_values=shap_values.data,
                labels=shap_values.output_names,
                true_labels=labels)
    def _shap_deep_explain(self, background, X):
        ex = shap.DeepExplainer(self.model, background)
        return ex.shap_values(X, ranked_outputs=4)
    def nat_deep_exp(self, images_background, eval_images, labels):
        background = images_background.to(self.device)
        test_X = eval_images.to(self.device)
        X_viz = test_X
        test_X = self.normalizer(test_X)
        shap_values, indexes = self._shap_deep_explain(background, test_X)
        shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]
        test_numpy = (np.swapaxes(np.swapaxes(X_viz.cpu().numpy(), 1, -1), 1, 2)  * 255).astype(np.uint8)
        index_names = np.vectorize(lambda x: self.classes[x])(indexes.cpu())
        return shap.plots.image(shap_values, test_numpy, true_labels=[self.classes[i] for i in labels.cpu().numpy()], labels=index_names, show=False)
    def adv_deep_exp(self, background_images, backgeound_labels, eval_images, eval_labels, attack_step=L2Step, num_iter=20, epsilon=0.5):
        attacker = Attacker(self.model, num_iter=num_iter, epsilon=epsilon, attack_step=attack_step, normalizer=self.normalizer)
        background = attacker(background_images, backgeound_labels).to(self.device)
        test_X = attacker(eval_images, eval_labels).cpu()
        X_viz = test_X
        test_X = self.normalizer(test_X)
        shap_values, indexes = self._shap_deep_explain(background, test_X)
        shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1) for s in shap_values]
        test_numpy = (np.swapaxes(np.swapaxes(X_viz.cpu().numpy(), 1, -1), 1, 2)  * 255).astype(np.uint8)
        index_names = np.vectorize(lambda x: self.classes[x])(indexes.cpu())
        return shap.plots.image(shap_values, test_numpy, true_labels=[self.classes[i] for i in eval_labels.cpu().numpy()], labels=index_names, show=False)
