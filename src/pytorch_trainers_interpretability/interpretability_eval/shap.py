import shap
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import torch
from ..attack import L2Step, Attacker
import matplotlib.pyplot as plt
from ._visualization_shap import image_plot, image_plot_single
from ..models import BatchPredictor


class ShapEval:
    def __init__(self, model=None, classes=None, normalizer=lambda x: x):
        if model is None or not isinstance(model, nn.Module):
            raise Exception("Invalid model")
        if classes is None or not isinstance(classes, list):
            raise Exception("Invalid classes")
        self.classes = classes
        self.normalizer = normalizer
        self.model = model
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        transform = [
            transforms.Lambda(self.nhwc_to_nchw),
            transforms.Lambda(self.normalizer),
            transforms.Lambda(self.nchw_to_nhwc),
        ]
        inv_transform = [
            transforms.Lambda(self.nhwc_to_nchw),
            transforms.Normalize(
                mean=(-1 * np.array(self.normalizer.mean) /
                      np.array(self.normalizer.std)).tolist(),
                std=(1 / np.array(self.normalizer.std)).tolist()
            ),
            transforms.Lambda(self.nchw_to_nhwc),
        ]
        self.transform = transforms.Compose(transform)
        self.inv_transform = transforms.Compose(inv_transform)

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
        masker = shap.maskers.Image("inpaint", X[0].shape)
        explainer = shap.Explainer(predict, masker, output_names=self.classes)
        return explainer(X, max_evals=1000, batch_size=50, outputs=shap.Explanation.argsort.flip[:4])

    def part_exp(self, images, labels):
        self.model.to(self.device)
        images = self.transform(images)
        shap_values = self._shap_part_explain(images)
        shap_values.data = self.inv_transform(shap_values.data).cpu().numpy()
        shap_values.values = [
            val for val in np.moveaxis(shap_values.values, -1, 0)]
        shap.image_plot(shap_values=shap_values.values,
                        pixel_values=shap_values.data,
                        labels=shap_values.output_names,
                        true_labels=labels)

    def set_shap_gradient_explainer(self, background, batch_size=10):
        self.grad_ex = shap.GradientExplainer(BatchPredictor(
            self.model, batch_size), background, batch_size=batch_size)

    def _shap_gradient_explain(self, X):
        return self.grad_ex.shap_values(X, ranked_outputs=1)

    def gradient_exp_one(self, image):
        self.model.to(self.device)
        test_X = image.to(self.device)
        X_viz = test_X
        test_X = self.normalizer(test_X)
        shap_values, indexes = self._shap_gradient_explain(test_X)
        shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1)
                       for s in shap_values]
        return image_plot_single(shap_values, X_viz.cpu().numpy().transpose(0, 2, 3, 1))

    def gradient_exp(self, images_background, eval_images, labels, batch_size=10):
        self.model.to(self.device)
        background = self.normalizer(images_background.to(self.device))
        test_X = eval_images.to(self.device)
        X_viz = test_X
        test_X = self.normalizer(test_X)
        self.set_shap_gradient_explainer(background, batch_size=batch_size)
        shap_values, indexes = self._shap_gradient_explain(test_X)
        index_names = np.vectorize(lambda x: self.classes[x])(indexes.cpu())
        shap_values = [np.swapaxes(np.swapaxes(s, 2, 3), 1, -1)
                       for s in shap_values]
        image_plot(shap_values, X_viz.cpu().numpy().transpose(
            0, 2, 3, 1), index_names,  true_labels=[self.classes[i] for i in labels.cpu().numpy()])
