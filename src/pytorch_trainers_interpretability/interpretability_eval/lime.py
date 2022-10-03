from skimage.segmentation import mark_boundaries
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np


class LimeEval:
    def __init__(self, model, transform_pil, normalizer=None):
        self.model = model
        self.normalizer = normalizer
        self.explainer = lime_image.LimeImageExplainer(verbose = False)
        self.segmenter = SegmentationAlgorithm('quickshift', n_segments=200, compactness=1, sigma=1)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    def _predict(self, input):
        self.model.eval()
        batch = torch.stack(tuple(self.normalizer(transforms.ToTensor()(i.astype(np.float32))) for i in input), dim=0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        batch = batch.to(self.device)
        logits = self.model(batch)
        probs = nn.functional.softmax(logits, dim=1)
        return probs.detach().cpu().numpy()
    def explain_model(self, image):
            image = image.astype(np.double)
            explanation = self.explainer.explain_instance(image, 
                                                    self._predict,
                                                    top_labels=1, 
                                                    hide_color=0, 
                                                    num_samples=1000, segmentation_fn=self.segmenter)
            temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False)
            return mark_boundaries(temp, mask, mode="outer")

