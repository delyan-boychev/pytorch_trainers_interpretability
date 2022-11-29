# ---------------------------------------------------------------------------- #
# An implementation of https://arxiv.org/pdf/1703.01365.pdf                    #
# and https://arxiv.org/abs/1908.06214                                         #
# ---------------------------------------------------------------------------- #
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pylab as plt
from tqdm import tqdm
from ._visualization import Visualize, pil_image

class IntegratedGrad:
    def __init__(self, model, normalizer=lambda x: x):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.normalizer = normalizer
    def _to_tensor(self, inp):
        inp = np.array(inp)
        inp = np.transpose(inp, (-1, 0, 1))
        inp_tensor = torch.from_numpy(inp).float().unsqueeze(0).to(self.device)
        return inp_tensor
    def _pred_grad(self, input, target_label_idx):
        inp = input.clone().requires_grad_(True)
        outputs = self.model(inp)
        outputs = F.softmax(outputs, dim=-1)[:, target_label_idx]
        outputs.backward(torch.ones(outputs.shape).to(self.device))
        gradients = inp.grad.detach().clone()
        return gradients, outputs.detach().clone().cpu().numpy()
    def _generate_staturate_batches(self, input, baseline, steps):
        alphas = torch.linspace(0.0, 1.0, steps+1).to(self.device)
        alphas = alphas[:, None, None, None]
        images = baseline + alphas*(input-baseline)
        return images
    def _integrated_grads(self, input, target_label_idx, baseline, steps=50, batch_size=30):
        scaled_inputs = self._generate_staturate_batches(input, baseline, steps)
        gradients = []
        predictions = []
        for i in range(0, steps, batch_size):
            start = i
            end = min(start+batch_size, len(scaled_inputs))
            batch = scaled_inputs[start:end]
            batch = self.normalizer(batch)
            gradient, pred = self._pred_grad(batch, target_label_idx)
            predictions.append(pred)
            gradients.append(gradient)
        predictions = np.hstack(predictions)
        gradients = torch.cat(gradients, axis=0)
        input = self.normalizer(input)
        baseline = self.normalizer(baseline)
        gradients = (gradients[:-1] + gradients[1:]) / 2.0
        integrated_grads = torch.mean(gradients, axis=0)
        delta_X = (input - baseline).to(self.device)
        integrated_grads = (delta_X*integrated_grads).squeeze(0).detach().cpu().numpy()
        integrated_grads = np.transpose(integrated_grads, (1, 2, 0))
        completeness = np.abs(np.sum(integrated_grads) - (predictions[-1] - predictions[0]))
        return integrated_grads, completeness, predictions
    def visualization(self, grad, image, treshold=0):
        return pil_image(Visualize(grad, (image*255).astype(np.uint8), clip_below_percentile=treshold))
    def black_baseline_integrated_grads(self, input, target_label_idx, steps=50, batch_size=30):
        input = self._to_tensor(input)
        baseline = torch.zeros(input.shape).to(self.device)
        integrated_grad, cm = self._integrated_grads(input, target_label_idx, baseline, steps, batch_size)
        return integrated_grad
    def gaussian_noise_integrated_grads(self, input, target_label_idx, steps, num_random_trials, batch_size=30):
        input = self._to_tensor(input)
        all_intgrads = []
        itr = tqdm(range(num_random_trials), unit="trial")
        completeness = []
        preds = []
        for i in itr:
            integrated_grad, cm, predictions = self._integrated_grads(input, target_label_idx, \
                                                    baseline=torch.normal(0, 0.4, input.shape).to(self.device), steps=steps, batch_size=batch_size)
            preds.append(predictions)
            all_intgrads.append(integrated_grad)
            completeness.append(cm)
            itr.set_postfix(completeness=np.average(completeness))
        avg_intgrads = np.average(np.array(all_intgrads), axis=0)
        return avg_intgrads, preds
    def random_baseline_integrated_grads(self, input, target_label_idx, steps, num_random_trials, batch_size=30):
        input = self._to_tensor(input)
        all_intgrads = []
        itr = tqdm(range(num_random_trials), unit="trial")
        completeness = []
        preds = []
        for i in itr:
            integrated_grad, cm, predictions = self._integrated_grads(input, target_label_idx, \
                                                    baseline=torch.rand(input.shape).to(self.device), steps=steps, batch_size=batch_size)
            preds.append(predictions)
            all_intgrads.append(integrated_grad)
            completeness.append(cm)
            itr.set_postfix(completeness=np.average(completeness))
        avg_intgrads = np.average(np.array(all_intgrads), axis=0)
        return avg_intgrads, preds