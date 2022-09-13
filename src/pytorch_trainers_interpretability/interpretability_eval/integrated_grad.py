import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import matplotlib.pylab as plt
from tqdm import tqdm

class IntegratedGrad:
    def __init__(self, model, transform=transforms.Compose([transforms.ToTensor()]), normalizer=None):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transform = transform
        if normalizer is not None:
            if not isinstance(normalizer, transforms.Normalize):
                raise Exception("Invalid normalizer")
        self.normalizer = normalizer
    def _pre_processing(self, inp):
        inp = np.array(inp)
        inp = np.transpose(inp, (-1, 0, 1))
        inp_tensor = torch.from_numpy(inp).float().unsqueeze(0).to(self.device)
        if self.normalizer is not None:
            inp_tensor = self.normalizer(inp_tensor)
        return inp_tensor
    def _pred_grad(self, input, target_label_idx):
        gradients = []
        for inp in input:
            inp = self._pre_processing(inp).requires_grad_(True)
            output = self.model(inp)
            output = F.softmax(output, dim=1)
            index = torch.ones((output.shape[0], 1), dtype=torch.int64) * target_label_idx
            index = index.to(self.device)
            output = output.gather(1, index)
            output.backward()
            gradient = inp.grad[0]
            gradients.append(gradient)
        return torch.cat(gradients)
    def _generate_staturate_batches(self, input, baseline, steps):
        return [baseline + (float(i) / steps) * (input - baseline) for i in range(0, steps + 1)]
    def _integrated_grads(self, input, target_label_idx, baseline, steps=50):
        scaled_inputs = self._generate_staturate_batches(input, baseline, steps)
        grads = self._pred_grad(scaled_inputs,target_label_idx)
        avg_grads = torch.mean(grads[:-1], axis=0)
        delta_X = (self._pre_processing(input) - self._pre_processing(baseline)).detach().squeeze(0)
        integrated_grad = (delta_X * avg_grads).detach().cpu().numpy()
        integrated_grad = np.transpose(integrated_grad, (1, 2, 0))
        return integrated_grad
    def get_attribution_mask(self, integrated_grad):
        return np.sum(np.abs(integrated_grad), axis=-1)
    def zero_baseline_integrated_grads(self, input, target_label_idx, steps=50):
        baseline = np.zeros(input.shape)
        return self._integrated_grads(input, target_label_idx, baseline, steps)
    def gausssian_noise_integrated_grads(self, input, target_label_idx, steps, num_random_trials, std=0.31622776601):
        all_intgrads = []

        itr = tqdm(range(num_random_trials), unit="trial")
        for i in itr:
            integrated_grad = self._integrated_grads(input, target_label_idx, \
                                                    baseline=np.random.normal(0, std, input.shape), steps=steps)
            all_intgrads.append(integrated_grad)
        avg_intgrads = np.average(np.array(all_intgrads), axis=0)
        return avg_intgrads
    def uniform_baseline_integrated_grads(self, input, target_label_idx, steps, num_random_trials):
        all_intgrads = []

        itr = tqdm(range(num_random_trials), unit="trial")
        for i in itr:
            integrated_grad = self._integrated_grads(input, target_label_idx, \
                                                    baseline=np.random.uniform(input.shape), steps=steps)
            all_intgrads.append(integrated_grad)
        avg_intgrads = np.average(np.array(all_intgrads), axis=0)
        return avg_intgrads
    def random_baseline_integrated_grads(self, input, target_label_idx, steps, num_random_trials):
        all_intgrads = []

        itr = tqdm(range(num_random_trials), unit="trial")
        for i in itr:
            integrated_grad = self._integrated_grads(input, target_label_idx, \
                                                    baseline=np.random.random(input.shape), steps=steps)
            all_intgrads.append(integrated_grad)
        avg_intgrads = np.average(np.array(all_intgrads), axis=0)
        return avg_intgrads