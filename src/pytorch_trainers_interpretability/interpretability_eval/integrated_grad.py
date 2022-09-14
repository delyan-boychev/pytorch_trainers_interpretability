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
            inp = inp.requires_grad_(True)
            output = self.model(inp)
            output = F.softmax(output, dim=1)[0, target_label_idx]
            output.backward()
            gradient = inp.grad
            gradients.append(gradient)
        return torch.cat(gradients)
    def _generate_staturate_batches(self, input, baseline, steps):
        return [baseline + a * (input - baseline) for a in np.linspace(0, 1, steps)]
    def _integrated_grads(self, input, target_label_idx, baseline, steps=50):
        scaled_inputs = self._generate_staturate_batches(input, baseline, steps)
        grads = self._pred_grad(scaled_inputs,target_label_idx)
        integrated_grads= torch.mean(grads[:-1], axis=0)
        delta_X = (input - baseline).detach().squeeze(0)
        integrated_grads = (delta_X * integrated_grads).detach().cpu().numpy()
        integrated_grads = np.transpose(integrated_grads, (1, 2, 0))
        return integrated_grads
    def _get_attribution_mask(self, integrated_grads):
        grad_arr = np.average(np.abs(integrated_grads), axis=-1)
        grad_arr /= np.quantile(grad_arr, 0.98)
        grad_arr = np.clip(grad_arr, 0, 1)
        return grad_arr
    def img_attributions(self, grad, image, treshhold=0):
        grad = self._get_attribution_mask(grad)
        mask = np.zeros([image.shape[0], image.shape[1], 3])
        mask[:, :, 1] = grad
        mask[mask<treshhold] = 0
        overlay = np.clip(image*0.7 + mask, 0, 1)
        return mask, overlay
    def zero_baseline_integrated_grads(self, input, target_label_idx, steps=50):
        input = self._pre_processing(input)
        baseline = torch.zeros(input.shape).to(self.device)
        return self._integrated_grads(input, target_label_idx, baseline, steps)
    def gaussian_noise_integrated_grads(self, input, target_label_idx, steps, num_random_trials):
        all_intgrads = []
        input = self._pre_processing(input)
        itr = tqdm(range(num_random_trials), unit="trial")
        for i in itr:
            integrated_grad = self._integrated_grads(input, target_label_idx, \
                                                    baseline=torch.normal(0, 0.4, input.shape).to(self.device), steps=steps)
            all_intgrads.append(integrated_grad)
        avg_intgrads = np.average(np.array(all_intgrads), axis=0)
        return avg_intgrads
    def random_baseline_integrated_grads(self, input, target_label_idx, steps, num_random_trials):
        all_intgrads = []
        input = self._pre_processing(input)
        itr = tqdm(range(num_random_trials), unit="trial")
        for i in itr:
            integrated_grad = self._integrated_grads(input, target_label_idx, \
                                                    baseline=torch.rand(input.shape).to(self.device), steps=steps)
            all_intgrads.append(integrated_grad)
        avg_intgrads = np.average(np.array(all_intgrads), axis=0)
        return avg_intgrads