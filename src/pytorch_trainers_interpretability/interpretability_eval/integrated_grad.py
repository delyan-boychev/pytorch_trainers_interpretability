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
        input = input.requires_grad_(True)
        output = self.model(input)
        output = F.softmax(output, dim=-1)[0, target_label_idx]
        output.backward()
        gradient = input.grad
        return gradient, output.item()
    def _generate_staturate_batches(self, input, baseline, steps):
        alphas = torch.linspace(0.0, 1.0, steps+1).to(self.device)
        alphas = alphas[:, None, None, None]
        images = baseline + alphas*(input-baseline)
        return images
    def _integrated_grads(self, input, target_label_idx, baseline, steps=50, batch_size=30):
        scaled_inputs = self._generate_staturate_batches(input, baseline, steps)
        gradients = []
        for i in range(0, steps, batch_size):
            start = i
            end = min(start+1, steps)
            batch = scaled_inputs[start:end]
            gradient, _ = self._pred_grad(batch, target_label_idx)
            gradients.append(gradient)
        gradients = torch.cat(gradients)
        gradients = (gradients[:-1] + gradients[1:]) / 2.0
        integrated_grads = torch.mean(gradients, axis=0)
        delta_X = (input - baseline).to(self.device)
        integrated_grads = (delta_X*integrated_grads).squeeze(0).detach().cpu().numpy()
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
    def black_baseline_integrated_grads(self, input, target_label_idx, steps=50, batch_size=30):
        input = self._pre_processing(input)
        baseline = torch.zeros(input.shape).to(self.device)
        return self._integrated_grads(input, target_label_idx, baseline, steps, batch_size)
    def gaussian_noise_integrated_grads(self, input, target_label_idx, steps, num_random_trials, batch_size=30):
        all_intgrads = []
        input = self._pre_processing(input)
        itr = tqdm(range(num_random_trials), unit="trial")
        for i in itr:
            integrated_grad = self._integrated_grads(input, target_label_idx, \
                                                    baseline=torch.normal(0, 0.4, input.shape).to(self.device), steps=steps, batch_size=batch_size)
            all_intgrads.append(integrated_grad)
        avg_intgrads = np.average(np.array(all_intgrads), axis=0)
        return avg_intgrads
    def random_baseline_integrated_grads(self, input, target_label_idx, steps, num_random_trials, batch_size=30):
        all_intgrads = []
        input = self._pre_processing(input)
        itr = tqdm(range(num_random_trials), unit="trial")
        for i in itr:
            integrated_grad = self._integrated_grads(input, target_label_idx, \
                                                    baseline=torch.rand(input.shape).to(self.device), steps=steps, batch_size=batch_size)
            all_intgrads.append(integrated_grad)
        avg_intgrads = np.average(np.array(all_intgrads), axis=0)
        return avg_intgrads