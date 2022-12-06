import torch
from ..attack import Attacker, L2Step

class RepVisualization:
    def __init__(self, model, normalizer=lambda x: x):
        self.model = model
        self.normalizer = normalizer
        self.att = Attacker(model, normalizer=normalizer, restart=False)
        self.device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
    def rep_inversion(self, orig_images, source_images):
        if orig_images.shape[0] != source_images.shape[0]:
            raise Exception("Orginal images shape must be equal to the source ones")
        orig_images = orig_images.to(self.device)
        source_images = source_images.to(self.device)
        self.att.epsilon = 1000
        self.att.num_iter = 10000
        self.att.attack_step = L2Step
        self.att.lr = 1
        def inversion_loss(m, inp, t):
            _, rep = m(inp, with_latent=True, fake_relu=True)
            loss = torch.div(torch.norm(rep-t, p=2), torch.norm(t))
            return loss
        self.att.criterion = inversion_loss
        _, rep = self.model(self.normalizer(orig_images.cuda()), with_latent=True)
        xadv = self.att(source_images.clone(), rep.clone(), use_best=False, targeted=True)
        return xadv
    def feature_vis(self, source_images, activations):
        self.att.epsilon = 1000
        self.att.num_iter = 200
        self.att.attack_step = L2Step
        self.att.lr = 1
        def activation_loss(m, inp, t):
            _, rep = m(inp, with_latent=True, fake_relu=True)
            loss = rep[:, t]
            return loss
        self.att.criterion = activation_loss
        activations = torch.tensor([activations])
        xadv = self.att(source_images, activations, use_best=False)
        return xadv
    def class_im_gen(self, source_images, class_idx):
        self.att.epsilon = 1000
        self.att.num_iter = 200
        self.att.attack_step = L2Step
        self.att.lr = 1
        def class_loss(m, inp, t):
            out = m(inp, fake_relu=True)
            loss = out[:, t]
            return loss
        self.att.criterion = class_loss
        class_idx = torch.tensor([class_idx])
        xadv = self.att(source_images, class_idx, use_best=False)
        return xadv