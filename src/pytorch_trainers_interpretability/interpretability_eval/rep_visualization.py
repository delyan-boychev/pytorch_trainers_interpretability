import torch
from tqdm import tqdm
import numpy as np
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
        #setting up parameters for PGD
        self.att.epsilon = 1000
        self.att.num_iter = 10000
        self.att.attack_step = L2Step
        self.att.lr = 1
        #inversion loss function to minimize the l2 distance between the rep vectors
        def inversion_loss(m, inp, t):
            _, rep = m(inp, with_latent=True, fake_relu=True)
            loss = torch.div(torch.norm(rep-t, p=2), torch.norm(t))
            return loss
        self.att.custom_loss = True
        self.att.criterion = inversion_loss
        _, r = self.model(self.normalizer(orig_images.cuda()), with_latent=True)
        xadv = self.att(source_images.clone(), r.clone(), use_best=False, targeted=True)
        return xadv
    def feature_vis(self, source_images, activations):
        #setting up settings for 
        self.att.epsilon = 1000
        self.att.num_iter = 200
        self.att.attack_step = L2Step
        self.att.lr = 1
        def activation_loss(m, inp, t):
            _, rep = m(inp, with_latent=True, fake_relu=True)
            loss = rep[:, t]
            return loss
        self.att.criterion = activation_loss
        self.att.custom_loss = True
        activations = torch.tensor([activations])
        xadv = self.att(source_images, activations, use_best=False)
        return xadv
    def set_testset_rep_vector(self, testloader):
        itr = tqdm(enumerate(testloader))
        img_test, rep_test = [], []
        n = 0
        for i, (im, targ) in itr:
            n += im.shape[0]
            itr.set_description(f"Completed: {i}/{len(testloader)}")
            with torch.no_grad():
                _, rep = self.model(self.normalizer(im.cuda()), with_latent=True)
                rep_test.append(rep.cpu().numpy())
            img_test.append(im)

        self.rep_test = np.concatenate(rep_test)
        self.img_test = torch.cat(img_test)
    def _get_topk_imgs(self, index, dist=lambda x:x):
        print(dist(self.rep_test[:,index]).shape)
        top_k = dist(self.rep_test[:,index]).argsort()[-3:][::-1]
        print(top_k)
        res = [self.img_test[i:i+1] for i in top_k]
        res = torch.cat(res)
        return res.cpu()
    def feature_vis_max_activation(self, actv, input_size):
        if self.rep_test is None or self.img_test is None:
            raise Exception("You should first run set_testset_rep_vector to get maximal activations")
        img_source = 0.5 * torch.ones(input_size) 
        vis = self.feature_vis(img_source, actv)
        dist1 = lambda x: x
        dist2 = lambda x: -x
        if type(actv) is list:
            dist1 = lambda x: np.squeeze(np.average(x, axis=2), axis=1)
            dist2 = lambda x: np.squeeze(np.average(-x, axis=2), axis=1)
        img_max = self._get_topk_imgs(actv, dist = dist1)
        img_min = self._get_topk_imgs(actv, dist=dist2)
        return img_source.cpu(), vis.cpu(), img_max.cpu(), img_min.cpu()
    def class_im_gen(self, source_images, class_idx):
        self.att.epsilon = 10000
        self.att.num_iter = 400
        self.att.attack_step = L2Step
        self.att.lr = 1
        def class_loss(m, inp, t):
            out = m(inp, fake_relu=True)
            loss = out[:, t]
            return loss
        self.att.criterion = class_loss
        self.att.custom_loss = True
        class_idx = torch.tensor([class_idx])
        xadv = self.att(source_images, class_idx, use_best=False)
        return xadv