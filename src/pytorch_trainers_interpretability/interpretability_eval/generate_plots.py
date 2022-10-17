import torch
from .integrated_grad import IntegratedGrad
import matplotlib.pyplot as plt
import numpy as np
from textwrap import wrap

def int_grads_compare_regular_robust(regular, robust, images, labels, classes, normalizer=lambda x: x, treshold=0):
    integrated_grad = IntegratedGrad(regular, normalizer=normalizer)
    integrated_grad2 = IntegratedGrad(robust, normalizer=normalizer)
    num_img = labels.shape[0]
    labels_text = [ '\n'.join(wrap(classes[l.item()], 20)) for l in labels.cpu() ]
    num_cols = np.ceil(num_img/9).astype(int)
    fig = plt.figure(figsize=(20*num_cols, 55))
    subfigs = fig.subfigures(nrows=1, ncols=num_cols)
    k = 0
    for j, sub in enumerate(subfigs):
        sub2 = sub.subfigures(nrows=9, ncols=1)
        for i, subfig in enumerate(sub2):
            if num_img == k:
                break
            axs = subfig.subplots(nrows=1, ncols=3)
            if i is 0:
                axs[0].set_title("Original image", fontsize=30)
                axs[1].set_title("Regular model", fontsize=30)
                axs[2].set_title("Robust model", fontsize=30)
            img = images[k:k+1].cuda()
            img = normalizer(img)
            pr = regular(img)
            pr2 = robust(img)
            image = images[k].cpu().permute(1, 2, 0).numpy()
            grad = integrated_grad.random_baseline_integrated_grads(image, labels[k], steps=50, num_random_trials=10, batch_size=50)
            grad2 = integrated_grad2.random_baseline_integrated_grads(image, labels[k], steps=50, num_random_trials=10, batch_size=50)
            pred = '\n'.join(wrap(classes[pr.argmax(dim=1).item()], 20))
            pred2 = '\n'.join(wrap(classes[pr2.argmax(dim=1).item()], 20))
            axs[0].set_xlabel(f"True class: {labels_text[k]}", fontsize=20)
            axs[0].imshow(image)
            axs[1].set_xlabel(f"Class: {pred}\n Prob: {torch.softmax(pr, dim=1).amax(dim=1).item():.2f}", fontsize=25)
            axs[1].imshow(integrated_grad.visualization(grad, image))
            axs[2].set_xlabel(f"Class: {pred2}\n Prob: {torch.softmax(pr2, dim=1).amax(dim=1).item():.2f}", fontsize=25)
            axs[2].imshow(integrated_grad2.visualization(grad2, image))
            axs[0].set_xticklabels([])
            axs[0].set_yticklabels([])
            axs[1].set_xticklabels([])
            axs[1].set_yticklabels([])
            axs[2].set_xticklabels([])
            axs[2].set_yticklabels([])
            k+=1
    plt.tight_layout()
