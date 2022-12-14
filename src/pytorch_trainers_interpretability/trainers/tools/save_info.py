import torch
import matplotlib.pyplot as plt
import json
import os

class SaveInfo:
    def __init__(self, resume_path, save_path, adv_train=False):
        self.save_path = save_path
        self.resume_path = resume_path
        self.best_comp_acc = -1
        self.best_test_acc = -1
        self.acc_train = []
        self.loss_train = []
        self.to_save_model = False
        self.acc_test = []
        self.loss_test = []
        if adv_train == True:
            self.acc_test_adv = []
            self.loss_test_adv = []
            self.best_adv_acc = -1
    def append_train(self, acc, loss):
        self.acc_train.append(acc)
        self.loss_train.append(loss)
    def append_test(self, acc, loss, acc_adv=None, loss_adv=None):
        self.acc_test.append(acc)
        self.loss_test.append(loss)
        if self.best_test_acc < acc:
            self.best_test_acc = acc
        compare_acc = acc
        if hasattr(self, "acc_test_adv"):
            self.acc_test_adv.append(acc_adv)
            self.loss_test_adv.append(loss_adv)
            if self.best_adv_acc < acc_adv:
                self.best_adv_acc = acc_adv
            compare_acc += acc_adv
            compare_acc /= 2
        if self.best_comp_acc < compare_acc:
            self.best_comp_acc = compare_acc
            self.to_save_model = True
    def save_model(self, model_state_dict, epoch, loss, optimizer_state_dict, lr_scheduler_state_dict=None, best=False):
        save = {"model_state_dict": model_state_dict, "epoch": epoch, "loss": loss, "optimizer_state_dict": optimizer_state_dict}
        if lr_scheduler_state_dict is not None:
            save["lr_scheduler_state_dict"] = lr_scheduler_state_dict
        filename = "checkpoint.pt"
        if best == True:
            self.to_save_model = False
            filename = "best.pt"
        torch.save(save, os.path.join(self.save_path, filename))
    def save_loss_plot(self):
        iters = [*range(1, len(self.loss_train)+1)]
        plt.plot(iters, self.loss_test, 'r-', label="Test loss natural")
        if hasattr(self, "acc_test_adv"):
            plt.plot(iters, self.loss_test_adv, 'g-', label="Test loss adversarial")
        plt.plot(iters, self.loss_train, 'b-', label="Train loss")
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig(os.path.join(self.save_path, "loss.pdf"))
        plt.clf()
    def save_acc_plot(self):
        iters = [*range(1, len(self.acc_train)+1)]
        plt.plot(iters, self.acc_test, 'r-', label="Test accuracy natural")
        if hasattr(self, "acc_test_adv"):
            plt.plot(iters, self.acc_test_adv, 'g-', label="Test accuracy adversarial")
        plt.plot(iters, self.acc_train, 'b-', label="Train accuracy")
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy(%)')
        plt.savefig(os.path.join(self.save_path, "accuracy.pdf"))
        plt.clf()
    def load_train_info(self, epoch):
        with open(os.path.join(self.resume_path, "train_info.json"), 'r') as fp:
            data = json.load(fp)
        self.acc_train = data["acc_train"][:epoch+1]
        self.loss_train = data["loss_train"][:epoch+1]
        self.acc_test =  data["acc_test"][:epoch+1]
        self.loss_test = data["loss_test"][:epoch+1]
        self.best_comp_acc = data["best_comp_acc"]
        if hasattr(self, "acc_test_adv"):
            self.best_adv_acc = data["best_adv_acc"]
            self.acc_test_adv = data["acc_test_adv"][:epoch+1]
            self.loss_test_adv = data["loss_test_adv"][:epoch+1]
    def save_train_info(self):
        info = {"acc_train": self.acc_train, "loss_train": self.loss_train, "acc_test": self.acc_test, "loss_test": self.loss_test, "best_test_acc": self.best_test_acc, "best_comp_acc": self.best_comp_acc}
        if hasattr(self, "acc_test_adv"):
            info["best_adv_acc"] = self.best_adv_acc
            info["acc_test_adv"] = self.acc_test_adv
            info["loss_test_adv"] = self.loss_test_adv
        with open(os.path.join(self.save_path, "train_info.json"), 'w') as fp:
            json.dump(info, fp)
            