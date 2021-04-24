import torch
import torch.nn as nn
import torchvision.transforms as TF
import torch.nn.functional as F
from algorithm.solver import Solver
from utils import accuracy

class BasicSolver(Solver):
    def allocate_tensors(self):
        self.tensors = {}
        self.tensors['images'] = torch.FloatTensor() 
        self.tensors['labels0'] = torch.LongTensor() # revised labels
        self.tensors['labels1'] = torch.LongTensor() # old labels
    
    def set_tensors(self, batch):
        images, labels0, labels1 = batch
        self.tensors['images'].resize_(images.size()).copy_(images)
        self.tensors['labels0'].resize_(labels0.size()).copy_(labels0)
        self.tensors['labels1'].resize_(labels1.size()).copy_(labels1)
        #
        #return self.tensors['images'], self.tensors['labels0'], self.tensors['labels1']
    
    def get_logits(self):
        images = self.tensors['images']
        labels = self.tensors['labels0']
        logits = self._model(images)
        return logits, labels


class ClassifierSolver(BasicSolver):
    def get_trainable_parameters(self):
        parameters = filter(lambda p: p.requires_grad, self._model.parameters())
        return parameters
        
    def train_on_batch(self, batch):
        self.set_tensors(batch)
        logits, labels = self.get_logits()
        loss = F.cross_entropy(logits, labels)
        acc1, acc2, acc3, acc4, acc5 = accuracy(logits, labels, topk=(1,2,3,4,5))
        stats = {}
        stats["loss"] = loss.item()
        stats["Top1Acc"] = acc1.item()
        # stats["Top2Acc"] = acc2.item()
        # stats["Top3Acc"] = acc3.item()
        # stats["Top4Acc"] = acc4.item()
        stats["Top5Acc"] = acc5.item()
        #
        # update model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #
        return stats

    def eval_on_batch(self, batch):
        self.set_tensors(batch)
        logits, labels = self.get_logits()
        loss = F.cross_entropy(logits, labels)
        acc1, acc2, acc3, acc4, acc5 = accuracy(logits, labels, topk=(1,2,3,4,5))
        stats = {}
        stats["loss"] = loss.item()
        stats["Top1Acc"] = acc1.item()
        # stats["Top2Acc"] = acc2.item()
        # stats["Top3Acc"] = acc3.item()
        # stats["Top4Acc"] = acc4.item()
        stats["Top5Acc"] = acc5.item()
        #
        # update model
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        #
        return stats
    
    def write_stats_summary(self, stats, phase):
        """Write train/val/test stats into tensorboard summary.

        Args:
        @param stats, the running stats.
        @param phase, the phase name, must be one of 'train', 'val' and 'test'.
        """
        self.summary_writer.add_scalars('loss', {phase: stats['loss']}, self.curr_epoch)
        self.summary_writer.add_scalars('acc1', {phase: stats['Top1Acc']}, self.curr_epoch)
        self.summary_writer.add_scalars('acc5', {phase: stats['Top5Acc']}, self.curr_epoch)

class DenseClassifierSolver(ClassifierSolver):
    def get_logits(self):
        images = self.tensors['images'] # Nx3xiHxiW
        labels = self.tensors['labels0'] # N
        assert self.model.avgpool == False # dense classification requires a feature map
        logits = self._model(images) # NxCxfHxfW
        N,C,fH,fW = logits.shape
        labels = labels.unsqueeze(dim=1).unsqueeze(dim=1).expand(-1, fH, fW)
        logits = logits.permute(0,2,3,1).reshape(-1,C)
        labels = labels.reshape(-1)
        return logits, labels

class DenseClassifierSoftlabelSolver(ClassifierSolver):
    """Dense classifier with soft label smoothing.
    """
    def init_model_callback(self):
        m = self.model.out_dim
        self.model.insert(
            'submodule_adp',
            nn.Sequential(
                nn.Conv2d(2*m, 8, kernel_size=3, padding=1, stride=1),
                nn.LeakyReLU(0.1),
                nn.Conv2d(8, 1, kernel_size=1, padding=0, stride=1),
            )
        )
        # self.model.submodule_adp.apply(init_weights)
        # cross attention module cannot work with average pooling.
        assert self.model.avgpool == False
    
    def get_logits(self):
        num_classes = self.model.num_classes
        images = self.tensors['images'] # Nx3xiHxiW
        labels = self.tensors['labels0'] # N
        assert self.model.avgpool == False # dense classification requires a feature map
        feats, logits = self._model(images) # NxfNxfHxfW, NxlNxlHxlW
        N,fN,fH,fW = feats.shape
        _,lN,lH,lW = logits.shape
        labels = labels.unsqueeze(dim=1).unsqueeze(dim=1).expand(-1, lH, lW)
        logits = logits.permute(0,2,3,1).reshape(-1,lN)
        labels = labels.reshape(-1)
        #
        _feats = torch.cat([feats, torch.mean(feats, dim=(2,3), keepdim=True).expand(-1,-1,fH,fW)], dim=1) # Nx2fNxfHxfW
        score = self.model.submodule_adp(_feats)
        score = F.sigmoid(score).view(-1, 1).clamp(min=1e-12)
        softlabels = (1.0 - score) * F.one_hot(labels, num_classes) + score / float(num_classes) 
        return logits, labels, softlabels
    
    def train_on_batch(self, batch):
        self.set_tensors(batch)
        logits, labels, softlabels = self.get_logits()
        logits = F.log_softmax(logits, dim=1)
        l1 = F.nll_loss(logits, labels)
        l2 = torch.mean(torch.sum(-logits * softlabels, dim=1))
        acc1, acc2, acc3, acc4, acc5 = accuracy(logits, labels, topk=(1,2,3,4,5))
        loss = 0.5 * l1 + 0.5 * l2
        stats = {}
        stats["loss"] = loss.item()
        stats["Top1Acc"] = acc1.item()
        # stats["Top2Acc"] = acc2.item()
        # stats["Top3Acc"] = acc3.item()
        # stats["Top4Acc"] = acc4.item()
        stats["Top5Acc"] = acc5.item()
        #
        # update model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        #
        return stats

    def eval_on_batch(self, batch):
        self.set_tensors(batch)
        logits, labels, softlabels = self.get_logits()
        logits = F.log_softmax(logits, dim=1)
        l1 = F.nll_loss(logits, labels)
        l2 = torch.mean(torch.sum(-logits * softlabels, dim=1))
        acc1, acc2, acc3, acc4, acc5 = accuracy(logits, labels, topk=(1,2,3,4,5))
        loss = 0.5 * l1 + 0.5 * l2
        stats = {}
        stats["loss"] = loss.item()
        stats["Top1Acc"] = acc1.item()
        # stats["Top2Acc"] = acc2.item()
        # stats["Top3Acc"] = acc3.item()
        # stats["Top4Acc"] = acc4.item()
        stats["Top5Acc"] = acc5.item()
        #
        # update model
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        #
        return stats