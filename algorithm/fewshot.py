import torch
import torch.nn as nn
import torchvision.transforms as TF
import torch.nn.functional as F
import numpy as np
import itertools
from algorithm.solver import Solver
from utils.module import Negative
from utils import accuracy, init_weights

class FewShotSolver(Solver):
    def allocate_tensors(self):
        self.tensors = {}
        self.tensors['images'] = torch.FloatTensor() 
        self.tensors['labels0'] = torch.LongTensor() # labels in the few-shot task or episode
        self.tensors['labels1'] = torch.LongTensor() # labels in the original dataset
    
    def set_tensors(self, batch):
        images, labels0, labels1 = batch
        self.tensors['images'].resize_(images.size()).copy_(images)
        self.tensors['labels0'].resize_(labels0.size()).copy_(labels0)
        self.tensors['labels1'].resize_(labels1.size()).copy_(labels1)
        #
        #return self.tensors['images'], self.tensors['labels0'], self.tensors['labels1']

    def get_trainable_parameters(self):
        parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        return parameters

    def get_logits(self):
        raise NotImplementedError


class ProtoNetSolver(FewShotSolver):
    def compute_logits(self, protos, querys):
        """
        Args:
        @param protos, torch.FloatTensor of size Bx#classxD.
        @param querys, torch.FloatTensor of size Bx#queryxD.

        Return:
        Logits of size Bx#queryx#class.
        """
        raise NotImplementedError

    def get_logits(self):
        images = self.tensors['images']
        labels = self.tensors['labels0']
        feats = self._model(images)
        fN, fC, fH, fW = feats.shape
        feats = feats.view(fN, -1)
        feat_size = feats.size(1)
        episode_meta = self.exp_dict['episode_train_meta'] if self._model.training else self.exp_dict['episode_test_meta']
        n_class = episode_meta.n_class
        n_support = episode_meta.n_support
        n_query = episode_meta.n_query
        n_unlabel = episode_meta.n_unlabel
        n_shot = n_support + n_query + n_unlabel
        episode_volume = n_class * n_shot # number of images in an episode
        assert (fN % episode_volume) == 0
        episode_number = int(fN / episode_volume)
        # onehot_labels = []
        # for _per_task_labels in labels.view(episode_number, -1):
        #     onehot_labels.append((_per_task_labels[:,None] == (_per_task_labels.unique())[None, :]).float())
        # onehot_labels = torch.stack(onehot_labels)
        # onehot_labels = onehot_labels.view(episode_number, n_class, n_shot, -1)
        feats = feats.view(episode_number, n_class, n_shot, feat_size)
        labels = labels.view(episode_number, n_class, n_shot)
        # write the creative main process here
        protos = (feats[:,:,:n_support]).mean(dim=2) # protos in random order
        _, indices = torch.sort(labels[:,:,0],  -1)
        _protos = torch.gather(protos, 1, indices[:,:,None].expand_as(protos)) # protos in order, of size Bx#classxD
        _querys = (feats[:,:,n_support:n_support+n_query]).reshape(episode_number, n_class*n_query, feat_size) # of size Bx#queryxD
        _labels = (labels[:,:,n_support:n_support+n_query]).reshape(episode_number, n_class*n_query)
        logits = self.compute_logits(_protos, _querys)
        return logits.reshape(-1, n_class), _labels.reshape(-1)

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

class ProtoNetPlusSolver(FewShotSolver):
    """Meta-train from a pretrained classifier in which the feature extractor part
    will be fine-tuned in the meta-train process, and the classifier part will hold
    still (if the classifier is initialzied from a pretrained model). 
    A coefficient is provided for balancing the meta-train loss and classification loss.
    """
    def get_trainable_parameters(self):
        assert self.model.num_classes > 0 # classifier exist.
        # assert self.initialize_pretrained # initialization from pretrained model.
        if self.initialize_pretrained:
            # fix classifier
            for param in self.model.classifier.parameters():
                param.requires_grad_(False)
        return super().get_trainable_parameters()

    def compute_logits(self, protos, querys):
        """
        Args:
        @param protos, torch.FloatTensor of size Bx#classxD.
        @param querys, torch.FloatTensor of size Bx#queryxD.

        Return:
        Logits of size Bx#queryx#class.
        """
        raise NotImplementedError

    def get_logits(self):
        images = self.tensors['images']
        fs_labels = self.tensors['labels0'] # labels in the few-shot task or episode
        cc_labels = self.tensors['labels1'] # labels in the original dataset
        feats, cc_logits = self._model(images)
        fN, fC, fH, fW = feats.shape
        feats = feats.view(fN, -1)
        feat_size = feats.size(1)
        episode_meta = self.exp_dict['episode_train_meta'] if self._model.training else self.exp_dict['episode_test_meta']
        n_class = episode_meta.n_class
        n_support = episode_meta.n_support
        n_query = episode_meta.n_query
        n_unlabel = episode_meta.n_unlabel
        n_shot = n_support + n_query + n_unlabel
        episode_volume = n_class * n_shot # number of images in an episode
        assert (fN % episode_volume) == 0
        episode_number = int(fN / episode_volume)
        # onehot_labels = []
        # for _per_task_labels in fs_labels.view(episode_number, -1):
        #     onehot_labels.append((_per_task_labels[:,None] == (_per_task_labels.unique())[None, :]).float())
        # onehot_labels = torch.stack(onehot_labels)
        # onehot_labels = onehot_labels.view(episode_number, n_class, n_shot, -1)
        feats = feats.view(episode_number, n_class, n_shot, feat_size)
        fs_labels = fs_labels.view(episode_number, n_class, n_shot)
        # write the creative main process here
        protos = (feats[:,:,:n_support]).mean(dim=2) # protos in random order
        _, indices = torch.sort(fs_labels[:,:,0],  -1)
        _protos = torch.gather(protos, 1, indices[:,:,None].expand_as(protos)) # protos in order, of size Bx#classxD
        _querys = (feats[:,:,n_support:n_support+n_query]).reshape(episode_number, n_class*n_query, feat_size) # of size Bx#queryxD
        _labels = (fs_labels[:,:,n_support:n_support+n_query]).reshape(episode_number, n_class*n_query)
        logits = self.compute_logits(_protos, _querys)
        # only support images are used when the extra classifier is envolved.
        cc_logits = cc_logits.view(episode_number, n_class, n_shot, -1)
        cc_labels = cc_labels.view(episode_number, n_class, n_shot)
        _cc_logits = (cc_logits[:,:,:n_support]).reshape(-1, cc_logits.size(-1))
        _cc_labels = (cc_labels[:,:,:n_support]).reshape(-1)
        return logits.reshape(-1, n_class), _labels.reshape(-1), _cc_logits, _cc_labels 

    def train_on_batch(self, batch):
        self.set_tensors(batch)
        fs_logits, fs_labels, cc_logits, cc_labels = self.get_logits()
        cc_loss = F.cross_entropy(cc_logits, cc_labels)
        fs_loss = F.cross_entropy(fs_logits, fs_labels)
        acc1, acc2, acc3, acc4, acc5 = accuracy(fs_logits, fs_labels, topk=(1,2,3,4,5))
        cc_coef = float(self.exp_dict['solver'].get('vanilla_classifier_balance_weight', 1.0))
        fs_coef = float(self.exp_dict['solver'].get('fewshot_classifier_balance_weight', 1.0))
        loss = fs_coef * fs_loss + cc_coef * cc_loss
        stats = {}
        stats["loss"] = loss.item()
        stats["fs_loss"] = fs_loss.item()
        stats["cc_loss"] = cc_loss.item()
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
        fs_logits, fs_labels, _, _ = self.get_logits()
        fs_loss = F.cross_entropy(fs_logits, fs_labels)
        acc1, acc2, acc3, acc4, acc5 = accuracy(fs_logits, fs_labels, topk=(1,2,3,4,5))
        loss = fs_loss
        stats = {}
        stats["loss"] = loss.item()
        stats["fs_loss"] = fs_loss.item()
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
        self.summary_writer.add_scalars('fs_loss', {phase: stats['fs_loss']}, self.curr_epoch)
        self.summary_writer.add_scalars('acc1', {phase: stats['Top1Acc']}, self.curr_epoch)
        self.summary_writer.add_scalars('acc5', {phase: stats['Top5Acc']}, self.curr_epoch)

class ProtoNetPlusPlusSolver(ProtoNetPlusSolver):
    """Meta-train from a pretrained classifier in which the feature extractor part
    will be fine-tuned in the meta-train process, and the classifier part will hold
    still (if the classifier is initialzied from a pretrained model). 
    A coefficient is provided for balancing the meta-train loss and classification loss.
    The classifier is a dense classifier, each feature cell is taken as an instance.
    """
    def get_logits(self):
        images = self.tensors['images']
        fs_labels = self.tensors['labels0'] # labels in the few-shot task or episode
        cc_labels = self.tensors['labels1'] # labels in the original dataset
        feats, cc_logits = self._model(images) # feats of size NxCxHxW, logits of size Nx #classes xHxW
        fN, fC, fH, fW = feats.shape
        lN, lC, lH, lW = cc_logits.shape
        # assert lH == fH
        # assert lW == fW
        cc_labels = cc_labels.view(lN,1,1).expand(-1, lH, lW)
        feats = feats.mean(dim=(-2,-1), keepdim=False)
        feat_size = feats.size(1)
        assert feat_size == fC
        episode_meta = self.exp_dict['episode_train_meta'] if self._model.training else self.exp_dict['episode_test_meta']
        n_class = episode_meta.n_class
        n_support = episode_meta.n_support
        n_query = episode_meta.n_query
        n_unlabel = episode_meta.n_unlabel
        n_shot = n_support + n_query + n_unlabel
        episode_volume = n_class * n_shot # number of images in an episode
        assert (fN % episode_volume) == 0
        episode_number = int(fN / episode_volume)
        # onehot_labels = []
        # for _per_task_labels in fs_labels.view(episode_number, -1):
        #     onehot_labels.append((_per_task_labels[:,None] == (_per_task_labels.unique())[None, :]).float())
        # onehot_labels = torch.stack(onehot_labels)
        # onehot_labels = onehot_labels.view(episode_number, n_class, n_shot, -1)
        feats = feats.view(episode_number, n_class, n_shot, feat_size)
        fs_labels = fs_labels.view(episode_number, n_class, n_shot)
        # write the creative main process here
        protos = (feats[:,:,:n_support]).mean(dim=2) # protos in random order
        _, indices = torch.sort(fs_labels[:,:,0],  -1)
        _protos = torch.gather(protos, 1, indices[:,:,None].expand_as(protos)) # protos in order, of size Bx#classxD
        _querys = (feats[:,:,n_support:n_support+n_query]).reshape(episode_number, n_class*n_query, feat_size) # of size Bx#queryxD
        _labels = (fs_labels[:,:,n_support:n_support+n_query]).reshape(episode_number, n_class*n_query)
        logits = self.compute_logits(_protos, _querys)
        # only support images are used when the extra classifier is envolved.
        cc_logits = cc_logits.view(episode_number, n_class, n_shot, lC, lH, lW)
        cc_labels = cc_labels.view(episode_number, n_class, n_shot, lH, lW)
        _cc_logits = (cc_logits[:,:,:n_support]).reshape(-1, lC, lH, lW)
        _cc_labels = (cc_labels[:,:,:n_support]).reshape(-1, lH, lW)
        return logits.reshape(-1, n_class), _labels.reshape(-1), _cc_logits, _cc_labels

class ProtoNetCosineMetricSolver(ProtoNetSolver):
    def init_model_callback(self):
        self.model.insert('tau', nn.Parameter(torch.tensor(10.)))
    
    def compute_logits(self, protos, querys):
        """
        Args:
        @param protos, torch.FloatTensor of size Bx#classxD.
        @param querys, torch.FloatTensor of size Bx#queryxD.

        Return:
        Logits of size Bx#queryx#class.
        """
        protos = F.normalize(protos, p=2, dim=-1)
        querys = F.normalize(querys, p=2, dim=-1)
        logits = torch.sum(
            querys[:,:, None, :] * protos[:,None,:,:],
            dim = -1
        ) * self.model.tau
        return logits
    
    def write_stats_summary(self, stats, phase):
        # visualize temperature
        self.summary_writer.add_scalars('tau', {phase: self.model.tau.item()}, self.curr_epoch)
        super().write_stats_summary(stats, phase)

class ProtoNetProjectionMetricSolver(ProtoNetSolver):
    def compute_logits(self, protos, querys):
        """
        Args:
        @param protos, torch.FloatTensor of size Bx#classxD.
        @param querys, torch.FloatTensor of size Bx#queryxD.

        Return:
        Logits of size Bx#queryx#class.
        """
        protos = F.normalize(protos, p=2, dim=-1)
        # querys = F.normalize(querys, p=2, dim=-1)
        logits = torch.sum(
            querys[:,:, None, :] * protos[:,None,:,:],
            dim = -1
        )
        return logits

class ProtoNetCosineMetricPlusSolver(ProtoNetCosineMetricSolver, ProtoNetPlusSolver):
    def compute_logits(self, protos, querys):
        return ProtoNetCosineMetricSolver.compute_logits(self, protos, querys)
    
    def get_logits(self):
        return ProtoNetPlusSolver.get_logits(self)
    
    def train_on_batch(self, batch):
        return ProtoNetPlusSolver.train_on_batch(self, batch)
    
    def eval_on_batch(self, batch):
        return ProtoNetPlusSolver.eval_on_batch(self, batch)
    
    def write_stats_summary(self, stats, phase):
        # visualize temperature
        self.summary_writer.add_scalars('tau', {phase: self.model.tau.item()}, self.curr_epoch)
        ProtoNetPlusSolver.write_stats_summary(self, stats, phase)

class ProtoNetProjectionMetricPlusSolver(ProtoNetProjectionMetricSolver, ProtoNetPlusSolver):
    def compute_logits(self, protos, querys):
        return ProtoNetProjectionMetricSolver.compute_logits(self, protos, querys)
    
    def get_logits(self):
        return ProtoNetPlusSolver.get_logits(self)
    
    def train_on_batch(self, batch):
        return ProtoNetPlusSolver.train_on_batch(self, batch)
    
    def eval_on_batch(self, batch):
        return ProtoNetPlusSolver.eval_on_batch(self, batch)
    
    def write_stats_summary(self, stats, phase):
        ProtoNetPlusSolver.write_stats_summary(self, stats, phase)

class ProtoNetCosineMetricPlusPlusSolver(ProtoNetCosineMetricSolver, ProtoNetPlusPlusSolver):
    def compute_logits(self, protos, querys):
        return ProtoNetCosineMetricSolver.compute_logits(self, protos, querys)
    
    def get_logits(self):
        return ProtoNetPlusPlusSolver.get_logits(self)
    
    def train_on_batch(self, batch):
        return ProtoNetPlusPlusSolver.train_on_batch(self, batch)
    
    def eval_on_batch(self, batch):
        return ProtoNetPlusPlusSolver.eval_on_batch(self, batch)
    
    def write_stats_summary(self, stats, phase):
        # visualize temperature
        self.summary_writer.add_scalars('tau', {phase: self.model.tau.item()}, self.curr_epoch)
        ProtoNetPlusPlusSolver.write_stats_summary(self, stats, phase)

class ProtoNetProjectionMetricPlusPlusSolver(ProtoNetProjectionMetricSolver, ProtoNetPlusPlusSolver):
    def compute_logits(self, protos, querys):
        return ProtoNetProjectionMetricSolver.compute_logits(self, protos, querys)
    
    def get_logits(self):
        return ProtoNetPlusPlusSolver.get_logits(self)
    
    def train_on_batch(self, batch):
        return ProtoNetPlusPlusSolver.train_on_batch(self, batch)
    
    def eval_on_batch(self, batch):
        return ProtoNetPlusPlusSolver.eval_on_batch(self, batch)
    
    def write_stats_summary(self, stats, phase):
        ProtoNetPlusPlusSolver.write_stats_summary(self, stats, phase)
