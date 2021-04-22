"""Define a generic class for training and testing learning algorithms."""
import os
import glob
import collections
from tqdm import tqdm
import torch
import datetime
import logging
from torch.utils.tensorboard import SummaryWriter
from algorithm.meter import DAverageMeter
from utils import import_x, import_module, init_weights, set_random_seeds

__all__ = ['Solver', ]

class Solver():
    def __init__(self, exp_dict):
        self.exp_dict = exp_dict
        self.init_context()
        self.set_experiment_name()
        self.set_experiment_dir()
        self.allocate_tensors()
        self.init_network()

    def set_experiment_dir(self):
        directory_path = self.exp_dict['exp_dir']
        self.exp_dir = directory_path
        if (not os.path.isdir(self.exp_dir)):
            os.makedirs(self.exp_dir, exist_ok=True)

        self.vis_dir = os.path.join(directory_path,'visuals')
        if (not os.path.isdir(self.vis_dir)):
            os.makedirs(self.vis_dir, exist_ok=True)

        self.preds_dir = os.path.join(directory_path,'preds')
        if (not os.path.isdir(self.preds_dir)):
            os.makedirs(self.preds_dir, exist_ok=True)
            
        self.set_log_file_handler()
        self.logger.info('Solver options => %s' % self.exp_dict)

    def set_experiment_name(self):
        name = self.exp_dict['name']
        self.exp_name = name
        if self.exp_dict.get('exp_dir', None) is None: # if not set, use default experiment folder
            self.exp_dict['exp_dir'] = os.path.join('experiment', self.exp_name)

    def set_log_file_handler(self):
        self.logger = logging.getLogger(self._typename)

        strHandler = logging.StreamHandler()
        formatter = logging.Formatter(
                '%(asctime)s - %(name)-8s - %(levelname)-6s - %(message)s')
        strHandler.setFormatter(formatter)
        self.logger.addHandler(strHandler)
        self.logger.setLevel(logging.INFO if not self.exp_dict.get('debug', False) else logging.DEBUG)

        if self._is_main_process and not self.exp_dict.get('noflog', False): # only main process can write log file
            self.log_dir = os.path.join(self.exp_dir, 'logs')
            if (not os.path.isdir(self.log_dir)):
                os.makedirs(self.log_dir, exist_ok=True)
            self.log_file = os.path.join(self.log_dir, 'LOG_INFO_'+self.timestamp+'.txt')
            self.log_fileHandler = logging.FileHandler(self.log_file)
            self.log_fileHandler.setFormatter(formatter)
            self.logger.addHandler(self.log_fileHandler)

    def init_context(self):
        self.is_tqdm = not not self.exp_dict.get('tqdm', False)
        self.is_cuda = self.exp_dict.get('iscuda', False)
        self.is_cuda_parallel = self.is_cuda and (self.exp_dict.get('ngpus',0)>1)
        self.min_lr_decay = float(self.exp_dict['solver'].get('min_lr_decay', 1e-6))
        self.keep_best_model_metric_name = self.exp_dict.get('best_metric', 'Top1Acc')
        self.curr_epoch = 0
        self.curr_iter = 0
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.dummy_input = torch.rand(4, 3, 84, 84)
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        self.initialize_pretrained = False
        if self.is_cuda_parallel: # multi-GPU setting requires torch.distributed package
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available.")
            if not torch.distributed.is_initialized():
                raise RuntimeError("Requires distributed progress group initialized first.")

    def allocate_tensors(self):
        """(Optional) allocate torch tensors that could potentially be used in
            in the train_step() or evaluation_step() functions. If the
            load_to_gpu() function is called then those tensors will be moved to
            the gpu device.
        """
        self.tensors = {}

    def init_network(self):
        """
        Create a network model instance with the definition in configure
        `self.exp_dict["network"]`. If pretrained model parameters exist, use 
        them to initialize the new created model.
        """
        self.model_defs = self.exp_dict['network']
        network = import_x(
            self.model_defs["module"], 
            import_module(self.model_defs["package"])) # load module meta
        if network is None:
            raise ValueError("Not supported or recognized network: "+self.model_defs["package"]+"."+self.model_defs["module"])
        self.logger.debug('==> Initiliaze network with params: %s' %(self.model_defs["param"]))
        self.model = network() if self.model_defs["param"] is None else network(**(self.model_defs["param"]))
        self.init_model_callback()
        self.model.apply(init_weights) # randomly initialization
        # load pretrained model
        self.pretrained_model_params = self.model_defs.get("pretrained", None)
        if self.pretrained_model_params is not None: self.load_pretrained(self.model, self.pretrained_model_params)
        #
        self.init_network_callback()
        #
        self.init_record_of_best_model()
        # load checkpoint
        self.checkpoint_params = self.load_checkpoint(self.exp_dict['checkpoint'], suffix='.best') or self.load_checkpoint(self.exp_dict['checkpoint'], suffix='')
        # load model into GPU devices
        if self.is_cuda: self.load_to_gpu(parallel=self.is_cuda_parallel)
        # init optimizer
        self.init_optimizer()

    def init_model_callback(self):
        """The callback method is called right after the model is built.
        """
        pass
        
    def init_network_callback(self):
        """
        The callback method is called right after network construction and before 
        the optimizer construction.
        """
        pass

    def load_pretrained(self, network, pretrained_path):
        """
        Intialize network model with pretrained parameters. Argument `pretrained_path`
        could be the serialized pretrained model or the path where the pretrained model
        are stored. If `pretrained_path` is a directory, get the last modifided pretained
        model to initialize the network model. If the network model does not have the
        same paramters with the pretrained one, only the matched parameters are copied
        to intialize the corresponding ones in the network model.

        Args:\n
        - network: nn.Module, the neural network model to be initialized.
        - pretrained_path: str, the path to serialized pretrained model(s).

        Output:\n
        NO RETURN VALUE
        """
        all_possible_files = sorted(glob.glob(pretrained_path), key=os.path.getmtime) # order returned model by last modification time
        if len(all_possible_files) == 0:
            raise ValueError('%s: no such file' % (pretrained_path))
        else:
            pretrained_path = all_possible_files[-1]
        assert(os.path.isfile(pretrained_path))
        self.logger.info('==> Load pretrained parameters from file %s.' %
                        pretrained_path)
        pretrained_model = torch.load(pretrained_path, map_location="cpu")
        if pretrained_model['model_classname'] == 'DistributedDataParallel':
            self.logger.warn('==> network is pretrained in multi-gpu mode,'
            'updating to be capable of being loaded.')
            for pname in list(pretrained_model['model_states'].keys()):
                assert pname[:6] == 'module' # a DataParallel model must start with module
                pretrained_model['model_states'][pname[7:]] = pretrained_model['model_states'].pop(pname)
        #
        if not (pretrained_model['model_states'].keys() == network.state_dict().keys()):
            self.logger.warn('==> network parameters in pre-trained file'
                            ' %s do not strictly match.' % (pretrained_path))
        freeze_pretrained = not not self.model_defs.get("freeze_pretrained", True)
        step = 0
        for pname, param in network.named_parameters():
            if pname in pretrained_model['model_states']:
                if param.data.size() == pretrained_model['model_states'][pname].data.size():
                    self.logger.debug('==> Copying parameter %s from file %s.' %
                                    (pname, pretrained_path))
                    param.data.copy_(pretrained_model['model_states'][pname])
                    param.requires_grad_(not freeze_pretrained)
                    step += 1
        #
        self.initialize_pretrained = True
        self.freeze_pretrained = freeze_pretrained
        if step > 0:
            self.logger.info('==> total %d network submodules are initialized with pretrained parameters.' % (step))
        else:
            self.logger.warn('==> none of the network submodules are initialized with pretrained parameters.')

    def get_trainable_parameters(self):
        # parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        # return parameters
        return NotImplementedError

    def init_optimizer(self):
        """
        Init optimizer for neural network model defined in `self.exp_dict["network"]`.
        If `self.exp_dict["optimizer"]` is not defined or `None`, no optimizer
        will be initialized for this model, i.e. the model can only work in
        evaluation mode. 
        """
        self.optim_defs = self.exp_dict.get("optimizer", None)
        if self.optim_defs is None:
            self.optimizer = None
            self.logger.warn("The model with no optimizer can not be updated.")
            return
        #
        self.logger.debug('Initialize optimizer: %s with params: %s'
            % (self.optim_defs["module"], self.optim_defs["param"]))
        optimizer = import_x(
            self.optim_defs["module"], 
            import_module(self.optim_defs["package"]))
        if optimizer is None:
            raise ValueError("Not supported or recognized optimizer: "+self.optim_defs["package"]+"."+self.optim_defs["module"])
        parameters = self.get_trainable_parameters()
        self.optimizer =  optimizer(parameters) if self.optim_defs["param"] is None else optimizer(parameters, **(self.optim_defs["param"]))
        self.init_optimizer_callback()
        # init scheduler
        self.init_lr_scheduler()

    def init_optimizer_callback(self):
        """
        The callback method is called right after optimizer construction and before learning
        rate scheduler construction.
        """
        pass

    
    def init_lr_scheduler(self):
        """
        Init learning rate scheduler for optimizer for neural network 
        model defined in `self.exp_dict["network"]`. If `self.exp_dict["scheduler"]`
        is not defined or `None`, no scheduler will be initialized, i.e.
        the model can only be updated with fixed learning rate when running
        in train mode.
        """
        self.sched_defs = self.exp_dict.get("scheduler", None)
        if self.sched_defs is None:
            self.scheduler = None
            self.logger.warn("The model with no scheduler can only be updated with fixed learning rate when running in train mode.")
            return

        # can only create scheduler for existed optimizer
        if self.optimizer is None:
            raise ValueError("Cannot create learning rate scheduler for network when no optimizer")
        
        self.logger.debug('Initialize lr scheduler: %s with params: %s.'
            % (self.sched_defs["module"], self.sched_defs["param"]))
        schedulers = import_x(
            self.sched_defs["module"], 
            import_module(self.sched_defs["package"]))
        if schedulers is None:
            raise ValueError("Not supported or recognized scheduler: "+self.sched_defs["package"]+"."+self.sched_defs["module"])
        # currently donot support ReduceLROnPlateau scheduler
        if 'onplateau' in schedulers.__name__.lower():
            raise TypeError("Currently ReduceLROnPlateau is not supported.")
        self.scheduler =  schedulers(self.optimizer) if self.sched_defs["param"] is None else schedulers(self.optimizer, **(self.sched_defs["param"]))
        for _ in range(self.curr_epoch): self.scheduler.step() # in case the model is initialized from checkpoint
        self.init_lr_scheduler_callback()
    
    def init_lr_scheduler_callback(self):
        """
        The callback method is called right after scheduler construction.
        """
        pass

    def load_to_gpu(self, parallel=False):
        """
        Transfer neural network model and initialized torch tensor(s) to GPU device(s).

        Args:\n
        - parallel: bool, set True to enable multi-gpu mode.

        Return the device context.
        """
        self.logger.debug("Load model and tensors into GPU devices" + (", enable multi-GPU mode." if parallel else "."))
        if parallel:
            # use DistributedDataParallel
            with torch.cuda.device(self.exp_dict['local_rank']):
                for key, tensor in self.tensors.items():
                    self.tensors[key] = tensor.cuda()
                self.model.cuda()
                self.dummy_input = self.dummy_input.cuda()
            self.parallel_model = torch.nn.parallel.DistributedDataParallel(self.model,device_ids=[self.exp_dict['local_rank']],output_device=self.exp_dict['local_rank'])
        else:
            for key, tensor in self.tensors.items():
                self.tensors[key] = tensor.cuda()
            self.model.cuda()
            self.dummy_input = self.dummy_input.cuda()
    
    @property
    def _model(self):
        return self.parallel_model if self.is_cuda_parallel else self.model
    
    @property
    def _is_main_process(self):
        return (self.is_cuda_parallel and torch.distributed.get_rank() ==0) or (not self.is_cuda_parallel)
    
    @property
    def _typename(self):
        return self.__class__.__name__
    
    @property
    def _is_write_summary(self):
        return not self.exp_dict.get('nosummary', False)

    def load_checkpoint(self, epoch, suffix=''):
        """
        Load the serialized model, and its associated optimizer and scheduler if possible.
        The method will use the serialized parameters to initialize model and return the
        checkpoint data.

        Args:\n
        - epoch: int, the epoch number, wildcard `*` to load the most recent epoch
        - suffix: str, suffix string of serialized files, default "".

        Return:\n
        The loaded checkpoint data.
        """
        if epoch is None: return None
        if epoch == '*':
            epoch = self.find_most_recent_epoch(suffix)
        if epoch < 1: return None
        if self.initialize_pretrained:
            self.logger.warn('Model is initialized with pretrained parameters, which will be overwritten by loading checkpoint.')
        chpt_file = self._get_checkpoint_filename(epoch) + suffix
        self.logger.info('Load checkpoint @ epoch %s from file %s.' % (str(epoch), chpt_file))
        if not os.path.isfile(chpt_file):
            self.logger.warn('Checkpoint file %s not found.'%(chpt_file))
            return None
        chpt_param = torch.load(chpt_file, map_location="cpu")
        assert chpt_param['model_classname'] == self.model.__class__.__name__
        assert chpt_param['model_states'].keys() == self.model.state_dict().keys()
        self.model.load_state_dict(chpt_param['model_states'])
        self.curr_epoch = epoch
        # load best record info
        assert self.keep_best_model_metric_name == chpt_param['best_rec_metric_name']
        self.best_metric_value = chpt_param['best_rec_metric_value']
        self.best_epoch = chpt_param['best_rec_epoch']
        self.best_stats = chpt_param['best_rec_stats']
        return chpt_param

    def find_most_recent_epoch(self, suffix):
        """
        Find the most recent epoch for neural network model.

        Args:\n
        - suffix: str, suffix string of the serialized model file.

        Output:\n
        `int`, the most recent epoch.
        """
        search_patern = self._get_checkpoint_filename('*') + suffix
        all_files = sorted(glob.glob(search_patern), key=os.path.getmtime) # order returned model by last modification time
        if len(all_files) == 0:
            raise ValueError('%s: no such file.' % (search_patern))

        substrings = search_patern.split('*')
        assert(len(substrings) == 2)
        start, end = substrings
        all_epochs = [fname.replace(start,'').replace(end,'') for fname in all_files]
        all_epochs = [int(epoch) for epoch in all_epochs if epoch.isdigit()]
        assert(len(all_epochs) > 0)
        all_epochs = sorted(all_epochs)
        most_recent_epoch = int(all_epochs[-1])
        self.logger.info('Load most recent checkpoint @ epoch %s' %
                         (str(most_recent_epoch)))
        return most_recent_epoch

    def save_checkpoint(self, epoch, suffix='', extra=None):
        """
        Serialize the neural network model.

        Args:\n
        - epoch: int, the last epoch.
        - suffix: str, suffix string of serialized files, default "".
        - extra: dict, the extra data to be serialized, default None.
        """
        # only main process can save checkpoint in multi-GPU multi-Host mode
        self.logger.info('Save checkpoint @ epoch %s' % (str(epoch)))
        state = {
            'epoch': epoch,
            'model_classname': self.model.__class__.__name__,
            'model_states': self.model.state_dict(),
            'best_rec_metric_name': self.keep_best_model_metric_name,
            'best_rec_metric_value': self.best_metric_value,
            'best_rec_epoch': self.best_epoch,
            'best_rec_stats': self.best_stats
        }
        if extra is not None:
            assert isinstance(extra, dict)
            # make sure all serialized tensors are on CPU devices --- DONOT NEED ANY MORE
            #for k,v in extra:
            #    if isinstance(v, torch.Tensor): extra[k]=v.cpu()
            state.update(extra)
        chpt_file = self._get_checkpoint_filename(epoch) + suffix
        torch.save(state, chpt_file)

    def delete_checkpoint(self, epoch, suffix=''):
        chpt_file = self._get_checkpoint_filename(epoch)+suffix
        if os.path.isfile(chpt_file): os.remove(chpt_file)

    def _get_checkpoint_filename(self, epoch):
        return os.path.join(self.exp_dir, self.exp_name + '_ckpt_epoch'+ str(epoch))
    
    def solve(self, data_loader_train, data_loader_val, data_loader_test=None):
        """Solve the model on train dataset and evaluate it on the validatin dataset.
        If the test dataset is provided, also evaluate the model on the test dataset.
        The latest model (solved on the train dataset) and the best model (evaluated 
        on the validation dataset) are kept in disk for future use. 
        """
        if self.optimizer is None:
            raise ValueError("Optimizer is missing so that the network cannot be optimized.")
        if self._is_main_process and self._is_write_summary:
            self.summary_writer = SummaryWriter(os.path.join(self.vis_dir, self.timestamp))
            self.summary_writer.add_graph(self.model, (self.dummy_input,), verbose=False)
        self.max_num_epochs = self.exp_dict['max_num_epochs']
        for self.curr_epoch in range(self.curr_epoch, self.max_num_epochs):
            if self.is_training_finished: break
            if self._is_main_process and self._is_write_summary: self.summary_writer.add_scalar('lr', self.curr_learning_rate, self.curr_epoch)
            self.logger.info('Training epoch [%4d / %4d]:' %
                             (self.curr_epoch + 1, self.max_num_epochs))
            train_stats = self.train_on_dataset(data_loader_train)
            if self._is_main_process and self._is_write_summary: self.write_stats_summary(train_stats, 'train')
            self.logger.info('==> Training stats [%4d / %4d]: %s' % (self.curr_epoch + 1, self.max_num_epochs, train_stats))
            if self._is_main_process: self.save_checkpoint(self.curr_epoch+1)
            # Synchronizes all processes.
            if self.is_cuda_parallel: torch.distributed.barrier()
            if self.curr_epoch >0 and self._is_main_process: self.delete_checkpoint(self.curr_epoch)
            if self.curr_epoch % self.exp_dict['evaluate_step'] ==0 or self.curr_epoch == self.max_num_epochs-1:
                # self.logger.info('Evaluation epoch [%4d / %4d]:' %
                #              (self.curr_epoch + 1, self.max_num_epochs))
                eval_stats = self.eval_on_dataset(data_loader_val)
                if self._is_main_process and self._is_write_summary: self.write_stats_summary(eval_stats, 'val')
                self.logger.info('==> Evaluation stats [%4d / %4d]: %s' % (self.curr_epoch + 1, self.max_num_epochs, eval_stats))
                if self.is_best(eval_stats): self.keep_record_of_best_model(eval_stats)
                if data_loader_test is not None:
                    # self.logger.info('Testing epoch [%4d / %4d]:' %
                    #          (self.curr_epoch + 1, self.max_num_epochs))
                    test_stats = self.eval_on_dataset(data_loader_test)
                    if self._is_main_process and self._is_write_summary: self.write_stats_summary(test_stats, 'test')
                    self.logger.info('Testing stats [%4d / %4d]: %s' %
                                (self.curr_epoch + 1, self.max_num_epochs, test_stats))
            else:
                self.logger.info('Evaluation stats [%4d / %4d]: skipped.' %
                             (self.curr_epoch + 1, self.max_num_epochs))
                if data_loader_test is not None:
                    self.logger.info('Testing stats [%4d / %4d]: skipped.' %
                                (self.curr_epoch + 1, self.max_num_epochs))
            self.adjust_learning_rate(stats=train_stats)
            # Synchronizes all processes.
            if self.is_cuda_parallel: torch.distributed.barrier()
        self.print_eval_stats_of_best_model()
        if self._is_main_process and self._is_write_summary: self.summary_writer.close()
    
    def evaluate(self, data_loader_test):
        eval_stats = self.eval_on_dataset(data_loader_test)
        if self.initialize_pretrained and self.curr_epoch==0:
            self.logger.info("Evaluation upon pretrained parameters: %s"%(eval_stats))
        elif self.curr_epoch > 0:
            self.logger.info("Evaluation upon checkpoint @ epoch [%4d]: %s" %(self.curr_epoch, eval_stats))
        else:
            self.logger.info("Evaluation upon ramdon initialized parameters: %s" %(eval_stats))

    def init_record_of_best_model(self):
        self.best_metric_value = None
        self.best_stats = None
        self.best_epoch = None
    
    def is_better(self, statsA, statsB):
        """This is the rules for selecting a better model.
        """
        metric_name = self.keep_best_model_metric_name
        assert metric_name in statsA
        assert metric_name in statsB
        if 'loss' in metric_name.lower():
            return statsA[metric_name] < statsB[metric_name]
        else:
            return statsA[metric_name] > statsB[metric_name]
    
    def is_best(self, stats):
        metric_name = self.keep_best_model_metric_name
        assert metric_name in stats
        if self.best_stats is None: # 1st epoch
            return True
        else:
            return self.is_better(stats, self.best_stats)

    def keep_record_of_best_model(self, eval_stats):
        metric_name = self.keep_best_model_metric_name
        pre_best_epoch = self.best_epoch
        self.best_metric_value = eval_stats[metric_name]
        self.best_stats = eval_stats
        self.best_epoch = self.curr_epoch + 1
        chpt_file = self._get_checkpoint_filename(self.best_epoch)+".best"
        if (not os.path.isfile(chpt_file)) or self._is_main_process:
            self.save_checkpoint(self.best_epoch, suffix='.best')
        else:
            pass
        if pre_best_epoch is not None:
            try:
                self.delete_checkpoint(pre_best_epoch, suffix='.best')
            except Exception as e:
                pass 
        self.print_eval_stats_of_best_model()

    def print_eval_stats_of_best_model(self):
        if self.best_stats is not None:
            metric_name = self.keep_best_model_metric_name
            self.logger.info('==> Best results w.r.t. %s metric @ epoch %d - %s'
                             % (metric_name, self.best_epoch, self.best_stats))
    
    @property
    def is_training_finished(self):
        if self.curr_epoch >= self.max_num_epochs: return True
        #
        assert self.optimizer is not None
        lrs = [group['lr'] for group in self.optimizer.param_groups]
        defaults = self.optimizer.defaults
        return lrs[0] < (defaults['lr'] * self.min_lr_decay)
    
    @property
    def curr_learning_rate(self):
        return self.get_lr()

    def get_lr(self):
        optimizer = getattr(self, 'optimizer', None)
        if optimizer is None:
            return self.optim_defs['param']['lr']
        else:
            return optimizer.param_groups[0]['lr']
        
    
    def write_stats_summary(self, stats, phase):
        """Write train/val/test stats into tensorboard summary.

        Args:
        @param stats, the running stats.
        @param phase, the phase name, must be one of 'train', 'val' and 'test'.
        """
        pass

    def adjust_learning_rate(self, **kwargs):
        """
        Adjust the training learning rate.
        """
        if self.scheduler is None: return
        self.scheduler.step()

    def train_on_dataset(self, data_loader):
        self._model.train()
        set_random_seeds(self.curr_epoch) # train on a different subset each time and all epoches come in static order.
        train_stats = DAverageMeter()
        for i, batch in enumerate(tqdm(data_loader) if self.is_tqdm else data_loader):
            train_stats_this = self.train_on_batch(batch)
            train_stats.update(train_stats_this)
        return train_stats.average()

    def train_on_batch(self, batch):
        raise NotImplementedError

    def eval_on_dataset(self, data_loader):
        def reset_confidence_interval_95():
            try:
                del self.eval_acc
            except AttributeError as e:
                pass
            self.eval_acc = {"Top1Acc":[], "Top5Acc":[]}
        #
        reset_confidence_interval_95()
        self._model.eval()
        set_random_seeds(0) # always evaluate on the same dataset.
        eval_stats = DAverageMeter()
        n = len(data_loader)
        with torch.no_grad():
            for i, batch in enumerate(tqdm(data_loader) if self.is_tqdm else data_loader):
                eval_stats_this = self.eval_on_batch(batch)
                eval_stats.update(eval_stats_this)
                self.eval_acc['Top1Acc'].append(eval_stats_this['Top1Acc'])
                self.eval_acc['Top5Acc'].append(eval_stats_this['Top5Acc'])
                if (i+1) == n:
                    for k,v in self.eval_acc.items():
                        stds = torch.std(torch.tensor(v).float())
                        ci95 = 1.96*stds/torch.sqrt(torch.tensor(n).float())
                        eval_stats.update({k+'_std': stds.item(), k+'_cnf95': ci95.item()})
        return eval_stats.average()

    def eval_on_batch(self, batch):
        raise NotImplementedError