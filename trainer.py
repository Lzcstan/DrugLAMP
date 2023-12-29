import torch
import torchmetrics as M
import pytorch_lightning as pl

from rich import print
from typing import Any, List, Optional, Union
from scheduler import CosineAnnealingWarmupRestarts
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from model.basic_model import binary_cross_entropy, cross_entropy_logits

from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.functional.classification.auroc import _binary_auroc_compute
from torchmetrics.classification.precision_recall_curve import BinaryPrecisionRecallCurve
from torchmetrics.functional.classification.average_precision import _binary_average_precision_compute

class BinaryAUSum(BinaryPrecisionRecallCurve):
    is_differentiable: bool = False
    higher_is_better: Optional[bool] = None
    full_state_update: bool = False

    def __init__(
        self,
        max_fpr: Optional[float] = None,
        thresholds: Optional[Union[int, List[float], torch.Tensor]] = None,
        ignore_index: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(thresholds=thresholds, ignore_index=ignore_index, validate_args=False, **kwargs)
        self.max_fpr = max_fpr

    def compute(self) -> torch.Tensor:
        if self.thresholds is None:
            state = [dim_zero_cat(self.preds), dim_zero_cat(self.target)]
        else:
            state = self.confmat
        return _binary_auroc_compute(state, self.thresholds, self.max_fpr) + _binary_average_precision_compute(state, self.thresholds)

class ExpModule(pl.LightningModule):
    def __init__(self, model, opt, train_dl, val_dl, test_dl,
                opt_ssl=None, opt_cm=None, logger=None, split='random', **config):
        super().__init__()
        # Lightning config
        self.automatic_optimization = False
        self.log_step = 10

        # Logger
        self.comet_logger = logger

        # Config
        self.config = config
        self.n_class = config["DECODER"]["BINARY"]
        self.seed = config['SOLVER']['SEED']
        self.epochs = config["SOLVER"]["MAX_EPOCH"]
        self.batch_size = config["SOLVER"]["BATCH_SIZE"]
        self.output_dir = config["RESULT"]["OUTPUT_DIR"]
        self.max_lr = config['SOLVER']['LR']
        self.max_ssl_lr = config['SOLVER']['SSL_LR']
        self.max_cm_lr = config['SOLVER']['CM_LR']
        self.use_ssl = config["RS"]["SSL"]
        if self.use_ssl and opt_ssl is None:
            print('Please offer optimizer for SSL!!!')
            self.use_ssl = False
        self.use_cm = config["RS"]["CM"]
        if self.use_cm and opt_cm is None:
            print('Please offer optimizer for CrossModality!!!')
            self.use_cm = False
        self.ssl_epoch_step = config["RS"]["EPOCH_STEP"]
        self.cm_init_epoch = config["RS"]["INIT_EPOCH"]

        # Data
        self.split = split
        self.exp_train_dl = train_dl
        self.exp_val_dl = val_dl
        self.exp_test_dl = test_dl
        self.n_batch_train = len(self.exp_train_dl)

        # Model
        self.exp_model = model

        # Optimizer and scheduler
        self.opt = opt
        self.opt_ssl = opt_ssl
        self.opt_cm = opt_cm
        self.schd = CosineAnnealingWarmupRestarts(
            optimizer=self.opt,
            first_cycle_steps=self.epochs,
            max_lr=self.max_lr,
            min_lr=1e-8,
            warmup_steps=int(self.epochs * 0.2)
        )
        self.schd_ssl = CosineAnnealingWarmupRestarts(
            optimizer=self.opt_ssl,
            first_cycle_steps=self.epochs,
            max_lr=self.max_ssl_lr,
            min_lr=1e-8,
            warmup_steps=int(self.epochs * 0.2)
        ) if self.use_ssl else None
        self.schd_cm = CosineAnnealingWarmupRestarts(
            optimizer=self.opt_cm,
            first_cycle_steps=int(self.epochs / self.ssl_epoch_step),
            max_lr=self.max_cm_lr,
            min_lr=1e-8,
            warmup_steps=int(self.epochs * 0.2 / self.ssl_epoch_step)
        ) if self.use_cm else None
        self.cm_weight = 1

        # Metrics
        self.val_auroc = M.AUROC(task='binary')
        self.val_auprc = M.AveragePrecision(task='binary')
        self.val_ausum = BinaryAUSum()

        self.test_auroc = M.AUROC(task='binary')
        self.test_auprc = M.AveragePrecision(task='binary')
        self.test_acc = M.Accuracy(task='binary')
        self.test_sn = M.Recall(task='binary')
        self.test_sp = M.Specificity(task='binary')
        self.test_f1 = M.F1Score(task='binary')
        self.test_pr = M.Precision(task='binary')

    def configure_optimizers(self):
        if self.use_cm and self.use_ssl:
            return [self.opt, self.opt_ssl, self.opt_cm], [self.schd, self.schd_ssl, self.schd_cm]
        elif self.use_ssl:
            return [self.opt, self.opt_ssl], [self.schd, self.schd_ssl]
        elif self.use_cm:
            return [self.opt, self.opt_cm], [self.schd, self.schd_cm]
        else:
            return [self.opt], [self.schd]

    def run_experiment(self):
        self.set_exp_trainer()
        self.exp_trainer.fit(model=self, train_dataloaders=self.exp_train_dl, val_dataloaders=self.exp_val_dl)
        self.load_state_dict(torch.load(self.exp_trainer.checkpoint_callback.best_model_path)['state_dict'], strict=False)
        self.exp_trainer.test(model=self, dataloaders=self.exp_test_dl)

    def run_fast_development(self, single=False):
        self.set_fast_dev_trainer(single=single)
        self.fast_dev_trainer.fit(model=self, train_dataloaders=self.exp_train_dl, val_dataloaders=self.exp_val_dl)

    def set_exp_trainer(self):
        pl.seed_everything(self.seed, workers=True)
        self.exp_trainer = pl.Trainer(
            max_epochs=self.epochs,
            log_every_n_steps=self.log_step,
            accelerator='auto',
            strategy='ddp_find_unused_parameters_true',
            logger=self.comet_logger,
            check_val_every_n_epoch=1,
            callbacks=[
                ModelCheckpoint(
                    monitor='val_ausum',
                    filename='max_{val_ausum: .5f}',
                    mode='max',
                    # save_last=True
                ),
                # ModelCheckpoint(filename='last_{epoch}'),
                EarlyStopping(
                    monitor='val_ausum',
                    mode='max',
                    patience=int(self.epochs / 4),
                )
            ]
        )

    def set_fast_dev_trainer(self, single=False):
        self.fast_dev_trainer = pl.Trainer(
            devices=1 if single else 'auto',
            strategy='auto' if single else 'ddp_find_unused_parameters_true',
            log_every_n_steps=self.log_step,
            fast_dev_run=self.log_step,
        )
    
    def on_before_batch_transfer(self, batch, dataloader_idx: int):
        if self.trainer.training:
            self.meta = batch[5]
        return batch[: 5]

    def training_step(self, batch, batch_idx):
        cur_epoch = self.current_epoch + 1
        if self.use_cm and self.use_ssl:
            opt, opt_ssl, opt_cm = self.optimizers()
        elif self.use_ssl:
            opt, opt_ssl = self.optimizers()
            opt_cm = None
        elif self.use_cm:
            opt, opt_cm = self.optimizers()
            opt_ssl = None
        else:
            opt = self.optimizers()
            opt_ssl, opt_cm = None, None
        compute_ssl = (cur_epoch % self.ssl_epoch_step == 0 and self.use_ssl)
        compute_cm = (cur_epoch >= self.cm_init_epoch and self.use_cm)

        feat_d, feat_p, labels, llm_d, llm_p = batch
        feat_d, feat_p, ssl_input, cm_input, score = self.exp_model(feat_d, feat_p, llm_d, llm_p)

        opt.zero_grad()
        _, cls_loss = binary_cross_entropy(score, labels) if (self.n_class == 1) else cross_entropy_logits(score, labels)
        self.manual_backward(cls_loss, retain_graph=True) if (compute_ssl or compute_ssl) else self.manual_backward(cls_loss)
        self.log('train_loss', cls_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True, prog_bar=True)
        loss = cls_loss
        if compute_ssl:
            opt_ssl.zero_grad()
            ssl_loss_dict = self.exp_model.ssl_model(**ssl_input)
            ssl_loss = (ssl_loss_dict['prot_ssl'] + ssl_loss_dict['drug_ssl']) * 0.1
            self.manual_backward(ssl_loss, retain_graph=True) if (compute_cm) else self.manual_backward(ssl_loss)
            self.log('ssl_loss', ssl_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True, prog_bar=True)
            loss += ssl_loss
        if compute_cm:
            opt_cm.zero_grad()
            cm_input['meta'] = self.meta
            cm_loss = self.exp_model.cm_model(**cm_input)
            if cur_epoch == self.cm_init_epoch and cm_loss.item() > 0:
                while cm_loss.item() * self.cm_weight / 10 > cls_loss.item():
                    self.cm_weight /= 10
                while cm_loss.item() * self.cm_weight * 10 < cls_loss.item():
                    self.cm_weight *= 10
            cm_loss = cm_loss * self.cm_weight
            self.manual_backward(cm_loss)
            self.log('cm_loss', cm_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True, prog_bar=True)
            loss += cm_loss
        
        opt.step()
        if compute_ssl:
            opt_ssl.step()
        if compute_cm:
            opt_cm.step()

        self.log('all_loss', loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)

    def on_train_epoch_end(self):
        cur_epoch = self.current_epoch + 1
        if self.use_cm and self.use_ssl:
            schd, schd_ssl, schd_cm = self.lr_schedulers()
        elif self.use_ssl:
            schd, schd_ssl = self.lr_schedulers()
            schd_cm = None
        elif self.use_cm:
            schd, schd_cm = self.lr_schedulers()
            schd_ssl = None
        else:
            schd = self.lr_schedulers()
            schd_ssl, schd_cm = None, None
        compute_ssl = (cur_epoch % self.ssl_epoch_step == 0 and self.use_ssl)
        compute_cm = (cur_epoch >= self.cm_init_epoch and self.use_cm)

        schd.step()
        if compute_ssl:
            schd_ssl.step()
        if compute_cm:
            schd_cm.step()
            self.exp_model.cm_model.step()

    def validation_step(self, batch, batch_idx):
        feat_d, feat_p, labels, llm_d, llm_p = batch
        feat_d, feat_p, _, _, score = self.exp_model(feat_d, feat_p, llm_d, llm_p)

        n, cls_loss = binary_cross_entropy(score, labels) if (self.n_class == 1) else cross_entropy_logits(score, labels)

        self.val_auroc.update(n, labels.long())
        self.val_auprc.update(n, labels.long())
        self.val_ausum.update(n, labels.long())

        self.log('val_loss', cls_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log('val_auroc', self.val_auroc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_auprc', self.val_auprc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_ausum', self.val_ausum, on_step=False, on_epoch=True, logger=True)

    def test_step(self, batch, batch_idx):
        feat_d, feat_p, labels, llm_d, llm_p = batch
        feat_d, feat_p, _, _, score = self.exp_model(feat_d, feat_p, llm_d, llm_p)

        n, cls_loss = binary_cross_entropy(score, labels) if (self.n_class == 1) else cross_entropy_logits(score, labels)
        
        self.test_auroc.update(n, labels.long())
        self.test_auprc.update(n, labels.long())
        self.test_acc.update(n, labels.long())
        self.test_sn.update(n, labels.long())
        self.test_sp.update(n, labels.long())
        self.test_f1.update(n, labels.long())
        self.test_pr.update(n, labels.long())

        self.log('test_loss', cls_loss, on_step=False, on_epoch=True, logger=True, sync_dist=True)
        self.log('test_auroc', self.test_auroc, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('test_auprc', self.test_auprc, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('test_acc', self.test_acc, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('test_sn', self.test_sn, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('test_sp', self.test_sp, on_step=False, on_epoch=True, logger=True, prog_bar=True)
        self.log('test_f1', self.test_f1, on_step=False, on_epoch=True, logger=True)
        self.log('test_pr', self.test_pr, on_step=False, on_epoch=True, logger=True)