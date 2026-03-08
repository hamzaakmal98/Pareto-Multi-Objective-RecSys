import json
from pathlib import Path
from typing import Dict, List
import time

import torch
from torch.utils.data import DataLoader
import logging

from src.models.encoder import SharedEncoder
from src.models.custom_mmoe import CustomMMoE
from src.models.heads import PredictionHeads
from src.models.losses import MultiTaskLoss
from src.models.dataset import InteractionDataset
from src.utils.io import ensure_dir, write_json


class Trainer:
    def __init__(self, config: Dict):
        self.config = config
        # support configs where training options are nested under 'training'
        training_cfg = config.get('training', {}) if isinstance(config.get('training', {}), dict) else {}
        self.device = torch.device('cuda' if torch.cuda.is_available() and training_cfg.get('use_cuda', config.get('use_cuda', True)) else 'cpu')

        # load dataset
        data_root = Path(config.get('data_root', '.'))
        train_path = Path(data_root) / 'data' / 'processed' / 'train.parquet'
        val_path = Path(data_root) / 'data' / 'processed' / 'val.parquet'

        # feature lists from config
        cat_cols = config['features'].get('categorical', [])
        num_cols = config['features'].get('numeric', [])
        target_cols = config['features'].get('targets', [])

        self.train_ds = InteractionDataset(train_path, [], cat_cols, num_cols, target_cols)
        self.val_ds = InteractionDataset(val_path, [], cat_cols, num_cols, target_cols)

        batch_size = training_cfg.get('batch_size', config.get('batch_size', 1024))
        self.train_loader = DataLoader(self.train_ds, batch_size=batch_size, shuffle=True, collate_fn=self.collate)
        self.val_loader = DataLoader(self.val_ds, batch_size=batch_size, shuffle=False, collate_fn=self.collate)

        # build model components
        emb_conf = config.get('embeddings', {})
        numeric_dim = len(num_cols)
        model_cfg = config.get('model', {}) if isinstance(config.get('model', {}), dict) else {}
        proj_dim = model_cfg.get('projection_dim', config.get('projection_dim', 128))
        self.encoder = SharedEncoder(emb_conf, numeric_dim, proj_dim).to(self.device)
        self.mmoe = CustomMMoE(input_dim=proj_dim, n_experts=model_cfg.get('n_experts', config.get('n_experts', 4)), expert_hidden=model_cfg.get('expert_hidden', config.get('expert_hidden', 128)), tower_hidden=model_cfg.get('tower_hidden', config.get('tower_hidden', 64)), use_task_residual=model_cfg.get('use_task_residual', config.get('use_task_residual', True))).to(self.device)
        self.heads = PredictionHeads(input_dim=1, use_creator=model_cfg.get('use_creator', config.get('use_creator', True))).to(self.device)

        # loss
        weights = config.get('loss_weights', {'like':1.0, 'longview':1.0, 'creator':1.0})
        self.loss_fn = MultiTaskLoss(weights)

        lr = training_cfg.get('lr', config.get('lr', 1e-3))
        self.optimizer = torch.optim.Adam(list(self.encoder.parameters()) + list(self.mmoe.parameters()) + list(self.heads.parameters()), lr=lr)

        # checkpoints
        self.out_dir = ensure_dir(Path(config.get('out_dir', './artifacts')))
        self.checkpoint_dir = ensure_dir(self.out_dir / 'checkpoints')
        self.history = []

        self.early_stop_patience = training_cfg.get('early_stop_patience', config.get('early_stop_patience', 5))

    def collate(self, batch):
        # batch is list of (cat_inputs, numeric, targets)
        batch_size = len(batch)
        # collect categorical tensors into dict of LongTensor (B,)
        cat_keys = set()
        for cat_inputs, _, _ in batch:
            cat_keys.update(cat_inputs.keys())
        cat_batch = {k: torch.tensor([bi[0].get(k, torch.tensor(0)).item() if isinstance(bi[0].get(k), torch.Tensor) else bi[0].get(k, 0) for bi in batch], dtype=torch.long) for k in cat_keys}
        num_batch = torch.stack([bi[1] if bi[1].numel()>0 else torch.zeros(1) for bi in batch]).float()
        targets = {}
        for t in batch[0][2].keys():
            targets[t] = torch.tensor([bi[2].get(t).item() if isinstance(bi[2].get(t), torch.Tensor) else (bi[2].get(t) if bi[2].get(t) is not None else 0) for bi in batch]).float()
        return cat_batch, num_batch, targets

    def train_epoch(self):
        self.encoder.train(); self.mmoe.train(); self.heads.train()
        epoch_loss = 0.0
        for i, (cat_batch, num_batch, targets) in enumerate(self.train_loader):
            # move to device
            cat_inputs = {k: v.to(self.device) for k, v in cat_batch.items()}
            num_batch = num_batch.to(self.device)
            targets = {k: v.to(self.device) for k, v in targets.items()}

            shared = self.encoder(cat_inputs, num_batch)
            preds = self.mmoe(shared)
            logits = {'like': preds['like'], 'longview': preds['longview']}
            if 'creator' in preds:
                logits['creator'] = preds['creator']

            loss, details = self.loss_fn(logits, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_loss += float(loss.item())

        return epoch_loss / len(self.train_loader)

    def validate(self):
        self.encoder.eval(); self.mmoe.eval(); self.heads.eval()
        val_loss = 0.0
        details_acc = {}
        with torch.no_grad():
            for cat_batch, num_batch, targets in self.val_loader:
                cat_inputs = {k: v.to(self.device) for k, v in cat_batch.items()}
                num_batch = num_batch.to(self.device)
                targets = {k: v.to(self.device) for k, v in targets.items()}
                shared = self.encoder(cat_inputs, num_batch)
                preds = self.mmoe(shared)
                logits = {'like': preds['like'], 'longview': preds['longview']}
                if 'creator' in preds:
                    logits['creator'] = preds['creator']

                loss, details = self.loss_fn(logits, targets)
                val_loss += float(loss.item())
                for k, v in details.items():
                    details_acc[k] = details_acc.get(k, 0.0) + v

        n = len(self.val_loader)
        avg_details = {k: v / n for k, v in details_acc.items()}
        return val_loss / n, avg_details

    def fit(self, epochs: int = 20):
        best_val = float('inf')
        best_epoch = -1
        logger = logging.getLogger(__name__)
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss = self.train_epoch()
            val_loss, val_details = self.validate()
            self.history.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss, 'val_details': val_details})
            logger.info("Epoch %d  train_loss=%.4f  val_loss=%.4f", epoch, train_loss, val_loss)

            # checkpoint
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch
                ckpt_path = self.checkpoint_dir / f"best_model_epoch_{epoch}.pt"
                torch.save({'encoder': self.encoder.state_dict(), 'mmoe': self.mmoe.state_dict(), 'heads': self.heads.state_dict(), 'config': self.config}, ckpt_path)
                logger.info('Saved best model: %s', ckpt_path)

            # early stopping
            if epoch - best_epoch >= self.early_stop_patience:
                logger.info('Early stopping triggered (no improvement for %d epochs)', self.early_stop_patience)
                break

        # write history
        write_json(self.out_dir / 'training_history.json', self.history)
        logger.info('Training complete. Best epoch: %s', best_epoch)
