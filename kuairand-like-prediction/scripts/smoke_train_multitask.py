import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score
import json
import time


def load_processed(root: Path):
    p = root / 'data' / 'processed'
    X = pd.read_csv(p / 'X.csv')
    y = pd.read_csv(p / 'y.csv')
    # read splits
    def read_idx(fn):
        fp = p / fn
        if not fp.exists():
            return np.array([], dtype=int)
        try:
            arr = pd.read_csv(fp, header=None).values.flatten()
        except Exception:
            arr = np.loadtxt(fp, delimiter=',', dtype=int)
        return arr.astype(int)

    train_idx = read_idx('train_idx.csv')
    val_idx = read_idx('val_idx.csv')
    test_idx = read_idx('test_idx.csv')
    return X, y, train_idx, val_idx, test_idx


class SimpleMultiHead(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(0.1))
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Linear(hidden // 2, 1)) for _ in range(3)])

    def forward(self, x):
        h = self.shared(x)
        outs = [head(h).squeeze(-1) for head in self.heads]
        return torch.stack(outs, dim=1)


def to_tensor(df):
    return torch.tensor(df.values, dtype=torch.float32)


def run(root='kuairand-like-prediction', epochs=3, batch_size=256, lr=1e-3, pos_weight_is_like: bool = False, loss_weights: dict = None, early_stopping_patience: int = 3):
    root = Path(root)
    X, y, train_idx, val_idx, test_idx = load_processed(root)

    # drop index-like columns
    drop_cols = [c for c in ['Unnamed: 0', 'split', 'user_id', 'video_id'] if c in X.columns]
    X_model = X.drop(columns=drop_cols, errors='ignore').fillna(0.0)

    # targets: ensure order for three-task setup
    targets = [t for t in ['is_click', 'is_like', 'long_view'] if t in y.columns]
    y_model = y[targets].fillna(0).astype(float)

    # prepare tensors
    X_t = to_tensor(X_model)
    y_t = torch.tensor(y_model.values, dtype=torch.float32)

    train_idx = np.array(train_idx, dtype=int)
    val_idx = np.array(val_idx, dtype=int)
    test_idx = np.array(test_idx, dtype=int)

    train_ds = TensorDataset(X_t[train_idx], y_t[train_idx])
    val_ds = TensorDataset(X_t[val_idx], y_t[val_idx])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleMultiHead(in_dim=X_t.shape[1], hidden=64).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    # create per-task loss functions; optionally apply positive class weighting for is_like
    criterions = []
    for t in targets:
        if t == 'is_like' and pos_weight_is_like:
            # compute pos weight from training data
            try:
                pos = float(y_model[t].iloc[train_idx].sum())
                neg = float(len(train_idx) - pos)
                pos_weight = torch.tensor([neg / pos]) if pos > 0 else torch.tensor([1.0])
            except Exception:
                pos_weight = torch.tensor([1.0])
            criterions.append(nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(torch.float32)))
        else:
            criterions.append(nn.BCEWithLogitsLoss())

    artifact_dir = root / 'artifacts'
    (artifact_dir / 'models').mkdir(parents=True, exist_ok=True)
    (artifact_dir / 'predictions').mkdir(parents=True, exist_ok=True)
    (root / 'reports' / 'analysis').mkdir(parents=True, exist_ok=True)

    history = {'train_loss': [], 'val_metrics_per_epoch': []}
    loss_weights = loss_weights or {'is_click': 1.0, 'is_like': 1.0, 'long_view': 1.0}
    best_metric = -1.0
    best_epoch = 0
    patience = early_stopping_patience
    no_improve = 0
    for ep in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        nb = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad()
            out = model(xb)  # shape (B, T)
            loss = 0.0
            # weighted per-task loss
            for i in range(out.shape[1]):
                task = targets[i]
                l = criterions[i](out[:, i], yb[:, i])
                w = float(loss_weights.get(task, 1.0))
                loss = loss + w * l
            denom = sum([float(loss_weights.get(t, 1.0)) for t in targets])
            loss = loss / denom if denom else loss
            loss.backward()
            optim.step()
            total_loss += float(loss.item()) * xb.size(0)
            nb += xb.size(0)
        avg_loss = total_loss / nb if nb else 0.0
        history['train_loss'].append(avg_loss)

        # validation
        model.eval()
        preds = []
        truths = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                out = model(xb)
                preds.append(out.cpu().numpy())
                truths.append(yb.numpy())
        if preds:
            preds = np.vstack(preds)
            truths = np.vstack(truths)
        else:
            preds = np.zeros((0, len(targets)))
            truths = np.zeros((0, len(targets)))

        # compute metrics per task
        val_metrics = {}
        for i, t in enumerate(targets):
            scores = 1 / (1 + np.exp(-preds[:, i])) if preds.size else np.array([])
            y_true = truths[:, i] if truths.size else np.array([])
            if y_true.size and len(np.unique(y_true)) > 1:
                acc = float(accuracy_score((scores > 0.5).astype(int), y_true))
                try:
                    auc = float(roc_auc_score(y_true, scores))
                except Exception:
                    auc = None
                try:
                    prauc = float(average_precision_score(y_true, scores))
                except Exception:
                    prauc = None
            elif y_true.size:
                acc = float(accuracy_score((scores > 0.5).astype(int), y_true))
                auc = None
                prauc = None
            else:
                acc = None
                auc = None
                prauc = None
            val_metrics[t] = {'accuracy': acc, 'auc': auc, 'pr_auc': prauc}

        # aggregate metric for early stopping (average PR-AUC across tasks that have PR-AUC)
        sum_prauc = 0.0
        count_prauc = 0
        for t in targets:
            if val_metrics[t].get('pr_auc') is not None:
                sum_prauc += val_metrics[t]['pr_auc']
                count_prauc += 1
        avg_prauc = (sum_prauc / count_prauc) if count_prauc else 0.0

        print(f'Epoch {ep}/{epochs}  train_loss={avg_loss:.4f}  avg_pr_auc={avg_prauc:.4f}')
        for t, m in val_metrics.items():
            a = f"acc={m['accuracy']:.4f}" if m['accuracy'] is not None else 'acc=None'
            b = f", auc={m['auc']:.4f}" if m['auc'] is not None else ''
            c = f", pr_auc={m['pr_auc']:.4f}" if m['pr_auc'] is not None else ''
            print(f'  val {t}: {a}{b}{c}')

        # record per-epoch val metrics
        history['val_metrics_per_epoch'].append(val_metrics)

        # early stopping and best-checkpoint saving based on avg_pr_auc
        if avg_prauc > best_metric:
            best_metric = avg_prauc
            best_epoch = ep
            no_improve = 0
            # save best checkpoint and predictions
            best_ckpt = artifact_dir / 'models' / 'best_weighted_checkpoint.pth'
            torch.save({'model_state': model.state_dict(), 'optimizer_state': optim.state_dict(), 'epoch': ep, 'val_metrics': val_metrics, 'targets': targets, 'loss_weights': loss_weights}, best_ckpt)
            # save predictions CSV for this epoch
            pred_df = pd.DataFrame(preds, columns=[f'pred_score_{t}' for t in targets])
            truth_df = pd.DataFrame(truths, columns=[f'true_{t}' for t in targets])
            out_df = pd.concat([pd.Series(val_idx, name='index').reset_index(drop=True), truth_df.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
            out_df.to_csv(artifact_dir / 'predictions' / 'best_weighted_val_predictions.csv', index=False)
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping at epoch {ep} (no improvement in {patience} epochs)')
                break

    # save last checkpoint
    ckpt_path = artifact_dir / 'models' / 'smoke_checkpoint.pth'
    torch.save({'model_state': model.state_dict(), 'optimizer_state': optim.state_dict(), 'targets': targets}, ckpt_path)

    # Save best validation predictions (if saved during training as best)
    best_pred_path = artifact_dir / 'predictions' / 'best_weighted_val_predictions.csv'
    if best_pred_path.exists():
        best_preds_df = pd.read_csv(best_pred_path)
    else:
        # fallback: save final epoch preds
        pred_df = pd.DataFrame(preds, columns=[f'pred_score_{t}' for t in targets])
        truth_df = pd.DataFrame(truths, columns=[f'true_{t}' for t in targets])
        out_df = pd.concat([pd.Series(val_idx, name='index').reset_index(drop=True), truth_df.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
        out_path = artifact_dir / 'predictions' / 'smoke_val_predictions.csv'
        out_df.to_csv(out_path, index=False)
        best_preds_df = out_df

    # write training history
    with (artifact_dir / 'models' / 'training_history.json').open('w', encoding='utf-8') as fh:
        json.dump(history, fh, indent=2)

    # prepare metrics CSV per epoch
    rows = []
    for i, vm in enumerate(history['val_metrics_per_epoch'], start=1):
        row = {'epoch': i}
        for t in targets:
            row[f'{t}_auc'] = vm.get(t, {}).get('auc')
            row[f'{t}_pr_auc'] = vm.get(t, {}).get('pr_auc')
        rows.append(row)
    metrics_df = pd.DataFrame(rows)
    tables_dir = artifact_dir / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_csv = tables_dir / 'weighted_training_metrics.csv'
    metrics_df.to_csv(metrics_csv, index=False)

    # write weighted training summary markdown
    md_lines = ['# Weighted Training Run', '']
    md_lines.append(f'- best_epoch: {best_epoch}')
    md_lines.append(f'- train_rows: {len(train_idx)}')
    md_lines.append(f'- val_rows: {len(val_idx)}')
    md_lines.append('')
    md_lines.append('## Best validation metrics')
    # read best metrics from saved best checkpoint file if exists
    best_ckpt = artifact_dir / 'models' / 'best_weighted_checkpoint.pth'
    if best_ckpt.exists():
        try:
            bk = torch.load(best_ckpt, map_location='cpu')
            best_metrics = bk.get('val_metrics', {})
        except Exception:
            best_metrics = val_metrics
    else:
        best_metrics = val_metrics
    for t, m in best_metrics.items():
        md_lines.append(f"- {t}: accuracy={m.get('accuracy')}, auc={m.get('auc')}, pr_auc={m.get('pr_auc')}")

    # Load best checkpoint and compute test predictions/metrics
    test_metrics = {}
    best_ckpt = artifact_dir / 'models' / 'best_weighted_checkpoint.pth'
    if best_ckpt.exists():
        try:
            bk = torch.load(best_ckpt, map_location=device)
            model.load_state_dict(bk['model_state'])
            model.eval()
            # compute test preds
            X_test_t = X_t[test_idx].to(device)
            with torch.no_grad():
                out_test = model(X_test_t).cpu().numpy()
            scores_test = 1 / (1 + np.exp(-out_test))
            truth_test = y_t[test_idx].cpu().numpy()
            pred_df = pd.DataFrame(scores_test, columns=[f'pred_score_{t}' for t in targets])
            truth_df = pd.DataFrame(truth_test, columns=[f'true_{t}' for t in targets])
            out_test_df = pd.concat([pd.Series(test_idx, name='index').reset_index(drop=True), truth_df.reset_index(drop=True), pred_df.reset_index(drop=True)], axis=1)
            out_test_df.to_csv(artifact_dir / 'predictions' / 'best_weighted_test_predictions.csv', index=False)
            # compute per-task metrics on test
            for i, t in enumerate(targets):
                s = scores_test[:, i]
                y_true = truth_test[:, i]
                if len(y_true) and len(np.unique(y_true)) > 1:
                    try:
                        roc = float(roc_auc_score(y_true, s))
                    except Exception:
                        roc = None
                    try:
                        pr = float(average_precision_score(y_true, s))
                    except Exception:
                        pr = None
                else:
                    roc = None; pr = None
                test_metrics[t] = {'roc_auc': roc, 'pr_auc': pr}
        except Exception:
            test_metrics = {}

    md_lines.append('')
    md_lines.append('## Test metrics (best checkpoint)')
    for t, m in test_metrics.items():
        md_lines.append(f"- {t}: roc_auc={m.get('roc_auc')}, pr_auc={m.get('pr_auc')}")

    md_lines.append('')
    md_lines.append('## Notes')
    md_lines.append('- Per-task loss weights used: ' + json.dumps(loss_weights))
    md_lines.append('- Early stopping patience: ' + str(patience))

    summary_path = root / 'reports' / 'analysis' / 'weighted_training_run.md'
    summary_path.write_text('\n'.join(md_lines), encoding='utf-8')
    print('Weighted training run complete. Best epoch:', best_epoch)


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=str, default='kuairand-like-prediction')
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--pos-weight-is-like', action='store_true')
    args = p.parse_args()
    run(root=args.root, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, pos_weight_is_like=args.pos_weight_is_like)
