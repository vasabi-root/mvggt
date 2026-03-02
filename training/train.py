import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

from mvggt.models.mvggt_training import MVGGT
from mvggt.models.loss import MVGGTLoss
from transformers import RobertaTokenizer

from dataset import ScanReferMvggtDataset
from scanrefer.lib.config import CONF

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.nn import DataParallel

from tqdm import tqdm

torch.backends.cuda.preferred_linalg_library("magma")

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_iou = 0.0
        self.best_loss = float('inf')

    def early_stop(self, current_loss, current_iou):
        if current_iou > self.best_iou + self.min_delta or current_loss < self.best_loss - self.min_delta:
            self.best_iou = max(self.best_iou, current_iou)
            self.best_loss = min(self.best_loss, current_loss)
            self.counter = 0
            return False
        self.counter += 1
        if self.counter >= self.patience:
            return True
        return False


def get_model(pretrained_path=None):
    model = MVGGT(
        use_referring_segmentation=True,
        freeze_visual_modules=True,
        freeze_encoder=True,
        num_multimodal_layers=12,
        multimodal_layer_selection='back',
        fusion_mode='pwa_only',
        use_lang_vision_fusion=False,
        use_controlnet_injection=True,
        use_lora=False,
        text_model_name="roberta-base",
        load_vggt=False if pretrained_path else True,
        use_pretrained_weights=False
    )
    
    if pretrained_path:
        state = torch.load(pretrained_path, map_location='cpu')  # Load to CPU
        model.load_state_dict(state, strict=False)               # Ignore mismatch keys

    return model

def get_dataloader(scanrefer_json, tokenizer, batch_size):
    dataset = ScanReferMvggtDataset(
        scanrefer_path=scanrefer_json,
        scannet_root=CONF.PATH.SCANNET,
        tokenizer=tokenizer,
        num_views=8,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=ScanReferMvggtDataset.collate_fn)
    return dataloader


def get_criterion():
    criterion = MVGGTLoss(
        use_referring_segmentation=True,
        referring_loss_weight_dict={'loss_mask': 1.0, 'loss_dice': 1.0},
        referring_layer_weight=0.5,
        dice_loss_mode='perview',
        perview_suppress_no_target=True,
        add_global_loss_weight=0.0,
        only_target=False
    )
    return criterion


def move_to_device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(v, device) for v in data]
    else:
        return data


@torch.no_grad()
def validate(model, criterion, dataloader, device, current_epoch, total_epochs):
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    num_batches = 0

    for sample in tqdm(dataloader, desc="Validation"):
        gt_raw = [move_to_device(v, device) for v in sample['gt_raw']]
        imgs = move_to_device(sample['imgs'], device)
        input_ids = move_to_device(sample['input_ids'], device)
        attention_mask = move_to_device(sample['attention_masks'], device)

        output = model(imgs, input_ids, attention_mask)
        loss, details = criterion(output, gt_raw, current_epoch=current_epoch, total_epochs=total_epochs)

        total_loss += loss.item()
        total_iou += details['refer_iou_score_global_per_sample'].mean().item()
        num_batches += 1

        torch.cuda.empty_cache()

    avg_loss = total_loss / num_batches
    avg_iou = total_iou / num_batches
    return avg_loss, avg_iou


def train(model, criterion, train_dataloader, val_dataloader, total_epochs=30):
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs * len(train_dataloader))

    writer = SummaryWriter("logs")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = DataParallel(model).to(device)

    early_stopper = EarlyStopper(patience=5)
    
    torch.cuda.empty_cache()

    for epoch in range(total_epochs):
        model.train()
        print(f'\nEpoch {epoch+1}/{total_epochs}')
        for batch_idx, sample in enumerate(t := tqdm(train_dataloader)):
            gt_raw = [move_to_device(v, device) for v in sample['gt_raw']]
            imgs = move_to_device(sample['imgs'], device)
            input_ids = move_to_device(sample['input_ids'], device)
            attention_mask = move_to_device(sample['attention_masks'], device)

            output = model(imgs, input_ids, attention_mask)
            loss, details = criterion(output, gt_raw, current_epoch=epoch, total_epochs=total_epochs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if batch_idx % 10 == 0:
                print(f"Batch {batch_idx} | Loss {loss.item():.4f} | IoU {details['refer_iou_score_global_per_sample'].mean():.4f}")
                writer.add_scalar("train/loss", loss.item(), epoch * len(train_dataloader) + batch_idx)
                writer.add_scalar("train/iou", details['refer_iou_score_global_per_sample'].mean(), epoch * len(train_dataloader) + batch_idx)

            torch.cuda.empty_cache()

        torch.save(model.module.state_dict(), f"ckpts/mvggt_scanrefer_latest.pth")

        val_loss, val_iou = validate(model, criterion, val_dataloader, device, epoch, total_epochs)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/iou", val_iou, epoch)

        if early_stopper.early_stop(val_loss, val_iou):
            print(f"Early stopping at epoch {epoch+1}")
            torch.save(model.module.state_dict(), f"ckpts/mvggt_scanrefer_best.pth")
            break

        if val_iou > early_stopper.best_iou:
            torch.save(model.module.state_dict(), f"ckpts/mvggt_scanrefer_best.pth")


def main():
    model = get_model('ckpts/mvggt_scanrefer_latest_2_epoch.pth')
    # model = get_model()
    tokenizer = RobertaTokenizer.from_pretrained('FacebookAI/roberta-base')
    criterion = get_criterion()
    torch.cuda.empty_cache()
    train_dataloader = get_dataloader(CONF.TRAIN_JSON, tokenizer, 12)
    val_dataloader = get_dataloader(CONF.VAL_JSON, tokenizer, 12)
    train(
        model=model,
        criterion=criterion,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        total_epochs=30,
    )


if __name__ == '__main__':
    main()