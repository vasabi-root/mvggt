import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[1]))

from mvggt.models.mvggt_training import MVGGT
from mvggt.models.loss import MVGGTLoss
from transformers import RobertaModel, RobertaTokenizer

# from scanrefer.lib.dataset import ScannetReferenceDataset
# from scanrefer.lib.solver import Solver

from dataset import ScanReferMvggtDataset
from scanrefer.lib.config import CONF

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torchvision
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

def get_model():
    model = MVGGT(
        use_referring_segmentation=True,
        freeze_visual_modules=True,           # frozen Pi3/VGGT visual
        freeze_encoder=True,
        num_multimodal_layers=12,             # from the article
        multimodal_layer_selection='back',
        fusion_mode='pwa_only',               # can be 'interleaved', but in the article only "point-wise add" used
        use_lang_vision_fusion=False,
        use_controlnet_injection=True,
        use_lora=False,                       # True to fine-tune decoder
        text_model_name="roberta-base",
        load_vggt=True,                       # VGGT-1B encoder/decoder
        use_pretrained_weights=False           # loads Pi3 VGGT decoder (multimodal_decoder)
    )
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
        referring_layer_weight=0.5,          # intermediate layers
        dice_loss_mode='perview',
        perview_suppress_no_target=True,     # hybrid 50% no-target
        add_global_loss_weight=0.0,          # 
        only_target=False
    )
    return criterion

def train(model, criterion, dataloader, total_epochs=30):
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30*len(dataloader))

    writer = SummaryWriter("logs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(total_epochs):
        model.train()
        print(f'\nEpoch {epoch+1}/{total_epochs}')
        for batch_idx, sample in enumerate(tqdm(dataloader)):
            
            gt_raw = sample['gt_raw']
            for vl in gt_raw:
                for key in vl:
                    vl[key] = vl[key].to(device)
                    
            imgs = sample['imgs']
            input_ids = sample['input_ids']
            attention_mask = sample['attention_masks']

            output = model(imgs.to(device), input_ids.to(device), attention_mask.to(device))
            loss, details = criterion(output, gt_raw, current_epoch=epoch, total_epochs=total_epochs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} | Loss {loss.item():.4f} | IoU {details['refer_iou_score_global_per_sample'].mean():.4f}")
                writer.add_scalar("loss", loss.item(), epoch*len(dataloader)+batch_idx)
                
            torch.cuda.empty_cache()
        
        torch.save(model.state_dict(), f"ckpts/mvggt_scanrefer_epoch_{epoch}.pth")
    
def main():
    model = get_model()
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base', add_pooling_layer=False)
    criterion = get_criterion()
    train_dataloader = get_dataloader(CONF.TRAIN_JSON, tokenizer, 2)
    val_dataloader = get_dataloader(CONF.VAL_JSON, tokenizer, 2)
    
    train(
        model=model,
        criterion=criterion,
        dataloader=train_dataloader,
        total_epochs=30,
    )
    
if __name__ == '__main__':
    main()
    