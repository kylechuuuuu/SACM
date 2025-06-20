import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from segment_anything import build_sam_vit_l
from segment_anything.modeling import ImageEncoderViT
import argparse
import torch.nn.functional as F
import logging
from datetime import datetime
from sklearn.metrics import f1_score
from tqdm import tqdm

def setup_logging(args):
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/training_{timestamp}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info("Training Configuration:")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def calculate_f1_score(pred_masks, true_masks, threshold=0.5):
    pred_masks = pred_masks.detach().cpu().numpy()
    true_masks = true_masks.detach().cpu().numpy()
    
    pred_masks = (pred_masks > threshold).astype(np.uint8)
    true_masks = (true_masks > threshold).astype(np.uint8)
    
    pred_masks = pred_masks.reshape(-1)
    true_masks = true_masks.reshape(-1)
    
    return f1_score(true_masks, pred_masks)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        # Apply sigmoid if not already applied
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
            
        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Calculate Dice
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        return 1.0 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(DiceBCELoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        pred_sigmoid = torch.sigmoid(pred)
        dice_loss = self.dice(pred_sigmoid, target)
        
        loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return loss

class SegmentationDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.image_dir = os.path.join(root_dir, split, 'images')
        self.mask_dir = os.path.join(root_dir, split, 'masks')
        
        self.image_files = []
        supported_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        
        for filename in os.listdir(self.image_dir):
            if filename.lower().endswith(supported_extensions):
                base_name = os.path.splitext(filename)[0]
                for ext in supported_extensions:
                    mask_path = os.path.join(self.mask_dir, base_name + ext)
                    if os.path.exists(mask_path):
                        self.image_files.append(filename)
                        break
        
        if not self.image_files:
            raise ValueError(f"No valid image-mask pairs found in {self.image_dir}")
        
        logging.info(f"Found {len(self.image_files)} image-mask pairs in {split} set")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        base_name = os.path.splitext(img_name)[0]
        
        mask_path = None
        supported_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        for ext in supported_extensions:
            potential_mask_path = os.path.join(self.mask_dir, base_name + ext)
            if os.path.exists(potential_mask_path):
                mask_path = potential_mask_path
                break
        
        if mask_path is None:
            raise ValueError(f"No corresponding mask found for image {img_name}")
        
        # Load images
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
            
        return image, mask

def train(args):
    setup_logging(args)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Build SAM model with adapters enabled
    sam = build_sam_vit_l(checkpoint=args.checkpoint, use_adapter=True, adapter_dim_ratio=args.adapter_dim_ratio)
    sam.to(device)
    
    total_params, trainable_params = count_parameters(sam)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    logging.info(f"Trainable parameters percentage: {trainable_params/total_params*100:.2f}%")
    
    # Freeze image encoder parameters except for adapters
    for name, param in sam.image_encoder.named_parameters():
        # Only train adapters and leave everything else frozen
        if 'adapter' not in name:
            param.requires_grad = False
    
    # Freeze prompt encoder parameters
    for param in sam.prompt_encoder.parameters():
        param.requires_grad = False
    
    # Enable mask decoder parameters - explicitly set to trainable
    for param in sam.mask_decoder.parameters():
        #######################################################################################
        param.requires_grad = True
        #######################################################################################
    
    # Count and log parameters after freezing
    total_params, trainable_params = count_parameters(sam)
    logging.info(f"After freezing - Total parameters: {total_params:,}")
    logging.info(f"After freezing - Trainable parameters: {trainable_params:,}")
    logging.info(f"After freezing - Trainable parameters percentage: {trainable_params/total_params*100:.2f}%")
    
    transform = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
    ])
    
    train_dataset = SegmentationDataset(args.data_root, 'train', transform)
    val_dataset = SegmentationDataset(args.data_root, 'val', transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    logging.info(f"Training dataset size: {len(train_dataset)}")
    logging.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Use Dice-BCE combined loss for better segmentation results
    criterion = DiceBCELoss(bce_weight=args.bce_weight, dice_weight=args.dice_weight)
    logging.info(f"Using DiceBCELoss with BCE weight: {args.bce_weight}, Dice weight: {args.dice_weight}")
    
    # Create parameter groups for optimizer - adapters and mask decoder
    adapter_params = []
    mask_decoder_params = []
    
    # Collect adapter parameters
    for name, param in sam.image_encoder.named_parameters():
        if 'adapter' in name and param.requires_grad:
            adapter_params.append(param)
    
    # Log the count of internal and external adapters
    internal_adapter_count = sum(1 for name, _ in sam.image_encoder.named_parameters() if 'blocks' in name and 'adapter' in name)
    external_adapter_count = sum(1 for name, _ in sam.image_encoder.named_parameters() if 'external_adapters' in name)
    logging.info(f"Found {internal_adapter_count} internal block adapters and {external_adapter_count} external adapters")
    
    # Collect mask decoder parameters
    for param in sam.mask_decoder.parameters():
        if param.requires_grad:
            mask_decoder_params.append(param)
    
    # Configure optimizer with different learning rates for different components
    optimizer = optim.AdamW([
        {'params': adapter_params, 'lr': args.adapter_lr, 'weight_decay': args.weight_decay},
        {'params': mask_decoder_params, 'lr': args.decoder_lr, 'weight_decay': args.weight_decay}
    ])
    
    # Set up learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=args.epochs,
            eta_min=args.min_lr
        )
    elif args.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.lr_step_size,
            gamma=args.lr_gamma
        )
    elif args.scheduler == 'reduce':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=args.lr_gamma,
            patience=args.patience,
            verbose=True
        )
    else:
        scheduler = None
    
    logging.info(f"Adapter parameters: {len(adapter_params)}, learning rate: {args.adapter_lr}")
    logging.info(f"Mask decoder parameters: {len(mask_decoder_params)}, learning rate: {args.decoder_lr}")
    logging.info(f"Scheduler: {args.scheduler}")
    
    
    best_f1_score = 0.0
    
    for epoch in range(args.epochs):
        sam.train()
        train_loss = 0.0
        
        # Create progress bar for training
        train_pbar = tqdm(train_loader, total=len(train_loader), desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        
        for images, masks in train_pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with image encoder returning features for skip connections
            encoder_output = sam.image_encoder(images, return_features=True)
            if len(encoder_output) == 3:
                image_embeddings, encoder_features, adapter_features = encoder_output
            else:
                image_embeddings, encoder_features = encoder_output
                adapter_features = None
            
            sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                points=None,
                boxes=None,
                masks=None
            )
            low_res_masks, iou_predictions = sam.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                adapter_features=adapter_features,
            )
            
            masks = F.interpolate(masks, size=(1024, 1024), mode='bilinear', align_corners=False)
            low_res_masks = F.interpolate(low_res_masks, size=(1024, 1024), mode='bilinear', align_corners=False)
            
            loss = criterion(low_res_masks, masks)
            
            loss.backward()
            
            # Add gradient clipping to improve stability
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in sam.parameters() if p.requires_grad],
                    args.clip_grad_norm
                )
                
            optimizer.step()
            
            train_loss += loss.item()
            
            # Update progress bar
            train_pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss/len(train_loader)
        logging.info(f'Epoch {epoch+1}/{args.epochs} - Average Train Loss: {avg_train_loss:.4f}')
        
        # Validate at specified intervals
        if (epoch + 1) % args.val_interval == 0:
            sam.eval()
            val_loss = 0.0
            all_pred_masks = []
            all_true_masks = []
            
            # Create progress bar for validation
            val_pbar = tqdm(val_loader, total=len(val_loader), desc=f"Epoch {epoch+1}/{args.epochs} [Val]")
            
            with torch.no_grad():
                for images, masks in val_pbar:
                    images = images.to(device)
                    masks = masks.to(device)
                    
                    encoder_output = sam.image_encoder(images, return_features=True)
                    if len(encoder_output) == 3:
                        image_embeddings, encoder_features, adapter_features = encoder_output
                    else:
                        image_embeddings, encoder_features = encoder_output
                        adapter_features = None
                    
                    sparse_embeddings, dense_embeddings = sam.prompt_encoder(
                        points=None,
                        boxes=None,
                        masks=None
                    )
                    low_res_masks, iou_predictions = sam.mask_decoder(
                        image_embeddings=image_embeddings,
                        image_pe=sam.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                        adapter_features=adapter_features,
                    )
                    
                    masks = F.interpolate(masks, size=(1024, 1024), mode='bilinear', align_corners=False)
                    low_res_masks = F.interpolate(low_res_masks, size=(1024, 1024), mode='bilinear', align_corners=False)
                    
                    loss = criterion(low_res_masks, masks)
                    val_loss += loss.item()
                    
                    # Update progress bar
                    val_pbar.set_postfix({'loss': loss.item()})
                    
                    # Collect predictions and ground truth for F1 calculation
                    all_pred_masks.append(low_res_masks)
                    all_true_masks.append(masks)
            
            # Calculate F1 score
            all_pred_masks = torch.cat(all_pred_masks, dim=0)
            all_true_masks = torch.cat(all_true_masks, dim=0)
            f1 = calculate_f1_score(all_pred_masks, all_true_masks)
            
            avg_val_loss = val_loss/len(val_loader)
            logging.info(f'Epoch {epoch+1}/{args.epochs} - Average Val Loss: {avg_val_loss:.4f}')
            logging.info(f'Epoch {epoch+1}/{args.epochs} - F1 Score: {f1:.4f}')
            
            # Update scheduler if using ReduceLROnPlateau
            if args.scheduler == 'reduce':
                scheduler.step(avg_val_loss)
            
            # Save best model based on F1 score
            if f1 > best_f1_score:
                best_f1_score = f1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': sam.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'f1_score': f1,
                }, 'best_model.pth')
                logging.info(f'👍New best model saved with F1 score: {best_f1_score:.4f}')
        
        # Step the scheduler (except for ReduceLROnPlateau which is updated during validation)
        if scheduler is not None and args.scheduler != 'reduce':
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            logging.info(f'Current learning rate: {current_lr:.7f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SAM with adapters')
    parser.add_argument('--data_root', type=str, default='/hy-tmp/sam/datasets/eee', help='Path to dataset root directory')
    parser.add_argument('--checkpoint', type=str, default='/hy-tmp/sam/sam_pth/sam_vit_l_0b3195.pth', help='Path to SAM checkpoint')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--adapter_dim_ratio', type=float, default=0.1, help='Ratio of adapter dimension to model dimension')
    parser.add_argument('--adapter_lr', type=float, default=1e-4, help='Learning rate for adapter modules')
    parser.add_argument('--decoder_lr', type=float, default=1e-4, help='Learning rate for mask decoder')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay for optimizer')
    parser.add_argument('--epochs', type=int, default=40, help='Number of epochs')
    parser.add_argument('--val_interval', type=int, default=1, help='Validation interval (epochs)')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0, help='Gradient clipping norm (0 to disable)')
    
    # Loss function weights
    parser.add_argument('--bce_weight', type=float, default=0.7, help='Weight for BCE loss in combined loss')
    parser.add_argument('--dice_weight', type=float, default=0.3, help='Weight for Dice loss in combined loss')
    
    # Scheduler parameters
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'step', 'reduce', 'none'], help='LR scheduler type')
    parser.add_argument('--min_lr', type=float, default=1e-6, help='Minimum learning rate for cosine scheduler')
    parser.add_argument('--lr_step_size', type=int, default=10, help='Step size for StepLR scheduler')
    parser.add_argument('--lr_gamma', type=float, default=0.1, help='Gamma for StepLR and ReduceLROnPlateau schedulers')
    parser.add_argument('--patience', type=int, default=3, help='Patience for ReduceLROnPlateau scheduler')
    
    args = parser.parse_args()
    train(args) 