import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from segment_anything import build_sam_vit_l
import logging
from datetime import datetime
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

def setup_logging():
    # Create logs directory if it doesn't exist
    if not os.path.exists('logs'):
        os.makedirs('logs')
    
    # Create a unique log file name with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/test_{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def calculate_f1_score(pred_masks, true_masks, threshold=0.5):
    # Convert to numpy arrays
    pred_masks = pred_masks.detach().cpu().numpy()
    true_masks = true_masks.detach().cpu().numpy()
    
    # Apply threshold
    pred_masks = (pred_masks > threshold).astype(np.uint8)
    true_masks = (true_masks > threshold).astype(np.uint8)
    
    # Flatten the arrays
    pred_masks = pred_masks.reshape(-1)
    true_masks = true_masks.reshape(-1)
    
    # Calculate F1 score
    return f1_score(true_masks, pred_masks)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        # Apply sigmoid to prediction if it hasn't been applied yet
        if pred.min() < 0 or pred.max() > 1:
            pred = torch.sigmoid(pred)
            
        # Flatten tensors
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        # Calculate intersection and union
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum()
        
        # Calculate Dice coefficient
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        
        # Return Dice loss
        return 1.0 - dice

class DiceBCELoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, smooth=1e-6):
        super(DiceBCELoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=smooth)
        
    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        
        # For Dice loss, we need to apply sigmoid to the prediction
        pred_sigmoid = torch.sigmoid(pred)
        dice_loss = self.dice(pred_sigmoid, target)
        
        # Combine losses with weights
        loss = self.bce_weight * bce_loss + self.dice_weight * dice_loss
        return loss

class SegmentationDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_dir = os.path.join(root_dir,  'images')
        self.mask_dir = os.path.join(root_dir,  'masks')
        
        # Get all image files with supported extensions
        self.image_files = []
        supported_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        
        # First get all image files
        for filename in os.listdir(self.image_dir):
            if filename.lower().endswith(supported_extensions):
                # Remove extension to get base name
                base_name = os.path.splitext(filename)[0]
                # Check if corresponding mask exists with any supported extension
                for ext in supported_extensions:
                    mask_path = os.path.join(self.mask_dir, base_name + ext)
                    if os.path.exists(mask_path):
                        self.image_files.append(filename)
                        break
        
        if not self.image_files:
            raise ValueError(f"No valid image-mask pairs found in {self.image_dir}")
        
        # logging.info(f"Found {len(self.image_files)} image-mask pairs in {split} set")
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_name)
        
        # Get base name without extension
        base_name = os.path.splitext(img_name)[0]
        
        # Find corresponding mask file with any supported extension
        mask_path = None
        supported_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
        for ext in supported_extensions:
            potential_mask_path = os.path.join(self.mask_dir, base_name + ext)
            if os.path.exists(potential_mask_path):
                mask_path = potential_mask_path
                break
        
        if mask_path is None:
            raise ValueError(f"No corresponding mask found for image {img_name}")
        
        # Load original image and mask
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # Get original size
        original_size = image.size
        
        # Create transforms for model input (1024x1024)
        model_transform = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
        ])
        
        # Create transforms for original size mask
        mask_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        # Transform images
        model_input = model_transform(image)
        original_mask = mask_transform(mask)
        
        return model_input, original_mask, original_size, img_name

def test(args):
    # Setup logging
    setup_logging()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Log test configuration
    logging.info("Test Configuration:")
    for arg, value in vars(args).items():
        logging.info(f"{arg}: {value}")
    
    # Load model with adapters
    sam = build_sam_vit_l(
        checkpoint=None, 
        use_adapter=True, 
        adapter_dim_ratio=args.adapter_dim_ratio
    )
    sam.to(device)
    
    # Log adapter information
    external_adapter_count = sum(1 for name, _ in sam.image_encoder.named_parameters() if 'external_adapters' in name)
    internal_adapter_count = sum(1 for name, _ in sam.image_encoder.named_parameters() if 'blocks' in name and 'adapter' in name)
    logging.info(f"Model has {internal_adapter_count} internal block adapters and {external_adapter_count} external adapters")
    
    # Load trained weights
    checkpoint = torch.load(args.trained_weights, map_location=device)
    sam.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"Loaded trained weights from {args.trained_weights}")
    logging.info(f"Trained F1 score: {checkpoint['f1_score']:.4f}")
    
    # Create dataset and dataloader
    val_dataset = SegmentationDataset(args.data_root)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    
    logging.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Use DiceBCELoss for evaluation
    criterion = DiceBCELoss(
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight
    )
    logging.info(f"Using DiceBCELoss with BCE weight: {args.bce_weight}, Dice weight: {args.dice_weight}")
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    sam.eval()
    f1_scores = []
    loss_values = []
    
    # Create progress bar for testing
    test_pbar = tqdm(val_loader, total=len(val_loader), desc="Testing")
        # zero_prompt
    sparse_embeddings = torch.zeros(
    (1, 0, 256),
    dtype=torch.float,
    device="cuda"
    )
    dense_embeddings = torch.zeros(
        (1, 256, 64, 64),
        dtype=torch.float,
        device="cuda"
    )
    with torch.no_grad():
        for model_input, original_mask, original_size, img_name in test_pbar:
            model_input = model_input.to(device)
            original_mask = original_mask.to(device)
            
            # Forward pass with model input size (1024x1024)
            encoder_output = sam.image_encoder(model_input, return_features=True)
            if len(encoder_output) == 3:
                image_embeddings, encoder_features, adapter_features = encoder_output
            else:
                image_embeddings, encoder_features = encoder_output
                adapter_features = None
            
            # sparse_embeddings, dense_embeddings = sam.prompt_encoder(
            #     points=None,
            #     boxes=None,
            #     masks=None
            # )
            low_res_masks, iou_predictions = sam.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=sam.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
                adapter_features=adapter_features,
            )
            
            # Resize prediction to original size
            low_res_masks = F.interpolate(low_res_masks, size=original_size[::-1], mode='bilinear', align_corners=False)
            
            # Calculate both loss and F1 score
            loss = criterion(low_res_masks, original_mask)
            loss_values.append(loss.item())
            
            f1 = calculate_f1_score(low_res_masks, original_mask)
            f1_scores.append(f1)
            
            # Update progress bar with current F1 score and loss
            test_pbar.set_postfix({'F1': f'{f1:.4f}', 'Loss': f'{loss.item():.4f}'})
            
            # Save prediction visualization
            pred_mask = (low_res_masks[0, 0] > 0.5).cpu().numpy().astype(np.uint8) * 255
            
            # Create visualization with exact original size
            width, height = original_size
            width = width.item() if torch.is_tensor(width) else width
            height = height.item() if torch.is_tensor(height) else height
            
            plt.figure(figsize=(width/100, height/100), dpi=100)
            plt.imshow(pred_mask, cmap='gray')
            plt.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            
            plt.savefig(os.path.join(args.output_dir, f'{img_name[0]}_result.png'), 
                       bbox_inches='tight', 
                       pad_inches=0,
                       dpi=100)
            plt.close()
            
            logging.info(f"Image: {img_name[0]}, F1 Score: {f1:.4f}, Loss: {loss.item():.4f}")
    
    # Calculate and log average F1 score and loss
    avg_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    avg_loss = np.mean(loss_values)
    
    logging.info(f"Average F1 Score: {avg_f1:.4f}")
    logging.info(f"F1 Score Std: {std_f1:.4f}")
    logging.info(f"Average Loss: {avg_loss:.4f}")
    
    print(f"\nTest Results:")
    print(f"Average F1 Score: {avg_f1:.4f}")
    print(f"F1 Score Std: {std_f1:.4f}")
    print(f"Average Loss: {avg_loss:.4f}")
    
    # Save summary results
    with open(os.path.join(args.output_dir, 'results_summary.txt'), 'w') as f:
        f.write(f"Model: {args.trained_weights}\n")
        f.write(f"Dataset: {args.data_root}\n")
        f.write(f"Num Samples: {len(val_dataset)}\n")
        f.write(f"Average F1 Score: {avg_f1:.4f}\n")
        f.write(f"F1 Score Std: {std_f1:.4f}\n")
        f.write(f"Average Loss: {avg_loss:.4f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test trained SAM model')
    parser.add_argument('--data_root', type=str, default='/hy-tmp/sam/datasets/wiree', help='Path to dataset root directory')
    parser.add_argument('--trained_weights', type=str, default='best_model.pth', help='Path to trained weights')
    parser.add_argument('--output_dir', type=str, default='test_results', help='Directory to save test results')
    parser.add_argument('--adapter_dim_ratio', type=float, default=0.1, help='Ratio of adapter dimension to model dimension')
    parser.add_argument('--bce_weight', type=float, default=0.7, help='Weight for BCE loss in combined loss')
    parser.add_argument('--dice_weight', type=float, default=0.3, help='Weight for Dice loss in combined loss')
    
    args = parser.parse_args()
    test(args) 