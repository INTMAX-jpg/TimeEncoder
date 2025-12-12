import os
import torch
import logging
import sys
from torch.utils.data import DataLoader

# Add repo root to path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.real_data_loader import ProcessedWeatherDataset
from training.base_trainer import BaseTrainer
from training.models import TimeXLModel, PrototypeManager
from training.loss import TimeXLLoss

# Configure Logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger("EncoderTrainer")

def train_encoder():
    logger.info("=== Starting Prototype-based Encoder Training ===")
    
    # Configuration
    DATA_DIR = '/root/timexl_repo/data'
    CITY = 'San Francisco'
    BATCH_SIZE = 64
    EPOCHS = 10  # Reduced to 10 for quick demonstration, increase for convergence
    LR = 0.001
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Device: {DEVICE}")
    
    # 1. Load Data
    logger.info(f"Loading data for {CITY}...")
    try:
        train_dataset = ProcessedWeatherDataset(DATA_DIR, city=CITY, split='train')
        val_dataset = ProcessedWeatherDataset(DATA_DIR, city=CITY, split='val')
        test_dataset = ProcessedWeatherDataset(DATA_DIR, city=CITY, split='test')
    except FileNotFoundError as e:
        logger.error(f"Data not found: {e}")
        logger.error("Please run data/preprocess_data.py first.")
        return

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    num_classes = len(train_dataset.label_map)
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Train samples: {len(train_dataset)}")
    
    # 2. Initialize Model & Components
    k = 10 # Prototypes per class
    model = TimeXLModel(num_classes=num_classes, k=k)
    pm = PrototypeManager(num_classes=num_classes, k=k)
    loss_fn = TimeXLLoss()
    
    # Move to device
    model.to(DEVICE)
    pm.to(DEVICE)
    
    # Optimizer
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(pm.parameters()), 
        lr=LR
    )
    
    # Trainer
    trainer = BaseTrainer(model, pm, loss_fn, optimizer, device=DEVICE)
    
    # 3. Training Loop
    best_val_acc = 0.0
    
    logger.info("Starting Training Loop...")
    for epoch in range(EPOCHS):
        logger.info(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # Train
        train_loss, train_acc, _ = trainer.train_epoch(train_loader)
        logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        
        # Validate
        val_loss, val_acc = trainer.validate(val_loader)
        logger.info(f"Val Loss: {val_loss:.4f}   | Val Acc: {val_acc:.4f}")
        
        # Save Best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            logger.info(f"✅ New Best Accuracy! Saving model...")
            torch.save(model.state_dict(), 'best_encoder.pth')
            torch.save(pm.state_dict(), 'best_prototypes.pth')
            
    # 4. Final Test
    logger.info("\n=== Final Testing ===")
    # Load best
    if os.path.exists('best_encoder.pth'):
        model.load_state_dict(torch.load('best_encoder.pth'))
        pm.load_state_dict(torch.load('best_prototypes.pth'))
        
    test_loss, test_acc = trainer.validate(test_loader)
    logger.info(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")
    
if __name__ == "__main__":
    train_encoder()
