import torch
import numpy as np
import os
import sys
import pandas as pd
from training.models import TimeXLModel, PrototypeManager
from training.base_trainer import BaseTrainer
from data.real_data_loader import ProcessedWeatherDataset
from torch.utils.data import DataLoader
from training.loss import TimeXLLoss
import torch.optim as optim

# Add root to path
sys.path.append('/root/timexl_repo')

# Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = '/root/timexl_repo/data'
CITY = 'San Francisco'
MODEL_PATH = '/root/best_encoder.pth'
PROTOTYPES_PATH = '/root/best_prototypes.pth'

def load_model_and_prototypes(num_classes=3, k=10):
    """Loads the trained model and prototypes."""
    print("Loading model and prototypes...")
    
    # Initialize model components
    model = TimeXLModel(num_classes=num_classes, k=k).to(DEVICE)
    pm = PrototypeManager(num_classes=num_classes, k=k).to(DEVICE)
    
    # Load weights
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PROTOTYPES_PATH):
        print("Error: Trained model files not found! Please train the encoder first.")
        sys.exit(1)
        
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    pm.load_state_dict(torch.load(PROTOTYPES_PATH, map_location=DEVICE))
    
    model.eval()
    pm.eval()
    
    return model, pm

def get_explanation_library(model, pm, num_classes=3, k=10):
    """
    Generates or loads the explanation library (mapping prototypes to raw training data).
    Ideally, this should be saved/loaded, but here we regenerate it on the fly for demonstration.
    """
    print("Generating explanation library from training data (this might take a moment)...")
    
    # Use a subset of training data for speed in this demo
    train_dataset = ProcessedWeatherDataset(DATA_DIR, city=CITY, split='train')
    # Using a smaller batch for faster startup, or full dataset for better accuracy
    # For a real app, this should be pre-computed.
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=False) 
    
    # We need a dummy trainer instance to use the projection method
    # Loss and optimizer are not needed for projection but required for init
    loss_fn = TimeXLLoss()
    optimizer = optim.Adam(model.parameters()) 
    trainer = BaseTrainer(model, pm, loss_fn, optimizer, device=DEVICE)
    
    time_expl, text_expl, label_expl = trainer.project_prototypes_to_raw_data(train_loader)
    return time_expl, text_expl, label_expl

def predict_custom_input(model, pm, time_expl, text_expl, input_text, input_numerical=None):
    """
    Predicts weather distribution for a custom input.
    If input_numerical is None, we use a dummy or zero vector (or ask user).
    For this interactive demo, we will focus on TEXT input as that's what users easily provide.
    We will use a dummy numerical input if real one isn't available, 
    or randomly sample one from test set to simulate 'current conditions'.
    """
    
    # 1. Prepare Input
    # For text, we wrap it in a list
    text_batch = [input_text]
    
    # For numerical, if not provided, we just use zeros or a placeholder
    # In a real app, this would come from sensors.
    if input_numerical is None:
        # Create a dummy numerical input: [1, 24, 5]
        numerical_tensor = torch.zeros(1, 24, 5).to(DEVICE)
    else:
        numerical_tensor = input_numerical.to(DEVICE)

    # 2. Encode
    with torch.no_grad():
        z_time = model.time_encoder(numerical_tensor)
        z_text = model.text_encoder(text_batch)
        
        # 3. Prototype Similarity
        sim_time_flat, sim_text_flat = pm(z_time, z_text)
        
        # 4. Fusion & Prediction
        logits = model.fusion(sim_time_flat, sim_text_flat)
        probs = torch.softmax(logits, dim=1)
        
    return probs.cpu().numpy()[0], sim_text_flat, z_text

def format_prediction(probs):
    categories = ["No Precipitation", "Rain", "Snow"]
    result = {}
    for i, cat in enumerate(categories):
        result[cat] = float(probs[i])
    return result

def find_top_contributors(sim_flat, num_classes=3, k=10, top_n=3):
    """Finds the prototypes that contributed most to the decision."""
    # sim_flat is [1, C*K]
    # Reshape to [C, K]
    sim_reshaped = sim_flat.reshape(num_classes, k)
    
    # Find global top contributors across all classes
    # Or find top contributors for the PREDICTED class
    
    # Let's find global top N indices
    flat_indices = torch.topk(sim_flat.flatten(), top_n).indices
    values = torch.topk(sim_flat.flatten(), top_n).values
    
    contributors = []
    for idx, val in zip(flat_indices, values):
        idx = idx.item()
        c = idx // k
        j = idx % k
        contributors.append({"class": c, "prototype_idx": j, "score": val.item()})
        
    return contributors

def interactive_session():
    print("\n" + "="*60)
    print("      TimeXL Weather Predictor - Interactive Mode      ")
    print("="*60)
    
    # Initialize
    model, pm = load_model_and_prototypes()
    time_expl, text_expl, label_expl = get_explanation_library(model, pm)
    
    print("\nSystem ready! You can now enter weather descriptions.")
    print("Example: 'mist, light rain, sky is clear, ...'")
    print("Type 'exit' or 'quit' to stop.\n")
    
    while True:
        try:
            user_input = input("Enter recent weather description (past 24h) > ")
            if user_input.lower() in ['exit', 'quit']:
                print("Goodbye!")
                break
            
            if not user_input.strip():
                continue
                
            print("\nAnalyzing...")
            
            # Perform prediction
            # Note: We are using dummy numerical data here as user only provides text.
            # In a full system, we would ask for numerical data or fetch it.
            probs, sim_text_flat, _ = predict_custom_input(model, pm, time_expl, text_expl, user_input)
            
            # Display Results
            print("\n" + "-"*40)
            print("PREDICTION RESULT (Next 24 Hours)")
            print("-"*40)
            
            categories = ["No Precipitation", "Rain", "Snow"]
            pred_class_idx = np.argmax(probs)
            pred_class = categories[pred_class_idx]
            
            print(f"Most Likely: **{pred_class.upper()}**")
            print("Probability Distribution:")
            for cat, p in format_prediction(probs).items():
                bar_len = int(p * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                print(f"  {cat:<16}: {bar} {p:.1%}")
                
            # Explainability
            print("\n" + "-"*40)
            print("WHY? (Top Matching Historical Cases)")
            print("-"*40)
            
            contributors = find_top_contributors(sim_text_flat)
            
            for i, contrib in enumerate(contributors):
                c, j, score = contrib['class'], contrib['prototype_idx'], contrib['score']
                
                # Retrieve explanation
                # Note: We are only looking at TEXT prototypes here since user input was text
                raw_text_expl = text_expl.get((c, j), "No explanation found")
                # Truncate long text
                display_text = (raw_text_expl[:75] + '...') if len(raw_text_expl) > 75 else raw_text_expl
                
                print(f"Reason #{i+1} (Similarity Score: {score:.2f})")
                print(f"  - Matches Pattern Class: {categories[c]}")
                print(f"  - Historical Example: \"{display_text}\"")
                print("")
                
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    interactive_session()
