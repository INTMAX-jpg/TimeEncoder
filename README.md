# TimeXL: Explainable Weather Forecasting with Prototype-based Encoders & LLMs

TimeXL is an interpretable deep learning framework designed for time-series forecasting (specifically weather prediction) that combines the efficiency of **Prototype Learning** with the reasoning capabilities of **Large Language Models (LLMs)**.

The system uses a **Prototype-based Encoder** to learn typical weather patterns ("prototypes") from historical data, allowing it to provide "Case-Based Reasoning" explanations (e.g., "I predict Rain because the current situation looks like these 3 historical rainy days...").

## 🌟 Key Features

*   **Multimodal Input**: Processes both numerical time-series data (temperature, humidity, pressure) and textual descriptions (weather summaries).
*   **Interpretability**: Uses a prototype layer to learn representative historical patterns. Every prediction is explained by citing the most similar historical examples from the training set.
*   **Distribution Prediction**: Outputs a probability distribution over weather categories (e.g., No Precipitation, Rain, Snow) rather than a single class.
*   **LLM Integration**: Can work alongside LLMs (like DeepSeek, GPT-4) to refine predictions or generate natural language reports based on retrieved prototypes.

---

## 📂 Project Structure

```
timexl_repo/
├── data/                       # Data storage and processing
│   ├── historical-hourly-weather-data/  # Raw CSV datasets
│   ├── processed/              # Processed PyTorch tensors (.pt)
│   ├── preprocess_data.py      # Script to clean and split data
│   └── real_data_loader.py     # PyTorch Dataset implementation
├── training/                   # Core training logic
│   ├── models.py               # TimeXLModel, PrototypeManager architecture
│   ├── loss.py                 # Custom Loss (KL Divergence + Prototype Losses)
│   ├── base_trainer.py         # Training loop and prototype projection
│   └── llm_agents.py           # LLM API wrappers (PredictionLLM, ReflectionLLM)
├── train_encoder.py            # Main script to train the Encoder
├── evaluate_encoder.py         # Script to evaluate model performance (Acc, KL, MAE)
├── interactive_predict.py      # Interactive CLI for testing predictions
├── compare_models.py           # Compare Encoder vs. Real LLM API
├── check_leakage.py            # Utility to verify train/test split integrity
└── README.md                   # Project documentation
```

---

## 🚀 Getting Started

### 1. Prerequisites

Ensure you have Python 3.8+ and PyTorch installed.

```bash
pip install torch pandas numpy scikit-learn
```

*(Optional)* If you plan to use the LLM comparison features, you will need an API key (e.g., DeepSeek, OpenAI).

### 2. Data Preparation

The project uses the "Historical Hourly Weather Data" dataset. First, process the raw CSVs into training/validation/testing tensors:

```bash
python data/preprocess_data.py
```
*   This will create `.pt` files in `data/processed/`.
*   It handles time-series splitting (Train: 80%, Val: 10%, Test: 10%) to prevent data leakage.

### 3. Training the Encoder

Train the prototype-based encoder. This model learns to map weather sequences to prototypes and predict future weather distributions.

```bash
python train_encoder.py
```
*   **Output**: Saves `best_encoder.pth` (Model weights) and `best_prototypes.pth` (Prototype vectors) to the root directory.
*   **Configuration**: You can adjust `BATCH_SIZE`, `EPOCHS`, and `CITY` inside the script.

### 4. Evaluation

Evaluate the trained model on the test set to get quantitative metrics:
*   **Accuracy**: Top-1 classification accuracy.
*   **KL Divergence**: Distance between predicted and true probability distributions.
*   **MAE (Mean Absolute Error)**: Average absolute difference in probabilities.

```bash
python evaluate_encoder.py
```

### 5. Interactive Prediction (Demo)

Want to try the model yourself? Run the interactive script to input weather descriptions and see predictions + explanations.

```bash
python interactive_predict.py
```

**Example Input:**
> mist, light rain, sky is clear

**Example Output:**
> **Prediction**: No Precipitation (76.8%), Rain (22.8%)
> **Why?**: Matches Pattern #1 (Similarity: 14.44) - Historical Example: "overcast clouds, scattered clouds..."

---

## 🧠 Model Architecture Details

The **TimeXL Encoder** consists of:
1.  **Time Encoder**: LSTM/GRU processing numerical features (Temp, Humidity, etc.).
2.  **Text Encoder**: Learnable embeddings for weather description keywords.
3.  **Prototype Layer**:
    *   Maintains $K$ learnable prototype vectors per class.
    *   Calculates similarity between input embeddings and prototypes.
4.  **Fusion Layer**: Combines similarity scores to output a probability distribution.

**Loss Function**:
$$ Loss = L_{KL} + \lambda_1 L_{Clst} + \lambda_2 L_{Evi} $$
*   $L_{KL}$: Minimizes divergence from true weather distribution.
*   $L_{Clst}$ (Clustering): Pulls inputs closer to their nearest class prototype.
*   $L_{Evi}$ (Evidence): Ensures every prototype has at least one close input sample.

---

## 🛠 Advanced: LLM Comparison

To compare the specialized Encoder against a general-purpose LLM:

1.  Export your API Key:
    ```bash
    export DEEPSEEK_API_KEY='your_api_key_here'
    ```
2.  Run the comparison script:
    ```bash
    python compare_models.py
    ```
    This will select random test samples, ask the LLM for a prediction, and compare the accuracy/KL-divergence of both approaches.

---

## ✅ Verification

To ensure data integrity (no future data leaking into training), run:
```bash
python check_leakage.py
```
