# TimeEncoder: Explainable Weather Forecasting with Prototype-based Encoder
(Based on the idea from the NeurIPS 2025 Paper: *Explainable Multi-modal Time Series Prediction with LLM-in-the-Loop*, hereafter referred to as "the Paper".)

🌈The Paper proposed a framework named **TimeXL**, an interpretable deep learning framework designed for time-series forecasting. It combines the efficiency of **Prototype Learning** with the reasoning capabilities of **Large Language Models (LLMs)**.

💪**Motivation**: I found the core idea of this paper quite interesting. However, no official implementation was available online. Therefore, for my course project this semester, I decided to attempt a practical code reproduction of the TimeXL framework.

This repository focuses on reproducing the **Prototype-based Encoder** module. This encoder is designed to learn typical weather patterns ("prototypes") from historical data, enabling it to provide "Case-Based Reasoning" explanations (e.g., "I predict Rain because the current situation resembles these 3 historical rainy days...").

We trained and tested this encoder on a public dataset: **Historical Hourly Weather Data 2012-2017 (HHWD)** (available on [Kaggle](https://www.kaggle.com/datasets/selfishgene/historical-hourly-weather-data)).

⭐**Prediction Task**: Given 24 hours of weather descriptions as input, the model predicts the proportional distribution over three categories (Rain, Snow, & Other) for the next 24 hours.

🤩On the test set, the predicted distribution and the actual distribution achieved a KL-divergence of only **0.08**.

<img width="476" height="295" alt="Prediction Results" src="https://github.com/user-attachments/assets/9e4d8f17-2cf0-4b05-87bd-e127375d41ce" />

🚗Furthermore, we have preliminarily achieved the "explainable prototype" visualization effect. After training, you can run the instruction `python demo_interpretability.py` to see it in action.

⚠️Regarding the LLM components mentioned in the Paper, we have implemented the basic functional framework (requiring users to configure their own API key to call a real LLM). However, **the prediction performance was not satisfactory.**
*   For the **PredictionLLM**, the performance was not terrible but consistently lagged behind the results from the Prototype-based Encoder. (You can run `python compare_models.py` to see the performance comparison).
*   For the latter two LLMs in the framework, which are mainly used to refine initial text descriptions as per the Paper, their utility seems limited for the HHWD dataset. This is because the `weather_description` field in HHWD is already highly processed, leaving little room for meaningful refinement by an LLM.

Therefore, for the HHWD dataset, the practical value of employing these LLM components requires further reconsideration.


## 🌟 Key Features

*   **Multimodal Input**: Processes both numerical time-series data (temperature, humidity, pressure) and textual descriptions (weather summaries).
*   **Explainability**: Uses a prototype layer to learn representative historical patterns. Every prediction is explained by citing the most similar historical examples from the training set.
*   **Distribution Prediction**: Outputs a probability distribution over weather categories (e.g., No Precipitation, Rain, Snow) rather than a single class.
*   **LLM Integration**: Can work alongside LLMs (like DeepSeek, GPT-4) to refine predictions or generate natural language reports based on retrieved prototypes.

---

## 📂 Project Structure

```
timexl_repo/
├── data/                       # Data storage and processing
│   ├── historical-hourly-weather-data/  # Raw CSV datasets from https://www.kaggle.com/datasets/selfishgene/historical-hourly-weather-data
│   ├── processed/              # Processed PyTorch tensors (.pt), where processed data has been split into training, validation, and test sets in an 8:1:1 ratio, following chronological order.
│   ├── preprocess_data.py      # Script to clean and split data
│   └── real_data_loader.py     # PyTorch Dataset implementation
├── training/                   # Core training logic
│   ├── models.py               # TimeXLModel, PrototypeManager architecture
│   ├── loss.py                 # Custom Loss (KL Divergence + Prototype Losses)
│   ├── base_trainer.py         # Training loop and prototype projection
│   └── llm_agents.py           # LLM API wrappers (PredictionLLM, ReflectionLLM & RefinementLLM)
├── train_encoder.py            # Main script to train the Encoder
├── evaluate_encoder.py         # Script to evaluate model performance (Acc, KL, MAE)
├── interactive_predict.py      # Interactive CLI for user testing predictions
├── compare_models.py           # run this to compare Encoder vs. Real LLM API
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
