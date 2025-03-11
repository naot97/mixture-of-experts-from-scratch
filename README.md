# Mixture of Experts (MoE) from Scratch

This repository implements a Mixture of Experts (MoE) model from scratch, showcasing the fundamentals of this powerful technique for enhancing model scalability and efficiency.

## 🚀 Features
- Custom-built MoE architecture with dynamic routing.
- Modular design for easy integration and experimentation.
- Comprehensive training pipeline with data loading, evaluation, and visualization tools.

## 📂 Project Structure
```
moe-from-scratch/
├── data/            # Sample datasets for testing
├── src/             # Custom MoE model implementations
├── main.py          # Training script for the MoE model
├── inference.py     # Inference script for model evaluation
├── requirements.txt # Dependencies
└── README.md        # Project documentation
```

## 🛠️ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/naot97/moe-from-scratch.git
   cd moe-from-scratch
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 📋 Usage
### Downloading data
To download the TinyStories dataset:
```bash
mkdir -p data && cd data \
  && wget -q --show-progress https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStories_all_data.tar.gz \
  && tar -xzf TinyStories_all_data.tar.gz && rm TinyStories_all_data.tar.gz
```

### Train the Model
```bash
python main.py --action train
```

### Inference
```bash
```


## 📈 Results
Results will be logged in the console, accompanied by visualizations of model performance over time. Example console output:
```bash
[INFO] Starting training with {config.max_iters} steps...
[INFO] Evaluating: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:00<00:00, 778.45it/s]
[INFO] step 0: train loss 5.5025, val loss 5.0905████████████████████████████████████████████████████▊                       | 154/200 [00:00<00:00, 790.92it/s]
[INFO] Saving model...
[INFO] Model saved.
[INFO] Evaluating: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:00<00:00, 728.56it/s]
[INFO] step 100: train loss 3.2076, val loss 2.8585█████████████████████████████████████████████████████▊                    | 160/200 [00:00<00:00, 801.71it/s]
[INFO] Saving model...
[INFO] Model saved.
[INFO] Evaluating: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:00<00:00, 457.24it/s]
[INFO] step 200: train loss 2.9717, val loss 2.6219██████████████████████████████████████████▏                               | 137/200 [00:00<00:00, 380.61it/s]
[INFO] Saving model...
[INFO] Model saved.
...
[INFO] Training complete.
```

## 🧪 Custom Configuration
Modify the `config/config.yaml` file to adjust model parameters, data paths, and hyperparameters.

## 🧩 Future Enhancements
- Integration with PyTorch Lightning for improved training scalability.
- Addition of Transformer-based MoE models.
- Enhanced visualization tools for interpretability.

## 🤝 Contributing
Contributions are welcome! Please open an issue or submit a pull request to discuss your ideas.

## 📄 License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## 📧 Contact
For questions or collaboration opportunities, feel free to reach out or open an issue.

