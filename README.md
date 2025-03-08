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
[INFO] Evaluating: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:00<00:00, 791.23it/s]
[INFO] step 0: train loss 5.5724, val loss 5.5368██████████████████████████████████████████████████████▊                     | 158/200 [00:00<00:00, 795.12it/s]
[INFO] Evaluating: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:00<00:00, 750.87it/s]
[INFO] step 100: train loss 4.3956, val loss 3.6213█████████████████████████████████████████████████▊                        | 152/200 [00:00<00:00, 765.14it/s]
[INFO] Evaluating: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:00<00:00, 791.72it/s]
[INFO] step 200: train loss 3.8865, val loss 3.1939█████████████████████████████████████████████████████▎                    | 159/200 [00:00<00:00, 803.97it/s]
[INFO] Evaluating: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:00<00:00, 796.77it/s]
[INFO] step 300: train loss 3.6333, val loss 2.9691██████████████████████████████████████████████████████▎                   | 161/200 [00:00<00:00, 810.59it/s]
[INFO] Evaluating: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:00<00:00, 788.14it/s]
[INFO] step 400: train loss 3.4595, val loss 2.8409█████████████████████████████████████████████████████▊                    | 160/200 [00:00<00:00, 786.20it/s]
[INFO] Evaluating: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:00<00:00, 716.31it/s]
[INFO] step 500: train loss 3.3324, val loss 2.8218██████████████████████████████████████████▋                               | 138/200 [00:00<00:00, 686.51it/s]
[INFO] Evaluating: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:00<00:00, 768.84it/s]
[INFO] step 600: train loss 3.2473, val loss 2.7575█████████████████████████████████████████████████████████▊                | 168/200 
...
[00:00<00:00, 770.86it/s][INFO] Training complete.
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

