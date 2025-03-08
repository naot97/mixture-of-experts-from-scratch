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

### Training the Model
```bash
python main.py --action train
```

### Custom Configuration
Modify the `config/config.yaml` file to adjust model parameters, data paths, and hyperparameters.

## 📈 Results
Results will be logged in the `logs/` directory with visualizations of model performance over time.

## 🧪 Experiments
- Varying the number of experts to analyze performance trade-offs.
- Exploring different gating mechanisms for improved routing efficiency.

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

