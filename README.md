# Classic vs Quantum ML Stock Prediction

This project compares classical machine learning techniques (LSTM) with quantum machine learning approaches for stock price prediction. It demonstrates how both paradigms can be applied to financial time series data.

## Project Structure

```
├── data/                      # Data storage directory
│   ├── raw/                   # Raw data downloaded from sources
│   └── processed/             # Processed data ready for modeling
├── models/                    # Saved model files
│   ├── classical/             # Classical ML models
│   └── quantum/               # Quantum ML models
├── notebooks/                 # Jupyter notebooks for exploration and research
│   └── Classic_vs_Quantum_ML_Stock_Prediction.ipynb
├── src/                       # Source code
│   ├── data/                  # Data collection and processing modules
│   │   ├── __init__.py
│   │   ├── data_collection.py # Functions to collect stock data
│   │   └── data_processing.py # Data preprocessing functions
│   ├── models/                # ML models implementation
│   │   ├── __init__.py
│   │   ├── classical.py       # Classical ML models (LSTM)
│   │   └── quantum.py         # Quantum ML models
│   ├── visualization/         # Data visualization tools
│   │   ├── __init__.py
│   │   └── plots.py           # Plotting functions
│   └── utils/                 # Utility functions
│       ├── __init__.py
│       └── metrics.py         # Performance metrics functions
├── tests/                     # Unit tests
│   ├── __init__.py
│   ├── test_data.py           # Tests for data functionality
│   └── test_models.py         # Tests for model functionality
├── .gitignore                 # Git ignore file
├── requirements.txt           # Project dependencies
├── setup.py                   # Package installation
└── README.md                  # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Simone-meli25/Classic_vs_Quantum_ML_Stock_Prediction.git
cd Classic_vs_Quantum_ML_Stock_Prediction
```

2. Create a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Collection
To download stock data for a specific ticker:
```python
from src.data.data_collection import fetch_stock_data

# Fetch 5 years of AAPL stock data
stock_data = fetch_stock_data(ticker="AAPL", period="5y")
```

### Training Models
```python
from src.models.classical import LSTMModel
from src.data.data_processing import prepare_data_for_lstm

# Prepare data
X_train, y_train, X_test, y_test, scaler = prepare_data_for_lstm(stock_data)

# Create and train LSTM model
model = LSTMModel()
model.train(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
```

### Visualization
```python
from src.visualization.plots import plot_predictions

# Plot actual vs predicted prices
plot_predictions(actual_prices, predictions, ticker="AAPL")
```

## License
[MIT License](LICENSE)

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.