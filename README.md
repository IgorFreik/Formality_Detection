# Formality Detection


# How to run

1. Install the required libraries using the following command:
```bash
# Set up a virutal environment if you want
# python3 -m venv venv
# source venv/bin/activate
pip install -r requirements.txt
```

2. Run the following command to prepare the data:
```bash
python prepare_data.py
```

3. Run the following command to evaluate the models:
```bash
python main.py
```

4. Get the ROC/AUC curves; confusion matrixes; samples of incorrectly classified texts using the following command:
```bash
python plot_results.py
```

