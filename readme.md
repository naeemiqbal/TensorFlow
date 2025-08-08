# TensorFlow Experiments

This repository contains various experiments and scripts using TensorFlow and Keras for deep learning tasks, including:

- MNIST digit classification
- Fruit image classification (fresh/rotten, apples/bananas/oranges)
- Model training, evaluation, and prediction scripts
- Data preprocessing and visualization utilities

## Structure

- `numberL4.py`,`numberL201.py`,`numberL20b.py`, Scripts for training on MNIST number dataset, L specifies the layers
- `numberTest.py`,`numberTest2.py`: Scripts for testing digit classifiers on MNIST or custom images.
- `fruits.py`, `fruitsTest.py`: Scripts for training and testing fruit classifiers.
- `../models/`: Saved Keras models (`.keras` files).
- `../data/`: Datasets for training and testing (not included in repo, see below).

## Requirements

- Python 3.8+
- TensorFlow 2.x
- Keras
- numpy, matplotlib, pillow

Install dependencies with:
```bash
pip install tensorflow keras numpy matplotlib pillow
```

## Usage

1. Place your datasets in the appropriate folders (see script comments for expected paths).
2. Run training scripts to train models and save them.
3. Use test scripts to evaluate models on new images.

## Notes

- Large files and datasets are excluded from version control via `.gitignore`.
- Update dataset paths in scripts as needed for your environment.
- For best results, ensure your input images match the expected format (size, color channels, normalization).

## License

This project is for educational and research purposes.