"""Evaluate a trained model on test data.

This script is written to be defensive:
- reports missing dependencies with install hints
- supports loading a Keras model (tf.keras.models.load_model) or a SavedModel
  with a `serving_default` signature
- accepts command-line overrides for the CSV and model paths
"""

import sys
import os
import argparse

try:
	import pandas as pd
	import numpy as np
except Exception as e:
	print("Missing required Python packages: pandas and numpy")
	print("Install them with: pip install pandas numpy")
	raise

try:
	import tensorflow as tf
	from sklearn.metrics import mean_squared_error
except Exception as e:
	print("Missing required packages: tensorflow and scikit-learn")
	print("Install them with: pip install tensorflow scikit-learn")
	raise

# Default paths (kept for backward compatibility)
TEST_CSV = r"C:\Users\soumy\OneDrive\Desktop\AI-PrognosAI\milestone_1_data_preparation\processed_data\test_processed_FD001.csv"
RUL_CSV = r"C:\Users\soumy\OneDrive\Desktop\AI-PrognosAI\milestone_1_data_preparation\processed_data\rul_targets_FD001.csv"
MODEL_PATH = r"C:\Users\soumy\OneDrive\Desktop\AI-PrognosAI\milestone_2_model_training\saved_models\lstm_rul_model_FD001.h5"


def load_model_any(path: str):
	"""Try to load a model either via tf.keras.models.load_model or tf.saved_model.load.

	Returns either a Keras Model (with .predict) or a callable (ConcreteFunction/signature).
	"""
	if not os.path.exists(path):
		raise FileNotFoundError(f"Model path does not exist: {path}")

	# First try the common keras loader with compile=False to avoid custom loss issues
	try:
		model = tf.keras.models.load_model(path, compile=False)
		print("Model loaded successfully (without compilation)")
		return model
	except Exception as e:
		print(f"Failed to load with keras loader: {e}")
		# Fall back to saved_model
		try:
			loaded = tf.saved_model.load(path)
			# prefer a serving_default signature if present
			if hasattr(loaded, 'signatures') and 'serving_default' in loaded.signatures:
				print("Loaded as SavedModel with serving_default signature")
				return loaded.signatures['serving_default']
			print("Loaded as SavedModel")
			return loaded
		except Exception as e2:
			print(f"Failed to load as SavedModel: {e2}")
			raise


def predict_with_model(model, X: np.ndarray) -> np.ndarray:
	"""Make predictions for a numpy array X using the provided model object.

	Handles Keras models and SavedModel signatures (callable objects that return tensors or dicts).
	"""
	# Keras-style
	if hasattr(model, 'predict'):
		preds = model.predict(X)
	else:
		# Assume callable signature (ConcreteFunction) that accepts tensors
		tf_X = tf.convert_to_tensor(X, dtype=tf.float32)
		out = model(tf_X)
		# signature outputs can be a dict of tensors
		if isinstance(out, dict):
			out = list(out.values())[0]
		# convert to numpy
		preds = out.numpy()

	return np.asarray(preds).squeeze()


def reshape_data_for_lstm(X, sequence_length=30):
	"""Reshape flat test data into sequences for LSTM input.
	
	Args:
		X: numpy array of shape (n_samples, n_features)
		sequence_length: number of time steps in each sequence
	
	Returns:
		X_sequences: numpy array of shape (n_sequences, sequence_length, n_features)
	"""
	n_samples, n_features = X.shape
	
	# Group by unit_id if it exists (first column is typically unit_id)
	# Assuming data is already sorted by unit and cycle
	unit_ids = X[:, 0].astype(int) if n_features > 24 else None
	
	sequences = []
	current_unit = None
	unit_data = []
	
	for i in range(n_samples):
		if unit_ids is not None:
			unit = unit_ids[i]
			if current_unit is None:
				current_unit = unit
			
			if unit != current_unit:
				# Process accumulated data for previous unit
				if len(unit_data) >= sequence_length:
					# Take only the features (exclude unit_id and cycle if present)
					features = np.array(unit_data)[:, 2:] if n_features > 24 else np.array(unit_data)
					# Take the last sequence_length rows
					sequences.append(features[-sequence_length:])
				unit_data = []
				current_unit = unit
			
			unit_data.append(X[i])
		else:
			# No unit ID, just reshape into chunks
			if i + sequence_length <= n_samples:
				sequences.append(X[i:i+sequence_length])
	
	# Don't forget the last unit
	if unit_ids is not None and len(unit_data) >= sequence_length:
		features = np.array(unit_data)[:, 2:] if n_features > 24 else np.array(unit_data)
		sequences.append(features[-sequence_length:])
	
	return np.array(sequences)


def main():
	parser = argparse.ArgumentParser(description="Evaluate a saved model on test CSV data")
	parser.add_argument('--test-csv', default=TEST_CSV, help='Path to test features CSV')
	parser.add_argument('--rul-csv', default=RUL_CSV, help='Path to RUL targets CSV')
	parser.add_argument('--model-path', default=MODEL_PATH, help='Path to saved model directory')
	parser.add_argument('--sequence-length', type=int, default=30, help='Sequence length for LSTM')

	args = parser.parse_args()

	print(f"Loading test data from: {args.test_csv}")
	print(f"Loading RUL targets from: {args.rul_csv}")
	print(f"Loading model from: {args.model_path}")
	print()

	# Load data
	X_test = pd.read_csv(args.test_csv)
	y_test = pd.read_csv(args.rul_csv).values.flatten()

	print(f"Test data shape (flat): {X_test.shape}")
	print(f"Target data shape: {y_test.shape}")
	print()

	# Convert X to numpy (models expect numeric arrays)
	X_flat = X_test.values
	
	# Reshape into sequences
	print(f"Reshaping data into sequences of length {args.sequence_length}...")
	X_np = reshape_data_for_lstm(X_flat, sequence_length=args.sequence_length)
	print(f"Reshaped data shape: {X_np.shape}")
	print()

	# Load model
	model = load_model_any(args.model_path)
	print()

	# Predict
	print("Making predictions...")
	predictions = predict_with_model(model, X_np)
	print(f"Predictions shape: {predictions.shape}")
	print()

	# Basic shape checks
	print(f"Number of predictions: {predictions.shape[0]}")
	print(f"Number of targets: {y_test.shape[0]}")
	
	if predictions.shape[0] != y_test.shape[0]:
		print(f"WARNING: Shape mismatch - using only the first {min(predictions.shape[0], y_test.shape[0])} samples")
		min_len = min(predictions.shape[0], y_test.shape[0])
		predictions = predictions[:min_len]
		y_test = y_test[:min_len]

	# Calculate metrics
	mse = mean_squared_error(y_test, predictions)
	rmse = np.sqrt(mse)
	mae = np.mean(np.abs(y_test - predictions))
	
	print("=" * 50)
	print("EVALUATION RESULTS")
	print("=" * 50)
	print(f"Mean Squared Error (MSE):  {mse:.4f}")
	print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
	print(f"Mean Absolute Error (MAE): {mae:.4f}")
	print("=" * 50)


if __name__ == '__main__':
	main()