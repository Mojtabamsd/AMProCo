import gzip
import numpy as np
import struct
from collections import Counter
import os

# Paths to the original dataset
train_images_path = '../../dataset/fashion/train-images-idx3-ubyte.gz'
train_labels_path = '../../dataset/fashion/train-labels-idx1-ubyte.gz'

# Paths to save the long-tailed dataset
train_images_lt_path = '../../dataset/fashion/train-images-idx3-ubyte-LT.gz'
train_labels_lt_path = '../../dataset/fashion/train-labels-idx1-ubyte-LT.gz'

# Paths to save the validation dataset
val_images_path = '../../dataset/fashion/val-images-idx3-ubyte.gz'
val_labels_path = '../../dataset/fashion/val-labels-idx1-ubyte.gz'


# Helper function to read IDX format from gzip files
def read_idx(file_path):
    with gzip.open(file_path, 'rb') as f:
        magic, num_items = struct.unpack('>II', f.read(8))
        if magic == 2051:  # Images file
            rows, cols = struct.unpack('>II', f.read(8))
            data = np.frombuffer(f.read(), dtype=np.uint8).reshape(num_items, rows, cols)
        elif magic == 2049:  # Labels file
            data = np.frombuffer(f.read(), dtype=np.uint8)
        else:
            raise ValueError("Invalid IDX file format")
    return data


# Helper function to save IDX format to gzip files
def save_idx(data, file_path, is_images):
    with gzip.open(file_path, 'wb') as f:
        if is_images:
            magic = 2051
            num_items, rows, cols = data.shape
            header = struct.pack('>IIII', magic, num_items, rows, cols)
        else:
            magic = 2049
            num_items = data.shape[0]
            header = struct.pack('>II', magic, num_items)
        f.write(header)
        f.write(data.tobytes())


# Read original data
train_images = read_idx(train_images_path)
train_labels = read_idx(train_labels_path)

# Create a long-tailed distribution
label_counts = Counter(train_labels)
num_classes = len(label_counts)

# Parameters for long-tail
max_count = 5900  # Max samples in the largest class
alpha = 1.5  # Decay factor

lt_class_counts = [int(max_count / (i + 1) ** alpha) for i in range(num_classes)]  # Long-tailed distribution

print(f"Original class counts: {label_counts}")
print(f"Long-tailed target counts: {lt_class_counts}")

# Generate the LT dataset and track unused indices for validation set
lt_images, lt_labels = [], []
unused_indices = []

for label, target_count in enumerate(lt_class_counts):
    indices = np.where(train_labels == label)[0]
    np.random.shuffle(indices)
    selected_indices = indices[:target_count]
    remaining_indices = indices[target_count:]

    lt_images.append(train_images[selected_indices])
    lt_labels.append(train_labels[selected_indices])
    unused_indices.append(remaining_indices)

# Concatenate the LT dataset
lt_images = np.concatenate(lt_images)
lt_labels = np.concatenate(lt_labels)

# Generate the validation dataset
val_images, val_labels = [], []

for label, remaining in enumerate(unused_indices):
    np.random.shuffle(remaining)
    val_indices = remaining[:50]  # Select 50 samples per class
    val_images.append(train_images[val_indices])
    val_labels.append(train_labels[val_indices])

# Concatenate the validation dataset
val_images = np.concatenate(val_images)
val_labels = np.concatenate(val_labels)

# Save the LT dataset
os.makedirs(os.path.dirname(train_images_lt_path), exist_ok=True)
save_idx(lt_images, train_images_lt_path, is_images=True)
save_idx(lt_labels, train_labels_lt_path, is_images=False)

# Save the validation dataset
os.makedirs(os.path.dirname(val_images_path), exist_ok=True)
save_idx(val_images, val_images_path, is_images=True)
save_idx(val_labels, val_labels_path, is_images=False)

print(f"Long-tailed dataset saved to {train_images_lt_path} and {train_labels_lt_path}")
print(f"Validation dataset saved to {val_images_path} and {val_labels_path}")
