#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import glob
from PIL import Image
from pathlib import Path
import xml.etree.ElementTree as ET

# Update the paths to use raw strings or double backslashes
dog_images_dir = glob.glob(r"C:\Users\krish\MINING ASSIGNMENT\dog\*\*")
annotations_dir = glob.glob(r"C:\Users\krish\MINING ASSIGNMENT\Annotations\*\*")
cropped_images_dir = r"C:\Users\ruthv\DataMining\cropped"

def get_bounding_boxes(annot):
    xml = annot
    tree = ET.parse(xml)
    root = tree.getroot()
    objects = root.findall('object')
    bbox = []
    for o in objects:
        bndbox = o.find('bndbox' )
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        bbox.append((xmin, ymin, xmax, ymax))
    return bbox

def get_image(annot):
    img_path = r"C:\Users\krish\MINING ASSIGNMENT\dog"
    file = annot.split('\\')
    img_filename = os.path.join(img_path, file[-2], file[-1] + '.jpg')
    return img_filename

# Check if both dog_images_dir and annotations_dir are not empty and have the same length
if dog_images_dir and annotations_dir and len(dog_images_dir) == len(annotations_dir):
    print(len(dog_images_dir))
    for dog_dir, annot_dir in zip(dog_images_dir, annotations_dir):
        bbox = get_bounding_boxes(annot_dir)
        dog = get_image(annot_dir)
        try:
            im = Image.open(dog)
        except Exception as e:
            print(f"Error opening image file: {dog}")
            print(f"Error details: {e}")
            continue

        print(im)
        for j, box in enumerate(bbox):
            im2 = im.crop(box)
            im2 = im2.resize((128, 128), Image.LANCZOS)
            new_path = dog.replace('dog', 'cropped')
            new_path = new_path.replace('.jpg', '-' + str(j) + '.jpg')
            head, tail = os.path.split(new_path)
            Path(head).mkdir(parents=True, exist_ok=True)
            im2.save(new_path)


# In[6]:


import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import color

# Function to get the list of images for each class
def get_images_per_class(images_dir):
    images_per_class = {}
    for image_path in images_dir:
        class_name = os.path.basename(os.path.dirname(image_path))
        if class_name not in images_per_class:
            images_per_class[class_name] = []
        images_per_class[class_name].append(image_path)
    return images_per_class

# Function to choose two images from each class
def choose_two_images_per_class(images_per_class):
    selected_images = {}
    for class_name, images in images_per_class.items():
        selected_images[class_name] = images[:2]
    return selected_images

# Function to convert color images to grayscale
def convert_to_grayscale(images):
    grayscale_images = []
    for image_path in images:
        image = Image.open(image_path)
        grayscale_image = color.rgb2gray(np.array(image))
        grayscale_images.append(grayscale_image)
    return grayscale_images

# Path to the directory containing images
dog_images_dir = glob.glob(r"C:\Users\krish\MINING ASSIGNMENT\dog\*\*")

# Get list of images per class
images_per_class = get_images_per_class(dog_images_dir)

# Choose two images from each class
selected_images = choose_two_images_per_class(images_per_class)

# Convert color images to grayscale
grayscale_images = {}
for class_name, images in selected_images.items():
    grayscale_images[class_name] = convert_to_grayscale(images)

# Display grayscale images
for class_name, images in grayscale_images.items():
    plt.figure(figsize=(8, 4))
    for i, image in enumerate(images, start=1):
        plt.subplot(1, 2, i)
        plt.imshow(image, cmap='gray')
        plt.title(f'{class_name} - Image {i}')
        plt.axis('off')
    plt.show()


# In[5]:


import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import color

# Function to get the list of images for each class
def get_images_per_class(images_dir):
    images_per_class = {}
    for image_path in images_dir:
        class_name = os.path.basename(os.path.dirname(image_path))
        if class_name not in images_per_class:
            images_per_class[class_name] = []
        images_per_class[class_name].append(image_path)
    return images_per_class

# Function to choose two images from each class
def choose_two_images_per_class(images_per_class):
    selected_images = {}
    for class_name, images in images_per_class.items():
        selected_images[class_name] = images[:2]
    return selected_images

# Function to convert color images to grayscale
def convert_to_grayscale(images):
    grayscale_images = []
    for image_path in images:
        image = Image.open(image_path)
        grayscale_image = color.rgb2gray(np.array(image))
        grayscale_images.append(grayscale_image)
    return grayscale_images

# Function to plot images and their histograms
def plot_images_and_histograms(grayscale_images):
    for class_name, images in grayscale_images.items():
        plt.figure(figsize=(10, 5))
        for i, image in enumerate(images, start=1):
            plt.subplot(2, 4, i)
            plt.imshow(image, cmap='gray')
            plt.title(f'{class_name} - Image {i}')
            plt.axis('off')

            plt.subplot(2, 4, i+4)
            plt.hist(image.ravel(), bins=256, color='black')
            plt.title('Histogram')
            plt.xlabel('Pixel Intensity')
            plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

# Path to the directory containing images
dog_images_dir = glob.glob(r"C:\Users\krish\MINING ASSIGNMENT\dog\*\*")

# Get list of images per class
images_per_class = get_images_per_class(dog_images_dir)

# Choose two images from each class
selected_images = choose_two_images_per_class(images_per_class)

# Convert color images to grayscale
grayscale_images = {}
for class_name, images in selected_images.items():
    grayscale_images[class_name] = convert_to_grayscale(images)

# Plot grayscale images and histograms
plot_images_and_histograms(grayscale_images)


# In[ ]:


pip install numpy


# In[ ]:


pip install opencv-python


# In[4]:


import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import color, filters

# Function to get the list of images for each class
def get_images_per_class(images_dir):
    images_per_class = {}
    for image_path in images_dir:
        class_name = os.path.basename(os.path.dirname(image_path))
        if class_name not in images_per_class:
            images_per_class[class_name] = []
        images_per_class[class_name].append(image_path)
    return images_per_class

# Function to choose two images from each class
def choose_two_images_per_class(images_per_class):
    selected_images = {}
    for class_name, images in images_per_class.items():
        selected_images[class_name] = images[:2]
    return selected_images

# Function to convert color images to grayscale
def convert_to_grayscale(images):
    grayscale_images = []
    for image_path in images:
        image = Image.open(image_path)
        grayscale_image = color.rgb2gray(np.array(image))
        grayscale_images.append(grayscale_image)
    return grayscale_images

# Function to perform edge detection using Sobel filter
def detect_edges(images):
    edge_images = []
    for image in images:
        edge_image = filters.sobel(image)
        edge_images.append(edge_image)
    return edge_images

# Plot grayscale images and their corresponding edge-detected images
def plot_images_and_edges(grayscale_images, edge_images):
    plt.figure(figsize=(12, 6))
    for i, (gray_image, edge_image) in enumerate(zip(grayscale_images, edge_images), start=1):
        plt.subplot(2, 4, i)
        plt.imshow(gray_image, cmap='gray')
        plt.title(f'Grayscale Image {i}')
        plt.axis('off')

        plt.subplot(2, 4, i+4)
        plt.imshow(edge_image, cmap='gray')
        plt.title(f'Edge-Detected Image {i}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Path to the directory containing images
dog_images_dir = glob.glob(r"C:\Users\krish\MINING ASSIGNMENT\dog\*\*")

# Get list of images per class
images_per_class = get_images_per_class(dog_images_dir)

# Choose two images from each class
selected_images = choose_two_images_per_class(images_per_class)

# Convert color images to grayscale
grayscale_images = {}
for class_name, images in selected_images.items():
    grayscale_images[class_name] = convert_to_grayscale(images)

# Perform edge detection using Sobel filter
edge_images = {}
for class_name, images in grayscale_images.items():
    edge_images[class_name] = detect_edges(images)

# Plot grayscale images and their corresponding edge-detected images
plot_images_and_edges(list(grayscale_images.values())[0], list(edge_images.values())[0])


# In[3]:


import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage import color, filters
import glob

# Function to get the list of images for each class
def get_images_per_class(images_dir):
    images_per_class = {}
    for image_path in images_dir:
        class_name = os.path.basename(os.path.dirname(image_path))
        if class_name not in images_per_class:
            images_per_class[class_name] = []
        images_per_class[class_name].append(image_path)
    return images_per_class

# Function to choose two images from each class
def choose_two_images_per_class(images_per_class):
    selected_images = {}
    for class_name, images in images_per_class.items():
        selected_images[class_name] = images[:2]
    return selected_images

# Function to convert color images to grayscale
def convert_to_grayscale(images):
    grayscale_images = []
    for image_path in images:
        image = Image.open(image_path)
        grayscale_image = color.rgb2gray(np.array(image))
        grayscale_images.append(grayscale_image)
    return grayscale_images

# Function to perform edge detection using Sobel filter
def detect_edges(images):
    edge_images = []
    for image in images:
        edge_image = filters.sobel(image)
        edge_images.append(edge_image)
    return edge_images

# Plot grayscale images and their corresponding edge-detected images
def plot_images_and_edges(grayscale_images, edge_images):
    plt.figure(figsize=(12, 6))
    for i, (gray_image, edge_image) in enumerate(zip(grayscale_images, edge_images), start=1):
        plt.subplot(2, 4, i)
        plt.imshow(gray_image, cmap='gray')
        plt.title(f'Grayscale Image {i}')
        plt.axis('off')

        plt.subplot(2, 4, i+4)
        plt.imshow(edge_image, cmap='gray')
        plt.title(f'Edge-Detected Image {i}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Path to the directory containing images
dog_images_dir = glob.glob(r"C:\Users\krish\MINING ASSIGNMENT\dog\*\*")

# Get list of images per class
images_per_class = get_images_per_class(dog_images_dir)

# Choose two images from each class
selected_images = choose_two_images_per_class(images_per_class)

# Convert color images to grayscale and perform edge detection
grayscale_images = {}
edge_images = {}
for class_name, images in selected_images.items():
    grayscale_images[class_name] = convert_to_grayscale(images)
    edge_images[class_name] = detect_edges(grayscale_images[class_name])

# Plot grayscale images and their corresponding edge-detected images
for class_name in grayscale_images.keys():
    plot_images_and_edges(grayscale_images[class_name], edge_images[class_name])


# In[2]:


import cv2
import numpy as np
from skimage import filters
import matplotlib.pyplot as plt

def angle(dx, dy):
    """Calculate the angles between horizontal and vertical operators."""
    return np.mod(np.arctan2(dy, dx), np.pi)

# Function to compute edge histograms
def compute_edge_histogram(image):
    sobel_h = filters.sobel_h(image)
    sobel_v = filters.sobel_v(image)
    angles = angle(sobel_h, sobel_v)
    hist, bin_edges = np.histogram(angles, bins=8, range=(0, np.pi))
    return hist

# Load one image from each class and convert them to grayscale
image_paths = [
    r"C:\Users\krish\MINING ASSIGNMENT\dog\n02090379-redbone\n02090379_1006.jpg",
    r"C:\Users\krish\MINING ASSIGNMENT\dog\n02097047-miniature_schnauzer\n02097047_124.jpg",
    r"C:\Users\krish\MINING ASSIGNMENT\dog\n02104365-schipperke\n02104365_1104.jpg",
    r"C:\Users\krish\MINING ASSIGNMENT\dog\n02112018-Pomeranian\n02112018_627.jpg"    # Add paths to other images here
]

grayscale_images = [cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) for image_path in image_paths]

# Compute edge histograms for each grayscale image
edge_histograms = [compute_edge_histogram(image) for image in grayscale_images]

# Plot edge histograms
plt.figure(figsize=(10, 6))
for i, hist in enumerate(edge_histograms):
    plt.plot(hist, label=f'Image {i+1}')
plt.xlabel('Angle Bins')
plt.ylabel('Frequency')
plt.title('Edge Histograms')
plt.legend()
plt.show()


# In[8]:


import cv2
import numpy as np
from skimage import filters, exposure
import matplotlib.pyplot as plt

def calculate_angles(dx, dy):
    """Calculate the angles between horizontal and vertical operators."""
    return np.mod(np.arctan2(dy, dx), np.pi)

def compute_edge_histogram(image):
    sobel_h = filters.sobel_h(image)
    sobel_v = filters.sobel_v(image)
    angles = calculate_angles(sobel_h, sobel_v)
    hist, bin_centers = exposure.histogram(image.ravel(), nbins=36, source_range='image')
    return hist, bin_centers

# Load grayscale images from different classes
image_paths = [
    r"C:\Users\krish\MINING ASSIGNMENT\dog\n02090379-redbone\n02090379_1006.jpg",
    r"C:\Users\krish\MINING ASSIGNMENT\dog\n02097047-miniature_schnauzer\n02097047_124.jpg",
    r"C:\Users\krish\MINING ASSIGNMENT\dog\n02104365-schipperke\n02104365_1104.jpg",
    r"C:\Users\krish\MINING ASSIGNMENT\dog\n02112018-Pomeranian\n02112018_627.jpg"   
    # Add paths to other images here
]
grayscale_images = [cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) for image_path in image_paths]

# Calculate edge histograms for each grayscale image
edge_histograms = [compute_edge_histogram(image) for image in grayscale_images]

# Visualize images with their corresponding edge histograms
fig, axes = plt.subplots(len(grayscale_images), 2, figsize=(10, 6 * len(grayscale_images)))

for i, (current_image, (current_hist, current_bin_centers)) in enumerate(zip(grayscale_images, edge_histograms)):
    # Display current image
    axes[i, 0].imshow(current_image, cmap='gray')
    axes[i, 0].set_title('Image')
    
    # Display current edge histogram
    axes[i, 1].bar(current_bin_centers, current_hist, width=np.pi / 18)
    axes[i, 1].set_xlabel('Bins')
    axes[i, 1].set_ylabel('Pixel Count')
    axes[i, 1].set_title('Edge Histogram')

plt.tight_layout()
plt.show()


# In[10]:


import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Define file paths for two images of the same dog breed
image1_path = r"C:\Users\krish\MINING ASSIGNMENT\dog\n02112018-Pomeranian\n02112018_1090.jpg"
image2_path = r"C:\Users\krish\MINING ASSIGNMENT\dog\n02097047-miniature_schnauzer\n02097047_535.jpg"

# Read grayscale images
hist1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
hist2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

# Resize both histograms to a specified shape
desired_shape = (500, 323)  # Adjust as needed
hist1 = cv2.resize(hist1, desired_shape)
hist2 = cv2.resize(hist2, desired_shape)

# Flatten the histograms for comparison
hist1_flat = hist1.flatten().reshape(1, -1)
hist2_flat = hist2.flatten().reshape(1, -1)

# Calculate Euclidean Distance
euclidean_distance = np.linalg.norm(hist1_flat - hist2_flat)

# Calculate Manhattan Distance
manhattan_distance = np.sum(np.abs(hist1_flat - hist2_flat))

# Calculate Cosine Similarity
cosine_similarity_score = cosine_similarity(hist1_flat, hist2_flat)[0][0]
cosine_distance = 1 - cosine_similarity_score

# Display the calculated distances
print(f"Euclidean Distance: {euclidean_distance}")
print(f"Manhattan Distance: {manhattan_distance}")
print(f"Cosine Distance: {cosine_distance}")


# In[11]:


import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Define file paths for two images of the same dog breed
image1_path = r"C:\Users\krish\MINING ASSIGNMENT\dog\n02090379-redbone\n02090379_364.jpg"
image2_path = r"C:\Users\krish\MINING ASSIGNMENT\dog\n02104365-schipperke\n02104365_1104.jpg"

# Read grayscale images
hist1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
hist2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

# Resize both histograms to a specified shape
desired_shape = (500, 323)  # Adjust as needed
hist1 = cv2.resize(hist1, desired_shape)
hist2 = cv2.resize(hist2, desired_shape)

# Flatten the histograms for comparison
hist1_flat = hist1.flatten().reshape(1, -1)
hist2_flat = hist2.flatten().reshape(1, -1)

# Calculate Euclidean Distance
euclidean_distance = np.linalg.norm(hist1_flat - hist2_flat)

# Calculate Manhattan Distance
manhattan_distance = np.sum(np.abs(hist1_flat - hist2_flat))

# Calculate Cosine Similarity
cosine_similarity_score = cosine_similarity(hist1_flat, hist2_flat)[0][0]
cosine_distance = 1 - cosine_similarity_score

# Display the calculated distances
print(f"Euclidean Distance: {euclidean_distance}")
print(f"Manhattan Distance: {manhattan_distance}")
print(f"Cosine Distance: {cosine_distance}")


# In[13]:


import cv2
import os
import numpy as np
from skimage import filters, exposure
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Function to convert images to edge histograms
def convert_to_edge_histograms(image_paths):
    edge_histograms = []
    for img_path in image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # Calculate vertical and horizontal sobel edge histograms
        angles = filters.sobel_v(img), filters.sobel_h(img)
        hist_v, _ = exposure.histogram(angles[0], nbins=36)
        hist_h, _ = exposure.histogram(angles[1], nbins=36)
        edge_histograms.append(np.concatenate((hist_v, hist_h)))
    return edge_histograms

# Define folders containing images for different classes
class_folders = {
    'class1': r"C:\Users\krish\MINING ASSIGNMENT\dog\n02090379-redbone",
    'class2': r"C:\Users\krish\MINING ASSIGNMENT\dog\n02104365-schipperke"
}

# Collect images from specified classes
selected_images = []
for class_name, folder in class_folders.items():
    image_files = os.listdir(folder)
    class_images = [os.path.join(folder, img) for img in image_files]
    selected_images.extend(class_images)

# Convert images to edge histograms
edge_histograms = convert_to_edge_histograms(selected_images)

# Perform PCA for dimensionality reduction to 2 components
pca = PCA(n_components=2)
pca_result = pca.fit_transform(edge_histograms)

# Plot the 2D points using different colors for each class
num_class1_images = len(os.listdir(class_folders['class1']))
plt.scatter(pca_result[:num_class1_images, 0], pca_result[:num_class1_images, 1], c='red', label='Class 1')
plt.scatter(pca_result[num_class1_images:, 0], pca_result[num_class1_images:, 1], c='blue', label='Class 2')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Dimensionality Reduction')
plt.legend()

# Set custom x and y axis limits
plt.xlim(pca_result[:, 0].min() - 1, pca_result[:, 0].max() + 1)
plt.ylim(pca_result[:, 1].min() - 1, pca_result[:, 1].max() + 1)

plt.show()


# In[ ]:




