import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

from sklearn.cluster import KMeans
import tkinter as tk
from tkinter import filedialog
import random as rng
import cv2
import imutils
from imutils import contours
from skimage.io import imread
import numpy as np
import matplotlib.pyplot as plt

from utils import *

def main():
    # --- Start of Changes ---

    # 1. Create a root window for the dialog box and hide it
    root = tk.Tk()
    root.withdraw()

    # 2. Open a file dialog to ask the user to select an image
    # The 'filetypes' argument filters for common image files
    image_path = filedialog.askopenfilename(
        title="Select a photo of a foot",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )

    # 3. Check if the user selected a file. If not, exit gracefully.
    if not image_path:
        print("No file was selected. Exiting the program.")
        return

    # 4. Read the image from the path the user selected
    oimg = imread(image_path)

    # --- End of Changes ---


    if not os.path.exists('output'):
        os.makedirs('output')

   