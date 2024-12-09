import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageTk
import cv2 
import customtkinter as ctk
from tkinter import filedialog,messagebox
from collections import deque
import os
# Fonction pour ajouter un bruit gaussien
def gaussien(image, variance):
    mean = 0
    noisy_image = image.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            u1, u2 = np.random.rand(), np.random.rand()
            z = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
            noise = mean + np.sqrt(variance) * z
            noisy_image[i, j] += noise
            noisy_image[i, j] = max(0, min(1, noisy_image[i, j]))
    return noisy_image

# Fonction pour ajouter un bruit poivre et sel
def poivre_sel(image, pourcentage):
    noisy_image = image.copy()
    num_pixels = int(pourcentage * image.size)
    height, width = image.shape

    for _ in range(num_pixels // 2):
        i, j = np.random.randint(0, height), np.random.randint(0, width)
        noisy_image[i, j] = 0

    for _ in range(num_pixels // 2):
        i, j = np.random.randint(0, height), np.random.randint(0, width)
        noisy_image[i, j] = 1

    return noisy_image


# Filtre Prewitt
def prewitt(image):
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    grad_x = np.zeros_like(image)
    grad_y = np.zeros_like(image)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            grad_x[i, j] = np.sum(kernel_x * image[i - 1:i + 2, j - 1:j + 2])
            grad_y[i, j] = np.sum(kernel_y * image[i - 1:i + 2, j - 1:j + 2])
    grad_x = np.abs(grad_x) / np.max(np.abs(grad_x))
    grad_y = np.abs(grad_y) / np.max(np.abs(grad_y))
    return grad_x, grad_y


# Filtre Sobel
def sobel(image):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    grad_x = np.zeros_like(image)
    grad_y = np.zeros_like(image)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            grad_x[i, j] = np.sum(kernel_x * image[i - 1:i + 2, j - 1:j + 2])
            grad_y[i, j] = np.sum(kernel_y * image[i - 1:i + 2, j - 1:j + 2])
    grad_x = np.abs(grad_x) / np.max(np.abs(grad_x))
    grad_y = np.abs(grad_y) / np.max(np.abs(grad_y))
    return grad_x, grad_y


# Filtre Robert
def robert(image):
    kernel_x = np.array([[1, 0], [0, -1]])
    kernel_y = np.array([[0, 1], [-1, 0]])
    grad_x = np.zeros_like(image)
    grad_y = np.zeros_like(image)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            grad_x[i, j] = np.sum(kernel_x * image[i:i + 2, j:j + 2])
            grad_y[i, j] = np.sum(kernel_y * image[i:i + 2, j:j + 2])
    grad_x = np.abs(grad_x) / np.max(np.abs(grad_x))
    grad_y = np.abs(grad_y) / np.max(np.abs(grad_y))
    return grad_x, grad_y


# Calcul du gradient global
def Gradient(grad_x, grad_y):
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    gradient = gradient / np.max(gradient)
    return gradient


# Fonction de seuillage simple
def SeuilSim(image, seuil):
    if image.max() > 1:
        image = image / 255.0
    binary_image = np.zeros_like(image)
    binary_image[image > seuil] = 1
    return (binary_image * 255).astype(np.uint8)
    # return cv.threshold(image, seuil, 1, cv.THRESH_BINARY)[1]



def SeuilHys(image, seuil_bas, seuil_haut):
    # Initialize binary image and visited matrix
    binary_image = np.zeros_like(image, dtype=np.uint8)
    binary_image[image > seuil_haut] = 1
    visited = np.zeros_like(image, dtype=bool)
    
    # Initialize queue for BFS
    queue = deque()

    # Enqueue all strong edges (above the high threshold)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            if binary_image[i, j] == 1 and not visited[i, j]:
                queue.append((i, j))

    # BFS for weak edge propagation
    while queue:
        x, y = queue.popleft()
        if visited[x, y]:
            continue
        visited[x, y] = True
        # Check if the current pixel qualifies as a weak edge
        if seuil_bas < image[x, y] <= seuil_haut:
            binary_image[x, y] = 1
            # Propagate to neighbors
            for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1]:
                    queue.append((nx, ny))
    
    return binary_image

    # return cv.Canny((image * 255).astype(np.uint8), seuil_bas, seuil_haut)
    
def log_kernel(sigma):
    
    # Create the grid for x and y coordinates
    x, y = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2))
    
    # Compute the Laplacian of Gaussian
    norm = (x**2 + y**2) / (2 * sigma**2)
    gaussian = np.exp(norm) * 4 / np.sqrt(2 * np.pi * sigma**2)
    laplacian = (norm - 1) * gaussian
    
    # Normalize the kernel to have a sum of zero
    log_kernel = laplacian - laplacian.mean()
    
    return log_kernel
def convolve(image, kernel):
    rows = image.shape[0]
    cols = image.shape[1]
    pad = kernel.shape[0] // 2
    output = np.zeros_like(image)

    for i in range(pad, rows-pad):
        for j in range(pad, cols-pad):
            if kernel.shape[0] % 2 == 0:
                area = image[i:i+pad+1, j:j+pad+1]
            else:
                area = image[i-1:i+2, j-1:j+2]
            output[i, j] = np.sum(kernel * area)

    return output


def LOG(image: Image.Image, sigma):
    laplacian_kernel = log_kernel(float(sigma))
    image_array = np.array(image)
    grad_arr = convolve(image_array, laplacian_kernel)

    # Handle invalid values
    grad_arr = np.nan_to_num(grad_arr)  # Replace NaN with 0
    grad_arr = np.clip(grad_arr, 0, 255)  # Clip values to [0, 255]

    # Convert to uint8 and create an image
    grad = Image.fromarray(grad_arr.astype(np.uint8))

    return grad


# Main GUI
class ImageProcessingApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Window Configuration
        self.title("Image Processing Application")
        self.geometry("800x800")
        self.configure(bg_color="lightgrey")

        # Scrollable Frame
        self.scrollable_frame = ctk.CTkScrollableFrame(self, width=780, height=780)
        self.scrollable_frame.pack(pady=10, padx=10, fill="both", expand=True)

        # Attributes
        self.image = None
        self.processed_image = None
        self.noisy_image = None

        # Section: Title
        self.title_label = ctk.CTkLabel(self.scrollable_frame, text="Image Processing Application", font=("Arial", 20, "bold"))
        self.title_label.pack(pady=20)

        # Section: Upload Image
        upload_frame = ctk.CTkFrame(self.scrollable_frame, corner_radius=10, border_width=2)
        upload_frame.pack(pady=20, padx=20, fill="x")
        upload_label = ctk.CTkLabel(upload_frame, text="Upload Image", font=("Arial", 16))
        upload_label.pack(pady=5)
        self.upload_button = ctk.CTkButton(upload_frame, text="Choose File", command=self.upload_image)
        self.upload_button.pack(pady=10)

        # Section: Noise Options
        noise_frame = ctk.CTkFrame(self.scrollable_frame, corner_radius=10, border_width=2)
        noise_frame.pack(pady=20, padx=20, fill="x")
        noise_label = ctk.CTkLabel(noise_frame, text="Add Noise to Image", font=("Arial", 16))
        noise_label.pack(pady=5)

        self.noise_var = ctk.StringVar(value="None")
        self.noise_menu = ctk.CTkOptionMenu(noise_frame, values=["None", "Gaussian", "Salt and Pepper"], variable=self.noise_var)
        self.noise_menu.pack(pady=5)

        self.noise_param_entry = ctk.CTkEntry(noise_frame, placeholder_text="Parameter (Variance/Percentage)")
        self.noise_param_entry.pack(pady=5)
        
        self.noise_button = ctk.CTkButton(noise_frame, text="Apply Noise", command=self.process_noise_image)
        self.noise_button.pack(pady=10)

        # Section: Filters
        filter_frame = ctk.CTkFrame(self.scrollable_frame, corner_radius=10, border_width=2)
        filter_frame.pack(pady=20, padx=20, fill="x")
        filter_label = ctk.CTkLabel(filter_frame, text="Apply Filters", font=("Arial", 16))
        filter_label.pack(pady=5)

        self.filter_var = ctk.StringVar(value="None")
        self.filter_menu = ctk.CTkOptionMenu(filter_frame, values=["None", "Prewitt", "Sobel", "Robert"], variable=self.filter_var)
        self.filter_menu.pack(pady=5)

        self.filter_button_noise = ctk.CTkButton(filter_frame, text="Filter Noisy Image", command=self.filter_Image_noise)
        self.filter_button_noise.pack(pady=5)
        self.filter_button_original = ctk.CTkButton(filter_frame, text="Filter Original Image", command=self.filter_Image_original)
        self.filter_button_original.pack(pady=5)

        # Section: Thresholding
        threshold_frame = ctk.CTkFrame(self.scrollable_frame, corner_radius=10, border_width=2)
        threshold_frame.pack(pady=20, padx=20, fill="x")
        threshold_label = ctk.CTkLabel(threshold_frame, text="Thresholding Options", font=("Arial", 16))
        threshold_label.pack(pady=5)
        self.threshold_var = ctk.StringVar(value="None")
        self.threshold_menu = ctk.CTkOptionMenu(
        threshold_frame,
        values=["None", "Simple", "Hysteresis"],             
        variable=self.threshold_var, 
        command=self.update_seuillage_entries
        )

        self.threshold_menu.pack(pady=5)

        self.threshold_low_entry = ctk.CTkEntry(threshold_frame, placeholder_text="Low Threshold")
        self.threshold_low_entry.pack(pady=5)
        self.threshold_high_entry = ctk.CTkEntry(threshold_frame, placeholder_text="High Threshold")
        self.threshold_high_entry.pack(pady=5)

        self.threshold_button = ctk.CTkButton(threshold_frame, text="Apply on Noisy Image", command=lambda: self.seuillage_on_gradient(gradient_image_bruit))
        self.threshold_button.pack(pady=5)
        self.threshold_button_original = ctk.CTkButton(threshold_frame, text="Apply on Original Image", command=lambda: self.seuillage_on_gradient(gradient_image_original))
        self.threshold_button_original.pack(pady=5)

        # Section: Log Image
        
        log_frame = ctk.CTkFrame(self.scrollable_frame, corner_radius=10, border_width=2)
        log_frame.pack(pady=20, padx=20, fill="x")
        log_label = ctk.CTkLabel(log_frame, text="Apply Laplace of Gaussian Filter", font=("Arial", 16))
        log_label.pack(pady=20)
        self.laplace_sigma_entry = ctk.CTkEntry(log_frame, placeholder_text="Sigma")
        self.laplace_sigma_entry.pack(pady=2)
        
        self.process_button = ctk.CTkButton(log_frame, text="Log Image", font=("Arial", 14, "bold"), command=self.log_image)
        self.process_button.pack(pady=20)

    def update_seuillage_entries(self, choice):
        if choice == "Simple":
            self.threshold_high_entry.configure(state="disabled")
        elif choice == "Hysteresis":
            self.threshold_high_entry.configure(state="normal")
        else:
            self.threshold_low_entry.configure(state="normal")
            self.threshold_high_entry.configure(state="normal")


    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.bmp;*.jpeg")])
        if file_path:
            img = Image.open(file_path).convert('L')
            self.image = np.array(img) / 255.0
            global image_name
            image_name = os.path.basename(file_path)  # Extract the image name from the file path
            
    def process_noise_image(self):
        if self.image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return

        # Check noise parameters
        if  self.noise_var.get() == "None":
            messagebox.showerror("Error", "Please choose a noise type.")
            return
        
        name,ext = os.path.splitext(image_name) 
        global noise_choice
        noise_choice = self.noise_var.get()
        if noise_choice in ["Gaussian", "Salt and Pepper"]:
            if not self.noise_param_entry.get():
                messagebox.showerror("Error", "Please enter a noise parameter.")
                return
        if noise_choice == "Gaussian":
            variance = float(self.noise_param_entry.get())
            self.noisy_image = gaussien(self.image, variance)
            cv2.imwrite(f'{noise_choice} , {name} , {variance}{ext}',self.noisy_image*255)
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("Original Image")
            plt.imshow(self.image, cmap="gray")

            plt.subplot(1, 2, 2)
            plt.title(f"Noisy Image {noise_choice}")
            plt.imshow(self.noisy_image, cmap="gray")

            plt.show()
        elif noise_choice == "Salt and Pepper":
            pourcentage = float(self.noise_param_entry.get())
            self.noisy_image = poivre_sel(self.image, pourcentage)
               
            cv2.imwrite(f'{noise_choice} , {name},{pourcentage}{ext}',self.noisy_image*255)
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("Original Image")
            plt.imshow(self.image, cmap="gray")

            plt.subplot(1, 2, 2)
            plt.title(f"Noisy Image {noise_choice}")
            plt.imshow(self.noisy_image, cmap="gray")

            plt.show()
        
    def plot_filtered_images(self, grad_x, grad_y, gradient_image, original_image, filter_choice ):
        plt.figure(figsize=(15, 5))

        # Plot original image
        plt.subplot(1, 4, 4)
        plt.title("input Image")
        plt.imshow(original_image, cmap="gray")

        # Plot horizontal gradient
        plt.subplot(1, 4, 1)
        plt.title(f"Image horizontal de {filter_choice}")
        plt.imshow(grad_x, cmap="gray")

        # Plot vertical gradient
        plt.subplot(1, 4, 2)
        plt.title(f"Image vertical de {filter_choice}")
        plt.imshow(grad_y, cmap="gray")

        # Plot gradient magnitude
        plt.subplot(1, 4, 3)
        plt.title(f"Gradient de {filter_choice}")
        plt.imshow(gradient_image, cmap="gray")

        plt.show()

            
    def filter_Image_noise(self):
        # Ensure an image is provided
        if self.noisy_image is None:
            messagebox.showerror("Error", "No noise image available for filtering.")
            return
        # Apply filter
        filter_choice = self.filter_var.get()
        if filter_choice == "None":
            messagebox.showerror("Error", "Please choose a filter.")
            return
        if filter_choice == "Prewitt":
            grad_x, grad_y = prewitt(self.noisy_image)
        elif filter_choice == "Sobel":
            grad_x, grad_y = sobel(self.noisy_image)
        elif filter_choice == "Robert":
            grad_x, grad_y = robert(self.noisy_image)
        else:
            grad_x, grad_y = None, None

        if grad_x is not None and grad_y is not None:
            global gradient_image_bruit
            gradient_image_bruit = Gradient(grad_x, grad_y)
            self.plot_filtered_images(grad_x, grad_y, gradient_image_bruit, self.noisy_image, filter_choice)

            
    def filter_Image_original(self):
        # Ensure an image is provided
        if self.image is None:
            messagebox.showerror("Error", "No image available for filtering.")
            return
        # Apply filter
        filter_choice = self.filter_var.get()
        if filter_choice == "None":
            messagebox.showerror("Error", "Please choose a filter.")
            return
        if filter_choice == "Prewitt":
            grad_x, grad_y = prewitt(self.image)
        elif filter_choice == "Sobel":
            grad_x, grad_y = sobel(self.image)
        elif filter_choice == "Robert":
            grad_x, grad_y = robert(self.image)
        else:
            grad_x, grad_y = None, None

        if grad_x is not None and grad_y is not None:
            global gradient_image_original
            gradient_image_original = Gradient(grad_x, grad_y)
            self.plot_filtered_images(grad_x, grad_y, gradient_image_original, self.image, filter_choice)
            
    def seuillage_on_gradient(self,image_gradient):
        # Apply thresholding
        if image_gradient is None:
            messagebox.showerror("Error", "No gradient image available for thresholding.")
            return
        threshold_choice = self.threshold_var.get()
        if threshold_choice == "Simple":
            low_threshold = self.threshold_low_entry.get()
            if not low_threshold:
                messagebox.showerror("Error", "Please enter the low threshold for Simple thresholding.")
                return
            seuil = float(self.threshold_low_entry.get())
            seuillage_gradient = SeuilSim(image_gradient, seuil)
            plt.figure(figsize=(12, 8))
            
            plt.subplot(1,3, 1)
            plt.title("input Image")
            plt.imshow(self.image, cmap="gray")
           
            
            plt.subplot(1, 3, 2)
            plt.title("Gradient")
            plt.imshow(image_gradient, cmap="gray")
                       
            
            plt.subplot(1, 3, 3)
            plt.title(f"Image after Seuillage {threshold_choice}")
            plt.imshow(seuillage_gradient, cmap="gray")
            

            plt.show()
        elif threshold_choice == "Hysteresis":
            seuil_bas = float(self.threshold_low_entry.get())
            seuil_haut = float(self.threshold_high_entry.get())
            if not seuil_bas or not seuil_haut:
                messagebox.showerror("Error", "Please enter both low and high thresholds for Hysteresis thresholding.")
                return
            seuillage_gradient = SeuilHys(image_gradient, seuil_bas, seuil_haut)
            
            plt.figure(figsize=(12, 8))            
            plt.subplot(1,3, 1)
            plt.title("input Image")
            plt.imshow(self.image, cmap="gray")
            
            plt.subplot(1, 3, 2)
            plt.title("Gradient")
            plt.imshow(image_gradient, cmap="gray")
           
            plt.subplot(1, 3, 3)
            plt.title(f"Image after Seuillage {threshold_choice}")
            plt.imshow(seuillage_gradient, cmap="gray")
           
            plt.show()

        # Apply Laplace filter
    def log_image(self):
        
        if self.image is None:
            messagebox.showerror("Error", "Please upload an image first.")
            return
        sigma = float(self.laplace_sigma_entry.get())
        if not sigma:
            messagebox.showerror("Error", "Please enter a sigma value for the Laplace filter.")
            return
        laplace_image = LOG(self.image, sigma)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(self.image, cmap="gray")
        
        plt.subplot(1, 2, 2)
        plt.title("Log Image")
        plt.imshow(laplace_image, cmap="gray")
        
        plt.show()

if __name__ == "__main__":
    app = ImageProcessingApp()
    app.mainloop()
