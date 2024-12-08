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

def LOG(img, sigma):
    height, width = img.shape[:2]
    kernel = log_kernel(sigma)  # The LoG kernel
    kernel_size = kernel.shape[0]
    offset = kernel_size // 2

    # Initialize the output image
    newImage = np.zeros_like(img)

    # Apply the convolution operation (integrating the convolve logic)
    for i in range(offset, height - offset):
        for j in range(offset, width - offset):
            # Element-wise multiplication and summation
            if kernel_size % 2 == 0:
                area = img[i:i+offset+1, j:j+offset+1]
            else:
                area = img[i-offset:i+offset+1, j-offset:j+offset+1]

            imgLaplace = np.sum(kernel * area)
            # Store the result in the output image
            newImage[i, j] = abs(imgLaplace)

    return newImage


# Main GUI
class ImageProcessorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Image Processing Application")
        self.geometry("800x800")
        
        # Attributes
        self.image = None
        self.processed_image = None
        self.noisy_image = None

        # Upload Button
        self.upload_button = ctk.CTkButton(self, text="Upload Image", command=self.upload_image)
        self.upload_button.pack(pady=10)

        # Noise Options
        self.noise_label = ctk.CTkLabel(self, text="Choose Noise:")
        self.noise_label.pack()
        self.noise_var = ctk.StringVar(value="None")
        self.noise_menu = ctk.CTkOptionMenu(self, values=["None", "Gaussian", "Salt and Pepper"], variable=self.noise_var)
        self.noise_menu.pack(pady=5)

        # Noise Parameter
        self.noise_param_label = ctk.CTkLabel(self, text="Parameter (Variance/Percentage):")
        self.noise_param_label.pack()
        self.noise_param_entry = ctk.CTkEntry(self)
        self.noise_param_entry.pack(pady=5)
        
        self.noise_button = ctk.CTkButton(self, text="Add Noise To Image and save it", command=self.process_noise_image)
        self.noise_button.pack(pady=10)

        # Filter Options
        self.filter_label = ctk.CTkLabel(self, text="Choose Filter:")
        self.filter_label.pack()
        self.filter_var = ctk.StringVar(value="None")
        self.filter_menu = ctk.CTkOptionMenu(self, values=["None", "Prewitt", "Sobel", "Robert"], variable=self.filter_var)
        self.filter_menu.pack(pady=5)
    
        self.filter_button_noise = ctk.CTkButton(self, text="Apply Filter on noise Image", command=self.filter_Image_noise)
        self.filter_button_noise.pack(pady=10)
        
        self.filter_button_original = ctk.CTkButton(self, text="Apply Filter on Original Image", command= self.filter_Image_original)
        self.filter_button_original.pack(pady=10)

        # Thresholding Options
        self.threshold_label = ctk.CTkLabel(self, text="Choose Thresholding:")
        self.threshold_label.pack()
        self.threshold_var = ctk.StringVar(value="None")
        self.threshold_menu = ctk.CTkOptionMenu(
            self, 
            values=["None", "Simple", "Hysteresis"], 
            variable=self.threshold_var, 
            command=self.update_seuillage_entries
        )
        self.threshold_menu.pack(pady=5)

        # Thresholding Parameters
        self.threshold_param_label = ctk.CTkLabel(self, text="Parameters DE Seuillage:")
        self.threshold_param_label.pack()
        self.threshold_low_entry = ctk.CTkEntry(self, placeholder_text="seuil_bas")
        self.threshold_low_entry.pack(pady=2)
        self.threshold_high_entry = ctk.CTkEntry(self, placeholder_text="seuil_haut")
        self.threshold_high_entry.pack(pady=2)

        # Laplace Filter Option
        self.laplace_label = ctk.CTkLabel(self, text="Apply Laplace Filter:")
        self.laplace_label.pack()
        self.laplace_var = ctk.StringVar(value="No")
        self.laplace_menu = ctk.CTkOptionMenu(self, values=["Yes", "No"], variable=self.laplace_var)
        self.laplace_menu.pack(pady=5)
        self.laplace_sigma_entry = ctk.CTkEntry(self, placeholder_text="Sigma")
        self.laplace_sigma_entry.pack(pady=2)

        # Process Button
        self.process_button = ctk.CTkButton(self, text="Process Image", command=self.process_image)
        self.process_button.pack(pady=20)

    def update_seuillage_entries(self, choice):
        if choice == "Simple":
            self.threshold_high_entry.configure(state="disabled")
        elif choice == "Hysteresis":
            self.threshold_high_entry.configure(state="normal")
        else:
            self.threshold_low_entry.configure(state="disabled")
            self.threshold_high_entry.configure(state="disabled")


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
        else:
            self.noisy_image = self.image
            
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
            global gradient_image
            gradient_image = Gradient(grad_x, grad_y)
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1,4,4)
            plt.title("Image original")
            plt.imshow(self.noisy_image, cmap="gray")
            
            plt.subplot(1, 4, 1)
            plt.title(f"Image horizontal de {filter_choice}")
            plt.imshow(grad_x, cmap="gray")

            plt.subplot(1, 4, 2)
            plt.title(f"Image vertical de {filter_choice}")
            plt.imshow(grad_y, cmap="gray")

            plt.subplot(1, 4, 3)
            plt.title(f"Gradient de {filter_choice}")
            plt.imshow(gradient_image, cmap="gray")

            plt.show()
            
    def process_image(self):
            
        # Apply thresholding
        threshold_choice = self.threshold_var.get()
        if threshold_choice == "Simple":
            low_threshold = self.threshold_low_entry.get()
            if not low_threshold:
                messagebox.showerror("Error", "Please enter the low threshold for Simple thresholding.")
                return
            seuil = float(self.threshold_low_entry.get())
            binary_image = SeuilSim(self.image, seuil)
            noisy_binary_image = SeuilSim(self.noisy_image, seuil)
            binary_gradient_image = SeuilSim(gradient_image, seuil)
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2,3, 1)
            plt.title("Original Image")
            plt.imshow(self.image, cmap="gray")
           
            
            plt.subplot(2, 3, 2)
            plt.title(f"Noisy Image{noise_choice}")
            plt.imshow(self.noisy_image, cmap="gray")
           
            
            plt.subplot(2, 3, 3)
            plt.title("Gradient")
            plt.imshow(gradient_image, cmap="gray")
           

            plt.subplot(2, 3, 4)
            plt.title(f"original Image after SeuilSim {threshold_choice}")
            plt.imshow(binary_image, cmap="gray")
            
            
            plt.subplot(2, 3, 5)
            plt.title(f"Noisy Image after Seuillage {threshold_choice}")
            plt.imshow(noisy_binary_image, cmap="gray")
           
            
            plt.subplot(2, 3, 6)
            plt.title(f"Gradient Image after Seuillage {threshold_choice}")
            plt.imshow(binary_gradient_image, cmap="gray")
            

            plt.show()
        elif threshold_choice == "Hysteresis":
            seuil_bas = float(self.threshold_low_entry.get())
            seuil_haut = float(self.threshold_high_entry.get())
            if not seuil_bas or not seuil_haut:
                messagebox.showerror("Error", "Please enter both low and high thresholds for Hysteresis thresholding.")
                return
            binary_gradient_image = SeuilHys(gradient_image, seuil_bas, seuil_haut)
            binary_image = SeuilHys(self.image, seuil_bas, seuil_haut)
            noisy_binary_image = SeuilHys(self.noisy_image, seuil_bas, seuil_haut)
            
            plt.figure(figsize=(12, 8))            
            plt.subplot(2,3, 1)
            plt.title("Original Image")
            plt.imshow(self.image, cmap="gray")
            
            
            plt.subplot(2, 3, 2)
            plt.title(f"Noisy Image {noise_choice}")
            plt.imshow(self.noisy_image, cmap="gray")
            
            
            plt.subplot(2, 3, 3)
            plt.title("Gradient")
            plt.imshow(gradient_image, cmap="gray")
           

            plt.subplot(2, 3, 4)
            plt.title(f"original Image after Seuillage {threshold_choice}")
            plt.imshow(binary_image, cmap="gray")
           
            
            plt.subplot(2, 3, 5)
            plt.title(f"Noisy Image after Seuillage {threshold_choice}")
            plt.imshow(noisy_binary_image, cmap="gray")
        
            
            plt.subplot(2, 3, 6)
            plt.title(f"Gradient Image after SeuilSim {threshold_choice}")
            plt.imshow(binary_gradient_image, cmap="gray")
            

            plt.show()

        # Apply Laplace filter
        if self.laplace_var.get() == "Yes":
            sigma = float(self.laplace_sigma_entry.get())
            if not sigma:
                messagebox.showerror("Error", "Please enter a sigma value for the Laplace filter.")
                return
            laplace_image = LOG(self.image, sigma)

        # Display the processed image (example for Laplace image)
        if self.laplace_var.get() == "Yes":
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title("Original Image")
            plt.imshow(self.image, cmap="gray")
           

            plt.subplot(1, 2, 2)
            plt.title("Log Image")
            plt.imshow(laplace_image, cmap="gray")
            

            plt.show()

if __name__ == "__main__":
    app = ImageProcessorApp()
    app.mainloop()

    # # Adjust grid size to 2 rows x 6 columns
    # plt.figure(figsize=(18, 8))
    
    # # Original image and filters
    # plt.subplot(2, 6, 1)
    # plt.title("Image Originale")
    # plt.imshow(image, cmap="gray")
    # plt.axis("off")
    
    # plt.subplot(2, 6, 2)
    # plt.title("Filtre Horizontal Original")
    # plt.imshow(grad_x, cmap="gray")
    # plt.axis("off")
    
    # plt.subplot(2, 6, 3)
    # plt.title("Filtre Vertical Original")
    # plt.imshow(grad_y, cmap="gray")
    # plt.axis("off")
    
    # plt.subplot(2, 6, 4)
    # plt.title("Gradient Original")
    # plt.imshow(gradient_image, cmap="gray")
    # plt.axis("off")
    
    # plt.subplot(2, 6, 5)
    # plt.title("Seuillage Original")
    # plt.imshow(binary_gradient_image, cmap="gray")
    # plt.axis("off")
    
    # plt.subplot(2, 6, 6)
    # plt.title("Filtre Laplace Original")
    # plt.imshow(laplace_image, cmap="gray")
    # plt.axis("off")
    
    # # Noisy image and filters
    # plt.subplot(2, 6, 7)
    # plt.title("Image Bruitée")
    # plt.imshow(noisy_image, cmap="gray")
    # plt.axis("off")
    
    # plt.subplot(2, 6, 8)
    # plt.title("Filtre Horizontal Bruitée")
    # plt.imshow(noisy_grad_x, cmap="gray")
    # plt.axis("off")
    
    # plt.subplot(2, 6, 9)
    # plt.title("Filtre Vertical Bruitée")
    # plt.imshow(noisy_grad_y, cmap="gray")
    # plt.axis("off")
    
    # plt.subplot(2, 6, 10)
    # plt.title("Gradient Bruitée")
    # plt.imshow(noisy_gradient_image, cmap="gray")
    # plt.axis("off")
    
    # plt.subplot(2, 6, 11)
    # plt.title("Seuillage Bruitée")
    # plt.imshow(noisy_binary_gradient_image, cmap="gray")
    # plt.axis("off")
    
    # plt.subplot(2, 6, 12)
    # plt.title("Filtre Laplace Bruitée")
    # plt.imshow(noisy_laplace_image, cmap="gray")
    # plt.axis("off")
    
    # # Layout adjustment
    # plt.tight_layout()
    # plt.show()



# if __name__ == "__main__":
#     image_path = 'cameraman.jpg'  # Chemin de votre image
#     main(image_path)
