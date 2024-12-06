import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


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


# Fonction de seuillage par hystérésis
def SeuilHys(image, seuil_bas, seuil_haut):
    binary_image = np.zeros_like(image)
    binary_image[image > seuil_haut] = 1
    visited = np.zeros_like(image, dtype=bool)
    queue = []

    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            if binary_image[i, j] == 1 and not visited[i, j]:
                queue.append((i, j))

    while queue:
        x, y = queue.pop(0)
        if visited[x, y]:
            continue
        visited[x, y] = True
        if image[x, y] > seuil_bas:
            binary_image[x, y] = 1
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1]:
                        queue.append((nx, ny))

    return binary_image


# Fonction principale
def main(image_path):
    img = Image.open(image_path).convert('L')
    image = np.array(img) / 255.0

    # Demander le type de bruit
    noise_choice = int(input("Choisissez le bruit (1: Gaussien, 2: Poivre et Sel, 3: Aucun): "))
    if noise_choice == 1:
        variance = float(input("Entrez la variance du bruit gaussien (ex: 0.02): "))
        noisy_image = gaussien(image, variance=variance)
    elif noise_choice == 2:
        pourcentage = float(input("Entrez le pourcentage de bruit poivre et sel (ex: 0.05): "))
        noisy_image = poivre_sel(image, pourcentage=pourcentage)
    else:
        noisy_image = image.copy()

    # Demander le filtre à appliquer
    filter_choice = int(input("Choisissez le filtre (1: Prewitt, 2: Sobel, 3: Robert): "))
    if filter_choice == 1:
        grad_x, grad_y = prewitt(image)
        noisy_grad_x, noisy_grad_y = prewitt(noisy_image)
    elif filter_choice == 2:
        grad_x, grad_y = sobel(image)
        noisy_grad_x, noisy_grad_y = sobel(noisy_image)
    elif filter_choice == 3:
        grad_x, grad_y = robert(image)
        noisy_grad_x, noisy_grad_y = robert(noisy_image)

    # Calculer les gradients
    gradient_image = Gradient(grad_x, grad_y)
    noisy_gradient_image = Gradient(noisy_grad_x, noisy_grad_y)

    # Demander le type de seuillage
    threshold_choice = int(input("Choisissez le type de seuillage (1: Simple, 2: Hystérésis): "))
    if threshold_choice == 1:
        seuil = float(input("Entrez la valeur du seuil (0 à 1): "))
        binary_gradient_image = SeuilSim(gradient_image, seuil)
        noisy_binary_gradient_image = SeuilSim(noisy_gradient_image, seuil)
    elif threshold_choice == 2:
        seuil_bas = float(input("Entrez la valeur du seuil bas (0 à 1): "))
        seuil_haut = float(input("Entrez la valeur du seuil haut (0 à 1): "))
        binary_gradient_image = SeuilHys(gradient_image, seuil_bas, seuil_haut)
        noisy_binary_gradient_image = SeuilHys(noisy_gradient_image, seuil_bas, seuil_haut)

    # Affichage des résultats
    plt.figure(figsize=(15, 10))

    # Image originale
    plt.subplot(2, 5, 1)
    plt.title("Image Originale")
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 5, 2)
    plt.title("Filtre Horizontal Original")
    plt.imshow(grad_x, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 5, 3)
    plt.title("Filtre Vertical Original")
    plt.imshow(grad_y, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 5, 4)
    plt.title("Gradient Original")
    plt.imshow(gradient_image, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 5, 5)
    plt.title("Seuillage Original")
    plt.imshow(binary_gradient_image, cmap="gray")
    plt.axis("off")
    # Image bruitée
    plt.subplot(2, 5, 6)
    plt.title("Image Bruitée")
    plt.imshow(noisy_image, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 5, 7)
    plt.title("Filtre Horizontal Bruitée")
    plt.imshow(noisy_grad_x, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 5, 8)
    plt.title("Filtre Vertical Bruitée")
    plt.imshow(noisy_grad_y, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 5, 9)
    plt.title("Gradient Bruitée")
    plt.imshow(noisy_gradient_image, cmap="gray")
    plt.axis("off")

    plt.subplot(2, 5, 10)
    plt.title("Seuillage Bruitée")
    plt.imshow(noisy_binary_gradient_image, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_path = 'prj.bmp'  # Chemin de votre image
    main(image_path)
