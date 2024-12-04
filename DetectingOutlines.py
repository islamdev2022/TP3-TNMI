import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from imageio import imwrite

# Fonction pour ajouter un bruit gaussien
def gaussien(image, variance):
    """Ajoute un bruit blanc gaussien à une image sans fonctions prédéfinies."""
    mean = 0
    noisy_image = image.copy()
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # Génération de bruit gaussien : Box-Muller
            u1, u2 = np.random.rand(), np.random.rand()
            z = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
            noise = mean + np.sqrt(variance) * z
            noisy_image[i, j] += noise
            # Limiter entre 0 et 1
            noisy_image[i, j] = max(0, min(1, noisy_image[i, j]))
    return noisy_image

# Fonction pour ajouter un bruit poivre et sel
def poivre_sel(image, pourcentage):
    """Ajoute un bruit poivre et sel à une image sans fonctions prédéfinies."""
    noisy_image = image.copy()
    num_pixels = int(pourcentage * image.size)
    height, width = image.shape

    # Ajouter du poivre (noir)
    for _ in range(num_pixels // 2):
        i, j = np.random.randint(0, height), np.random.randint(0, width)
        noisy_image[i, j] = 0

    # Ajouter du sel (blanc)
    for _ in range(num_pixels // 2):
        i, j = np.random.randint(0, height), np.random.randint(0, width)
        noisy_image[i, j] = 1

    return noisy_image

# Fonction pour appliquer un filtre de Prewitt (horizontal et vertical séparés)
def prewitt(image):
    """Applique un filtre de Prewitt sur une image sans fonctions prédéfinies."""
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    # Filtrage horizontal
    grad_x = np.zeros_like(image)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            grad_x[i, j] = np.sum(kernel_x * image[i - 1:i + 2, j - 1:j + 2])

    # Filtrage vertical
    grad_y = np.zeros_like(image)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            grad_y[i, j] = np.sum(kernel_y * image[i - 1:i + 2, j - 1:j + 2])

    return grad_x, grad_y


# Fonction pour appliquer un filtre de Sobel (horizontal et vertical séparés)
def sobel(image):
    """Applique un filtre de Sobel sur une image sans fonctions prédéfinies."""
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # Filtrage horizontal
    grad_x = np.zeros_like(image)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            grad_x[i, j] = np.sum(kernel_x * image[i - 1:i + 2, j - 1:j + 2])

    # Filtrage vertical
    grad_y = np.zeros_like(image)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            grad_y[i, j] = np.sum(kernel_y * image[i - 1:i + 2, j - 1:j + 2])

    return grad_x, grad_y


# Fonction pour appliquer un filtre de Robert (horizontal et vertical séparés)
def robert(image):
    """Applique un filtre de Robert sur une image sans fonctions prédéfinies."""
    kernel_x = np.array([[1, 0], [0, -1]])
    kernel_y = np.array([[0, 1], [-1, 0]])

    # Dimensions de l'image
    grad_x = np.zeros_like(image)
    grad_y = np.zeros_like(image)

    for i in range(image.shape[0] - 1):
        for j in range(image.shape[1] - 1):
            # Calcul du gradient horizontal (Robert)
            grad_x[i, j] = np.sum(kernel_x * image[i:i+2, j:j+2])
            # Calcul du gradient vertical (Robert)
            grad_y[i, j] = np.sum(kernel_y * image[i:i+2, j:j+2])

    return grad_x, grad_y


# Fonction pour appliquer le seuillage simple
def SeuilSim(image, seuil):
    """Applique le seuillage simple à une image."""
    binary_image = np.where(image > seuil, 1, 0)
    return binary_image


# Fonction pour calculer le gradient entre deux matrices (par exemple, grad_x et grad_y)
def Gradient(grad_x, grad_y):
    """Calcul du gradient global entre deux matrices (gradient horizontal et vertical)."""
    # Calculer le gradient en utilisant la racine carrée de la somme des carrés
    gradient = np.sqrt(grad_x**2 + grad_y**2)
    return gradient


# Fonction principale pour gérer l'affichage et l'application des filtres
def main(image_path):
    # Ouverture de l'image à partir du chemin fourni
    img = Image.open(image_path).convert('L')  # Convertir en niveaux de gris
    image = np.array(img) / 255.0  # Normaliser l'image en [0, 1]

    # Choix du filtre
    filtre = int(input("Entrez le numéro du filtre que vous voulez utiliser (1: Prewitt, 2: Sobel, 3: Robert) : "))

    # Appliquer le filtre sélectionné
    if filtre == 1:  # Prewitt
        grad_x, grad_y = prewitt(image)
    elif filtre == 2:  # Sobel
        grad_x, grad_y = sobel(image)
    elif filtre == 3:  # Robert
        grad_x, grad_y = robert(image)
    else:
        raise ValueError("Filtre non reconnu")

    # Calculer le gradient global entre grad_x et grad_y
    gradient_image = Gradient(grad_x, grad_y)

    # Demander quel type de bruit ajouter
    bruit = input("Voulez-vous ajouter un bruit ? (gaussien/poivre_sel/aucun) : ")
    if bruit == "gaussien":
        variance = float(input("Entrez la variance du bruit gaussien (par exemple 0.01) : "))
        image_bruit = gaussien(image, variance)
        imwrite("image_bruitee.png", (image_bruit * 255).astype(np.uint8))  # Sauvegarde de l'image bruitée
    elif bruit == "poivre_sel":
        pourcentage = float(input("Entrez le pourcentage du bruit poivre et sel (par exemple 0.05) : "))
        image_bruit = poivre_sel(image, pourcentage)
        imwrite("image_bruitee.png", (image_bruit * 255).astype(np.uint8))  # Sauvegarde de l'image bruitée
    else:
        image_bruit = image

    # Appliquer le seuillage simple
    seuil = float(input("Entrez la valeur du seuil pour appliquer le seuillage : "))

    # Seuillage sur l'image originale
    binary_original = SeuilSim(image, seuil)

    # Seuillage sur l'image bruitée
    binary_bruitee = SeuilSim(image_bruit, seuil)

    # Affichage des résultats
    plt.figure(figsize=(15, 10))

    # Image originale
    plt.subplot(3, 3, 1)
    plt.title("Image Originale")
    plt.imshow(image, cmap="gray")
    plt.axis("off")

    # Résultat du filtre horizontal
    plt.subplot(3, 3, 2)
    plt.title(f"Filtre Horizontal ({['Prewitt', 'Sobel', 'Robert'][filtre - 1]})")
    plt.imshow(grad_x, cmap="gray")
    plt.axis("off")

    # Résultat du filtre vertical
    plt.subplot(3, 3, 3)
    plt.title(f"Filtre Vertical ({['Prewitt', 'Sobel', 'Robert'][filtre - 1]})")
    plt.imshow(grad_y, cmap="gray")
    plt.axis("off")

    # Gradient global
    plt.subplot(3, 3, 4)
    plt.title(f"Gradient Global ({['Prewitt', 'Sobel', 'Robert'][filtre - 1]})")
    plt.imshow(gradient_image, cmap="gray")
    plt.axis("off")

    # Image bruitée
    plt.subplot(3, 3, 5)
    plt.title("Image Bruitée")
    plt.imshow(image_bruit, cmap="gray")
    plt.axis("off")

    # Seuillage sur l'image originale
    plt.subplot(3, 3, 6)
    plt.title("Seuillage Image Originale")
    plt.imshow(binary_original, cmap="gray")
    plt.axis("off")

    # Seuillage sur l'image bruitée
    plt.subplot(3, 3, 7)
    plt.title("Seuillage Image Bruitée")
    plt.imshow(binary_bruitee, cmap="gray")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Chemin de l'image
    image_path = 'prj.bmp'
    main(image_path)
