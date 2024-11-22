import cv2
import numpy as np
from typing import Tuple

def normalize_img(img: np.ndarray) -> np.ndarray:
    normalized = (img - np.mean(img)) / np.std(img)
    return 255 * (normalized - np.min(normalized)) / (np.max(normalized) - np.min(normalized))

def compute_covariance_matrix(mat: np.ndarray) -> np.ndarray:
    # each row (2D pixel) is a observation (not a variable)
    return np.cov(mat, rowvar=False)

def compute_eigen_values_vectors(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    evalues, evectors = np.linalg.eig(mat)
    idx = np.argsort(evalues)[::-1]
    evalues = evectors[idx]
    evectors = evectors[:, idx]
    return evalues, evectors

def project_to_basis(mat: np.ndarray, basis: np.ndarray) -> np.ndarray:
    return np.dot(mat, basis)

def reconstruct_original(x_pca: np.ndarray, mean: np.ndarray, eigenvectors: np.ndarray) -> np.ndarray:
    return (np.dot(x_pca, eigenvectors.T) + mean)

def high_error(mat1: np.ndarray, mat2: np.ndarray, threshold: float) -> bool:
    mse = np.mean((mat1 - mat2) ** 2)
    
    print(mse)
    if mse >= threshold:
        return True
    
    return False

input_img = cv2.imread('./dataset/1.png', cv2.IMREAD_GRAYSCALE)

gradient_x = cv2.Sobel(input_img, cv2.CV_64F, 1, 0, ksize=3)
gradient_y = cv2.Sobel(input_img, cv2.CV_64F, 0, 1, ksize=3)

gradient_x_normalized = normalize_img(img=gradient_x)
gradient_y_normalized = normalize_img(img=gradient_y) 

# cv2.imwrite('./output/input_img.jpg', input_img)
# cv2.imwrite('./output/gradient_x.jpg', gradient_x)
# cv2.imwrite('./output/gradient_y.jpg', gradient_y)
# cv2.imwrite('./output/gradient_x_normalized.jpg', gradient_x_normalized)
# cv2.imwrite('./output/gradient_y_normalized.jpg', gradient_y_normalized)


gradient_x_flattened, gradient_y_flattened = gradient_x.flatten(), gradient_y.flatten()
X = np.vstack([gradient_x_flattened, gradient_y_flattened]).T
N_X = (X - np.mean(X)) / np.std(X)
COV_X = compute_covariance_matrix(N_X)
E_VAL, E_VEC = compute_eigen_values_vectors(COV_X)
PCA_X = project_to_basis(N_X, E_VEC)
PCA_X_IMAGE = PCA_X.reshape(input_img.shape[0], input_img.shape[1], 2)
PCA_X_IMAGE_SEP = PCA_X_IMAGE.transpose(2, 0, 1)

for i, pca_img in enumerate(PCA_X_IMAGE_SEP):
    cv2.imwrite(f'./output_pca_{i}.jpg', normalize_img(pca_img))

RECON_X = reconstruct_original(PCA_X, np.mean(X), E_VEC)
error = high_error(X, RECON_X, 400.)
print(f"Has high error: {error}")
