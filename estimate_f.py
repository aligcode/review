import numpy as np



def estimate_F(points1, points2): 
    
    points1 = np.hstack((points1, np.ones((points1.shape[0], 1))))
    points2 = np.hstack((points2, np.ones((points2.shape[0], 1))))
    
    N = points1.shape[0]
    A = np.zeros((N, 9))
    for i in range(N):
        x1, x2, y1, y2 = points1[i, 0], points2[i, 0], points1[i, 1], points2[i, 2]
        A[i, :] = [x1*x2, x1*y2, x2, y1*x2, y1*y2, y2, x1, y1, 1]

    _, _, V_T = np.linalg.svd(A)
    F = V_T[-1].reshape(3, 3)
    
    U, S, VT = np.linalg.svd(F)
    S[2] = 0
    
    return U @ np.diag(S) @ VT

points1 = np.array([
    [100, 150],
    [200, 180],
    [300, 200],
    [400, 220],
    [120, 250],
    [220, 280],
    [320, 300],
    [420, 320]
])

points2 = np.array([
    [110, 160],
    [210, 190],
    [310, 210],
    [410, 230],
    [130, 260],
    [230, 290],
    [330, 310],
    [430, 330]
])

print(f"Esimated fundamental matrix: {estimate_F(points1, points2)}")


import numpy as np

# Example rank-1 matrix
A = np.array([
    [1, 2],
    [2, 4]
])

# Perform SVD
U, S, VT = np.linalg.svd(A)

print("U:\n", U)
print("S (Singular values):", S)
print("VT:\n", VT)