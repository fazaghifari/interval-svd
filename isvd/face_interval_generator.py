import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from tqdm import tqdm

def face_loader(path, size=32):
    """load face data

    Args:
        path (str): path of the face file
        size (int, optional): size of the image 32x32 or 64x64. Defaults to 32.

    image source: http://www.cad.zju.edu.cn/home/dengcai/Data/FaceData.html

    Returns:
        all_faces (list): list of faces
    """
    data = scipy.io.loadmat(path)
    all_faces = []
    single_faces = []
    for idx in range(400):
        face = data['fea'][idx,:].reshape(size,size).T
        single_faces.append(face)
        if (idx+1) % 10 == 0:
            all_faces.append(single_faces)
            single_faces = []
    return all_faces


def _within_array(arr: np.ndarray, x: int, y: int) -> bool:
    """helper function in get_neighbors to check whether index is within array

    Args:
        arr (np.ndarray): image array
        x (int): pixel coordinate x
        y (int): pixel coordinate y

    Returns:
        bool: condition True or False
    """
    if x <= arr.shape[0]-1 and x <= arr.shape[0]-1:
        cond = True
    else:
        cond = False
    return cond


def get_neighbors(arr: np.ndarray, i: int, j: int, r=1) -> list:
    """get neighbors around a pixel

    Args:
        arr (np.ndarray): image array
        i (int): pixel coordinate
        j (int): pixel coordinate
        r (int, optional): L1 radius. Defaults to 1.

    Returns:
        list: list of neighbors
    """
    neighbors = []
    for x in range(max(0, i-r), min(arr.shape[0], i+r+1)):
        for y in range(max(0, j-r), min(arr.shape[1], j+r+1)):
            if (x,y) != (i,j) and np.abs(x-i) + np.abs(y-j) <= r*2 and _within_array(arr,x,y):
                neighbors.append(arr[x,y])
    return neighbors


def neighbor_pixels(img: np.ndarray, r: int) -> list:
    """get neighboring pixels for each pixel in img

    Args:
        img (np.ndarray): image array
        r (int): L1 radius

    Returns:
        list: list of neighbors for each pixels
    """
    x = img.shape[0]
    y = img.shape[1]
    s = []
    for i in range(x):
        s_i = []
        for j in range(y):
            neighbors = get_neighbors(img, i, j, r)
            s_i.append(neighbors)
        s.append(s_i)
    
    return s

def delta_pixels(neighbors_array: list, alpha: float) -> np.ndarray:
    """calculate the delta pixels given neighbors array from `neighbor_pixels`

    Args:
        neighbors_array (list): output from `neighbor_pixels()`
        alpha (float): arbitrary coefficient

    Returns:
        np.ndarray: delta array
    """
    x = len(neighbors_array)
    y = len(neighbors_array[0])

    arr = []
    for i in range(x):
        arr_i = []
        for j in range(y):
            delta_ij = np.std(neighbors_array[i][j]) * alpha
            arr_i.append(delta_ij)
        arr.append(arr_i)
    arr = np.array(arr)

    return arr

def compute_interval(img: np.ndarray, r: int, alpha: float) -> list:
    """compute interval given an image

    Args:
        img (np.ndarray): single image array
        r (int): L1 radius
        alpha (float): arbitrary coefficient for delta

    Returns:
        list: list of output consist of [delta, img_low, img_up]
    """
    s = neighbor_pixels(img, r)
    delta = delta_pixels(s, alpha)
    img_low = img - delta
    img_up = img + delta

    return [delta, img_low, img_up]

def iterate(all_faces: list, r: int, alpha: float):
    faces_avg = []
    faces_low = []
    faces_up = []

    for faces in tqdm(all_faces):
        temp_avg = []
        temp_low = []
        temp_up = []
        for face in faces:
            delta, img_low, img_up = compute_interval(face, 3, 2)
            temp_low.append(img_low.flatten())
            temp_up.append(img_up.flatten())
            temp_avg.append(face.flatten())
        faces_avg.append(temp_avg)
        faces_low.append(temp_low)
    faces_up.append(temp_up)

    return [faces_avg, faces_low, faces_up]