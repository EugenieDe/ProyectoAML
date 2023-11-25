import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

def visualize_dist_instr(root, labeled_path, unlabeled_path, save_folder,num):
    labeled_path = labeled_path
    unlabeled_path = unlabeled_path
    root = root
    with open(labeled_path, 'r') as f:
        id_labeled = f.read().splitlines()
    with open(unlabeled_path, 'r') as f:
        id_unlabeled = f.read().splitlines()

    labels_labeled = np.zeros(8)
    labels_unlabeled = np.zeros(8)

    for i in range(0, len(id_labeled)):
        idx = id_labeled[i]
        mask = np.array(Image.open(os.path.join(root, idx.split(' ')[1])))
        labels = np.unique(mask)
        for j in labels:
            if j>3:
                labels_labeled[j-2] += 1
            else:
                labels_labeled[j] += 1

    for i in range(0, len(id_unlabeled)):
        idx = id_unlabeled[i]
        mask = np.array(Image.open(os.path.join(root, idx.split(' ')[1])))
        labels = np.unique(mask)
        for j in labels:
            if j>3:
                labels_unlabeled[j-2] += 1
            else:
                labels_unlabeled[j] += 1

    print(f"Labeled: {labels_labeled}\n")
    print(f"Unabeled: {labels_unlabeled}\n")
    print(f"Difference: {np.abs(labels_labeled-labels_unlabeled)}")

    alls =  labels_labeled + labels_unlabeled
    labels_labeled = np.divide(labels_labeled, alls)
    labels_unlabeled = np.divide(labels_unlabeled, alls)

    names = ["BF", "PF", "LND", "MCS", "UP", "SI", "CA"]

    plt.figure()
    plt.bar(np.arange(1,8), labels_labeled[1:], width=0.4, align='edge', label="labeled")
    plt.bar(np.arange(1,8), labels_unlabeled[1:], width=-0.4, align='edge', label='unnlabeled')
    plt.title(f'Splits {100/num}%')
    plt.legend()
    plt.xticks(np.arange(1,8), names)
    plt.xlabel("Instrument categories")
    plt.ylabel("Percentage of instances")
    plt.savefig(os.path.join(save_folder, f"Stats_labels_1_{num}.png"))

def main():
    root = '/home/eugenie/These/data/endovis2018/unsorted'

    save_folder = "/home/eugenie/These/ProyectoAML/data/stats/"

    labeled_1_2_path = "/home/eugenie/These/UniMatch/splits/endovis2018/unsorted/1_2/labeled.txt"
    unlabeled_1_2_path = "/home/eugenie/These/UniMatch/splits/endovis2018/unsorted/1_2/unlabeled.txt"

    labeled_1_4_path = "/home/eugenie/These/UniMatch/splits/endovis2018/unsorted/1_4/labeled.txt"
    unlabeled_1_4_path = "/home/eugenie/These/UniMatch/splits/endovis2018/unsorted/1_4/unlabeled.txt"

    labeled_1_8_path = "/home/eugenie/These/UniMatch/splits/endovis2018/unsorted/1_8/labeled.txt"
    unlabeled_1_8_path = "/home/eugenie/These/UniMatch/splits/endovis2018/unsorted/1_8/unlabeled.txt"

    visualize_dist_instr(root, labeled_1_2_path, unlabeled_1_2_path, save_folder, 2)
    visualize_dist_instr(root, labeled_1_4_path, unlabeled_1_4_path, save_folder, 4)
    visualize_dist_instr(root, labeled_1_8_path, unlabeled_1_8_path, save_folder, 8)

