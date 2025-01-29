from torchvision.utils import save_image
from torch import zeros
import numpy as np
# from sklearn.decomposition import PCA
# import umap.umap_ as umap
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random

from sklearn.manifold import TSNE


def visualization_output(img_org, outputs, visualisation_path, epoch, batch_size=32, gray=True):
    if gray:
        dimensionality = 1
    else:
        dimensionality = 3

    img_input = img_org.cpu().data
    # img_input = img_input.reshape(round(img_input.shape[0] / dimensionality), dimensionality,
    #                               image_size, image_size)
    img_output = outputs.cpu().data
    # img_output = img_output.reshape(round(img_output.shape[0] / dimensionality), dimensionality,
    #                                 image_size, image_size)
    # num_patch_show = round(inputs.shape[0] / dimensionality)
    num_patch_show = round(img_org.shape[0])
    if num_patch_show > batch_size:
        num_patch_show = batch_size
    concat = zeros(
        [
            num_patch_show * 2,
            dimensionality,
            img_input.shape[2],
            img_input.shape[3]
        ]
    )

    concat[0::2, ...] = img_input[0:num_patch_show, ...]
    concat[1::2, ...] = img_output[0:num_patch_show, ...]

    concat_img_path = visualisation_path / (
            "reconstruction_" + str(epoch+1).zfill(5) + ".png"
    )
    save_image(concat, concat_img_path, nrow=4)

    # # save first layer edge
    # weight_img_path = visualisation_path / (
    #     "weight_" + str(epoch).zfill(5) + ".png"
    # )
    # model.save_cnn_weight_image(str(weight_img_path))


def tsne_plot(latent_vectors, all_labels, int_to_label, out_path):

    tsne = TSNE(n_components=2, random_state=42, learning_rate=500, n_iter=5000)
    latent_tsne = tsne.fit_transform(np.vstack(latent_vectors))

    # pca = PCA(n_components=2)
    # latent_tsne = pca.fit_transform(np.vstack(latent_vectors))

    # umap_model = umap.UMAP(n_components=2)
    # latent_tsne = umap_model.fit_transform(np.vstack(latent_vectors))

    plt.figure(figsize=(10, 8))

    for label in np.unique(all_labels):
        indices = all_labels == label
        plt.scatter(latent_tsne[indices, 0], latent_tsne[indices, 1],
                    marker=".",
                    # s=1.4,
                    label=int_to_label[label])

    plt.title('t-SNE Plot of Latent Vectors')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    # plt.show()

    out_path_name = out_path / "tsne_plot.png"
    plt.savefig(out_path_name, dpi=600)


import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE

def plot_tsne_from_validate(
    all_features,     # [N, feature_dim] numpy array or tensor
    total_labels,     # [N], each in [0..(num_classes-1)]
    class_to_superclass=None,  # dict: leaf_id -> superclass_id if desired
    class_names=None,          # list of length num_classes with string names
    num_samples=2000,
    seed=42,
    title_prefix="",
    save_dir=None,    # directory path where figures will be saved (e.g. "plots/")
    epoch=None        # optional epoch number for filename
):
    """
    Runs t-SNE on a subset of the data and plots:
      1) Color by leaf label
      2) If class_to_superclass is provided, color by superclass

    all_features: 2D array/tensor [N, D] of embeddings
    total_labels: 1D array/tensor [N] of leaf labels
    class_to_superclass: dict { leaf_id -> super_id }, optional
    class_names: optional list (index -> class name) for text labeling
    num_samples: how many points to sample for t-SNE
    seed: random seed for reproducible subsampling + t-SNE init
    title_prefix: string prefix for plot title
    save_dir: if provided, figures will be saved to this directory
    epoch: optional epoch number for naming the output files
    """

    # 0) Convert to numpy if they're torch tensors
    if hasattr(all_features, 'cpu'):
        all_features = all_features.cpu().numpy()
    if hasattr(total_labels, 'cpu'):
        total_labels = total_labels.cpu().numpy()

    N = all_features.shape[0]
    dim = all_features.shape[1]
    print(f"[plot_tsne_from_validate] We have {N} samples, feature_dim = {dim}.")

    # 1) Subsample for t-SNE
    random.seed(seed)
    if N > num_samples:
        idxs = random.sample(range(N), k=num_samples)
        feats_sub = all_features[idxs]
        labels_sub = total_labels[idxs]
    else:
        feats_sub = all_features
        labels_sub = total_labels

    # 2) Run t-SNE
    print("[plot_tsne_from_validate] Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, init='pca',
                learning_rate='auto', random_state=seed, verbose=1)
    feats_2d = tsne.fit_transform(feats_sub)  # shape [num_samples, 2]

    # If save_dir is specified, ensure the directory exists
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)

    # 3) Plot color by leaf label
    leaf_fig, leaf_ax = plt.subplots(figsize=(7, 6))
    sc1 = leaf_ax.scatter(
        feats_2d[:, 0], feats_2d[:, 1],
        c=labels_sub,
        cmap='tab20',
        alpha=0.7,
        s=10
    )
    leaf_ax.set_title(f"{title_prefix} t-SNE (Leaf labels)")
    leaf_cb = leaf_fig.colorbar(sc1, ax=leaf_ax, fraction=0.046, pad=0.04)
    leaf_cb.set_label("Leaf Label Index")

    # (Optional) text labels for small subset
    # for i in range(50):  # label 50 random points
    #     idx_pt = random.randint(0, len(feats_2d)-1)
    #     x, y = feats_2d[idx_pt]
    #     leaf_id = labels_sub[idx_pt]
    #     name = class_names[leaf_id] if class_names else str(leaf_id)
    #     leaf_ax.text(x, y, name, fontsize=6)

    # Save or show
    if save_dir is not None:
        # Construct file name
        if epoch is not None:
            leaf_filename = f"tsne_leaf_epoch{epoch}.png"
        else:
            leaf_filename = "tsne_leaf.png"
        leaf_path = os.path.join(save_dir, leaf_filename)
        leaf_fig.savefig(leaf_path, dpi=200, bbox_inches='tight')
        print(f"[plot_tsne_from_validate] Saved leaf-level t-SNE to {leaf_path}")
    else:
        plt.show()

    plt.close(leaf_fig)

    # 4) (Optional) color by SUPERCLASS
    if class_to_superclass is not None:
        super_labels_sub = np.array([class_to_superclass[l] for l in labels_sub])
        super_fig, super_ax = plt.subplots(figsize=(7, 6))
        sc2 = super_ax.scatter(
            feats_2d[:, 0], feats_2d[:, 1],
            c=super_labels_sub,
            cmap='tab10',
            alpha=0.7,
            s=10
        )
        super_ax.set_title(f"{title_prefix} t-SNE (Superclasses)")
        super_cb = super_fig.colorbar(sc2, ax=super_ax, fraction=0.046, pad=0.04)
        super_cb.set_label("Superclass Index")

        if save_dir is not None:
            if epoch is not None:
                super_filename = f"tsne_super_epoch{epoch}.png"
            else:
                super_filename = "tsne_super.png"
            super_path = os.path.join(save_dir, super_filename)
            super_fig.savefig(super_path, dpi=200, bbox_inches='tight')
            print(f"[plot_tsne_from_validate] Saved superclass t-SNE to {super_path}")
        else:
            plt.show()

        plt.close(super_fig)

    print("[plot_tsne_from_validate] Done.")

