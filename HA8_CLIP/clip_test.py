import torch
import clip
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import Caltech101
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import seaborn as sns
import random

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ===============================
#  Dataset: Caltech101
# ===============================
transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB")),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

dataset = Caltech101(root='./data', download=False, transform=transform)
classes = dataset.categories
print(f'Number of classes: {len(classes)}')

if hasattr(dataset, "class_to_idx"):
    # Newer torchvision versions
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
else:
    # Older versions of Caltech101
    # dataset.categories gives list of class names
    idx_to_class = {i: name for i, name in enumerate(dataset.categories)}

# ===============================
#  Load CLIP model
# ===============================
model, preprocess = clip.load("ViT-B/16", device=device)
model.eval()

# ===============================
#  Define text prompt templates
# ===============================
templates = [
    'a photo of a {}.',
    'a painting of a {}.',
    'a plastic {}.',
    'a sculpture of a {}.',
    'a sketch of a {}.',
    'a tattoo of a {}.',
    'a toy {}.',
    'a rendition of a {}.',
    'a embroidered {}.',
    'a cartoon {}.',
    'a {} in a video game.',
    'a plushie {}.',
    'a origami {}.',
    'art of a {}.',
    'graffiti of a {}.',
    'a drawing of a {}.',
    'a doodle of a {}.',
    'a photo of the {}.',
    'a painting of the {}.',
    'the plastic {}.',
    'a sculpture of the {}.',
    'a sketch of the {}.',
    'a tattoo of the {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'the embroidered {}.',
    'the cartoon {}.',
    'the {} in a video game.',
    'the plushie {}.',
    'the origami {}.',
    'art of the {}.',
    'graffiti of the {}.',
    'a drawing of the {}.',
    'a doodle of the {}.'
]

simple_templates = ['a photo of a {}.']

optimal_templates = [
    'a photo of a {}.',
    'a photo of the {}.',
    'a picture of a {}.',
    'a cartoon of a {}.',
    'a close-up photo of a {}.',
    'a cropped photo of the {}.',
    'a bright photo of the {}.',
    'a low-resolution photo of the {}.',
    'a black and white photo of a {}.',
]

# ===============================
#  Encode text prompts
# ===============================
def build_text_features(classes, templates):
    all_features = []
    with torch.no_grad():
        for c in classes:
            texts = [t.format(c) for t in templates]
            tokens = clip.tokenize(texts).to(device)
            embeddings = model.encode_text(tokens)
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
            all_features.append(embeddings.mean(0))
        text_features = torch.stack(all_features)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

# ===============================
#  Zero-shot evaluation
# ===============================
def zero_shot_accuracy(dataloader, text_features):
    correct, total = 0, 0
    with torch.no_grad():
        for images, targets in tqdm(dataloader):
            images = images.to(device)
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = image_features @ text_features.T
            preds = similarity.argmax(dim=-1).cpu()
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / total

# ===============================
#  Build text features
# ===============================
print('Building simple prompt features...')
text_simple = build_text_features(classes, simple_templates)

print('Building ensemble prompt features...')
text_ensemble = build_text_features(classes, templates)

print('Building optimal ensemble prompt features...')
text_optimal = build_text_features(classes, optimal_templates)

# ===============================
#  Evaluate CLIP zero-shot accuracy
# ===============================
loader = DataLoader(dataset, batch_size=32, shuffle=False)

acc_simple = zero_shot_accuracy(loader, text_simple)
acc_ensemble = zero_shot_accuracy(loader, text_ensemble)
acc_optimal = zero_shot_accuracy(loader, text_optimal)

print(f'Simple template accuracy: {acc_simple * 100:.2f}%')
print(f'Ensemble template accuracy: {acc_ensemble * 100:.2f}%')
print(f'Optimal emplate accuracy: {acc_optimal * 100:.2f}%')


# ===============================
#  Visualization (random samples + correct label mapping)
# ===============================
def visualize_predictions(
    dataset,
    text_features,
    idx_to_class,
    num_examples=8,
    save_path="clip_predictions_example.png",
    sample_indices=None
):
    plt.figure(figsize=(12, 6))

    if sample_indices is None:
        sample_indices = random.sample(range(len(dataset)), num_examples)

    for i, idx in enumerate(sample_indices):
        image, label = dataset[idx]
        image_input = image.unsqueeze(0).to(device)
        with torch.no_grad():
            image_feat = model.encode_image(image_input)
            image_feat /= image_feat.norm(dim=-1, keepdim=True)
            sim = (100.0 * image_feat @ text_features.T).softmax(dim=-1)
            pred = sim.argmax(dim=-1).item()

        true_name = idx_to_class[label]
        pred_name = idx_to_class[pred]

        plt.subplot(2, 4, i + 1)
        plt.imshow(transforms.ToPILImage()(image))
        plt.axis('off')
        plt.title(f"True: {true_name[:18]}\nPred: {pred_name[:18]}", fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved prediction figure to {save_path}")

sample_indices = random.sample(range(len(dataset)), 8)
visualize_predictions(dataset, text_ensemble, idx_to_class,
                      num_examples=8,
                      save_path="clip_ViT-B_16_ensemble.png",
                      sample_indices=sample_indices)
visualize_predictions(dataset, text_simple, idx_to_class,
                      num_examples=8,
                      save_path="clip_ViT-B_16_simple.png",
                      sample_indices=sample_indices)

visualize_predictions(dataset, text_optimal, idx_to_class,
                      num_examples=8,
                      save_path="clip_ViT-B_16_optimal.png",
                      sample_indices=sample_indices)

def visualize_predictions_with_heatmap(
    dataset,
    text_features,
    idx_to_class,
    model,
    device='cuda',
    num_examples=4,
    topk=5,
    save_path="clip_predictions_heatmap.png",
    sample_indices=None
):
    """
    Visualize CLIP's top-k zero-shot predictions with cosine similarity heatmap.

    Args:
        dataset (torch.utils.data.Dataset): Image dataset.
        text_features (torch.Tensor): Encoded text representations. (num_classes, 512)
        idx_to_class (dict): Mapping from class indices to names.
        model (torch.nn.Module): CLIP model.
        device (str): 'cuda' or 'cpu'.
        num_examples (int): Number of images to visualize.
        topk (int): Show top-k predicted classes in the heatmap.
        save_path (str): Path to save the figure.
        sample_indices (list): Optional, specify which samples to visualize.
    """

    if sample_indices is None:
        sample_indices = random.sample(range(len(dataset)), num_examples)

    plt.figure(figsize=(14, 4 * num_examples))

    for i, idx in enumerate(sample_indices):
        image, label = dataset[idx]
        image_input = image.unsqueeze(0).to(device)

        with torch.no_grad():
            image_features = model.encode_image(image_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            sims = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            sims = sims[0].cpu().numpy()

        # top-k predicted classes
        topk_idx = np.argsort(sims)[::-1][:topk]
        topk_names = [idx_to_class[j] for j in topk_idx]
        topk_scores = sims[topk_idx]

        true_name = idx_to_class[label]
        pred_name = topk_names[0]

        # ---------- Plot: Image ----------
        plt.subplot(num_examples, 2, 2 * i + 1)
        plt.imshow(transforms.ToPILImage()(image))
        plt.axis("off")
        plt.title(f"True: {true_name[:18]}\nPred: {pred_name[:18]}", fontsize=10)

        # ---------- Plot: Heatmap ----------
        plt.subplot(num_examples, 2, 2 * i + 2)
        sns.heatmap(
            topk_scores.reshape(1, -1),
            annot=True, fmt=".2f", cmap="coolwarm", cbar=False,
            xticklabels=topk_names, yticklabels=['similarity']
        )
        plt.xticks(rotation=45, ha='right', fontsize=9)
        plt.yticks(fontsize=9)
        plt.title(f"Top-{topk} predicted classes", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization with heatmap to: {save_path}")


sample_indices = random.sample(range(len(dataset)), 4)

visualize_predictions_with_heatmap(
    dataset=dataset,
    text_features=text_ensemble,
    idx_to_class=idx_to_class,
    model=model,
    device=device,
    num_examples=4,
    topk=5,
    save_path="ensemble_clip_zero_shot_heatmap.png",
    sample_indices=sample_indices
)

visualize_predictions_with_heatmap(
    dataset=dataset,
    text_features=text_simple,
    idx_to_class=idx_to_class,
    model=model,
    device=device,
    num_examples=4,
    topk=5,
    save_path="simple_clip_zero_shot_heatmap.png",
    sample_indices=sample_indices
)

visualize_predictions_with_heatmap(
    dataset=dataset,
    text_features=text_optimal,
    idx_to_class=idx_to_class,
    model=model,
    device=device,
    num_examples=4,
    topk=5,
    save_path="optimal_clip_zero_shot_heatmap.png",
    sample_indices=sample_indices
)

def visualize_clip_similarity_matrix(
    dataset,
    text_features,
    idx_to_class,
    model,
    device='cuda',
    num_images=8,
    num_classes=8,
    save_path="clip_similarity_matrix.png",
    sample_indices=None,
):
    """
    Visualize image-text cosine similarity matrix.

    Args:
        dataset (torch.utils.data.Dataset): dataset containing (image, label)
        text_features (torch.Tensor): text embeddings (n_classes, 512)
        idx_to_class (dict): mapping from class index to name
        model (torch.nn.Module): CLIP model
        device (str): 'cuda' or 'cpu'
        num_images (int): number of images to visualize
        num_classes (int): number of classes (subset of total classes)
        save_path (str): filename to save the figure.
        sample_indices (list): optional indices to visualize
    """

    # Randomly pick some images and classes
    if sample_indices is None:
        sample_indices = random.sample(range(len(dataset)), num_images)

    sampled_images, sampled_labels = [], []
    for idx in sample_indices:
        image, label = dataset[idx]
        sampled_images.append(image)
        sampled_labels.append(label)

    # Select a small subset of classes for visualization
    sampled_class_indices = random.sample(range(len(idx_to_class)), num_classes)
    sampled_classes = [idx_to_class[c] for c in sampled_class_indices]

    with torch.no_grad():
        # Image features
        image_inputs = torch.stack(sampled_images).to(device)
        image_features = model.encode_image(image_inputs)
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # Text subset features
        subset_text_features = text_features[sampled_class_indices]
        subset_text_features /= subset_text_features.norm(dim=-1, keepdim=True)

        # Compute similarity
        similarity = image_features @ subset_text_features.T
        similarity = similarity.cpu().numpy()

    # ============ Plot ============
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        similarity,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        xticklabels=sampled_classes,
        yticklabels=[idx_to_class[l] for l in sampled_labels],
    )
    plt.title(f"CLIP image-text cosine similarity ({model.visual.input_resolution} px)")
    plt.xlabel("Text Classes")
    plt.ylabel("Image True Labels")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved image-text similarity matrix → {save_path}")


visualize_clip_similarity_matrix(
    dataset=dataset,
    text_features=text_optimal,
    idx_to_class=idx_to_class,
    model=model,
    device=device,
    num_images=8,
    num_classes=8,
    save_path="clip_similarity_matrix_optimal.png"
)

visualize_clip_similarity_matrix(
    dataset=dataset,
    text_features=text_simple,
    idx_to_class=idx_to_class,
    model=model,
    device=device,
    num_images=8,
    num_classes=8,
    save_path="clip_similarity_matrix_simple.png"
)
visualize_clip_similarity_matrix(
    dataset=dataset,
    text_features=text_ensemble,
    idx_to_class=idx_to_class,
    model=model,
    device=device,
    num_images=8,
    num_classes=8,
    save_path="clip_similarity_matrix_ensemble.png"
)

def visualize_caltech101_clip_matrices(
    model,
    preprocess,
    dataset,
    idx_to_class,
    text_templates_dict,
    device='cuda',
    num_images=8,
    save_dir="./clip_similarity_results",
    vmin=0.1,
    vmax=0.3,
):
    os.makedirs(save_dir, exist_ok=True)
    sample_indices = random.sample(range(len(dataset)), num_images)
    sampled_images, sampled_labels = [], []

    for idx in sample_indices:
        img, label = dataset[idx]
        sampled_images.append(transforms.ToPILImage()(img))
        sampled_labels.append(idx_to_class[label])

    print(f"Sampled {num_images} images from Caltech101.")

    for template_name, template_list in text_templates_dict.items():
        print(f"\n Generating CLIP similarity matrix for template group: {template_name}")

        text_list = [template_list[0].format(label) for label in sampled_labels]

        with torch.no_grad():
            img_inputs = torch.stack([preprocess(img) for img in sampled_images]).to(device)
            img_features = model.encode_image(img_inputs)
            img_features /= img_features.norm(dim=-1, keepdim=True)

            text_features = []
            for label in sampled_labels:
                texts = [tmpl.format(label) for tmpl in template_list]
                tokens = clip.tokenize(texts).to(device)
                emb = model.encode_text(tokens)
                emb /= emb.norm(dim=-1, keepdim=True)
                text_features.append(emb.mean(0))
            text_features = torch.stack(text_features)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarity = text_features.cpu().numpy() @ img_features.cpu().numpy().T

        plt.figure(figsize=(15, 10))
        sns.heatmap(
            similarity,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            cbar=False,
            xticklabels=False,
            yticklabels=text_list,
        )
        for i, img in enumerate(sampled_images):
            plt.imshow(img, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")

        for side in ["left", "top", "right", "bottom"]:
            plt.gca().spines[side].set_visible(False)
        plt.xticks([])
        plt.yticks(fontsize=12)
        plt.xlim([-0.5, num_images - 0.5])
        plt.ylim([num_images + 0.5, -2])
        plt.title(f"Cosine similarity between text and image features\nTemplate: {template_name}", size=18)
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"caltech101_similarity_{template_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Saved: {save_path}")

text_templates_dict = {
    "Simple": simple_templates,     
    "Ensemble": templates,           
    "Optimal": optimal_templates    
}

visualize_caltech101_clip_matrices(
    model=model,                 
    preprocess=preprocess,      
    dataset=dataset,          
    idx_to_class=idx_to_class,   
    text_templates_dict=text_templates_dict,
    device=device,            
    num_images=8,               
    save_dir="./clip_similarity_results" 
)