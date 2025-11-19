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

# ===============================
#  Evaluate CLIP zero-shot accuracy
# ===============================
loader = DataLoader(dataset, batch_size=32, shuffle=False)

acc_simple = zero_shot_accuracy(loader, text_simple)
acc_ensemble = zero_shot_accuracy(loader, text_ensemble)

print(f'Simple template accuracy: {acc_simple * 100:.2f}%')
print(f'Ensemble template accuracy: {acc_ensemble * 100:.2f}%')


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