import torch
from datasets import load_dataset
import clip
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import random
import os

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ===============================
#  Stanford Cars Dataset Wrapper
# ===============================
class StanfordCarsDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.ds = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        image = self.ds[idx]["image"].convert("RGB")
        label = self.ds[idx]["label"]
        if self.transform:
            image = self.transform(image)
        return image, label


# ===============================
#  Load dataset from HuggingFace
# ===============================
print("Loading Stanford Cars dataset from HuggingFace...")
raw_train = load_dataset("tanganke/stanford_cars", split="train")
raw_test = load_dataset("tanganke/stanford_cars", split="test")

classes = raw_train.features["label"].names
print(f"Number of classes: {len(classes)}")

idx_to_class = {i: c for i, c in enumerate(classes)}

# ===============================
#  Transform
# ===============================
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = StanfordCarsDataset(raw_train, transform)
test_dataset = StanfordCarsDataset(raw_test, transform)

print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

ensemble_templates = [
    'a photo of a {}.',
    'a photo of the {}.',
    'a photo of my {}.',
    'i love my {}!',
    'a photo of my dirty {}.',
    'a photo of my clean {}.',
    'a photo of my new {}.',
    'a photo of my old {}.',
]

simple_templates = ['a photo of a {}.']

# ===============================
#  Load CLIP Model
# ===============================
print("Loading CLIP ViT-B/16 model...")
model, preprocess = clip.load("ViT-B/16", device=device)
model.eval()

# ===============================
#  Build Text Features
# ===============================
def build_text_features(classes, templates):
    all_features = []
    with torch.no_grad():
        for c in tqdm(classes, desc="Encoding class prompts"):
            texts = [t.format(c) for t in templates]
            tokens = clip.tokenize(texts).to(device)
            embeddings = model.encode_text(tokens)
            embeddings /= embeddings.norm(dim=-1, keepdim=True)
            mean_emb = embeddings.mean(0)
            mean_emb /= mean_emb.norm()
            all_features.append(mean_emb)
    text_features = torch.stack(all_features)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    return text_features

print('Building simple prompt features...')
text_simple = build_text_features(classes, simple_templates)

print('Building ensemble prompt features...')
text_ensemble = build_text_features(classes, ensemble_templates)

# ===============================
#  Zero-shot accuracy evaluation
# ===============================
def zero_shot_accuracy(dataloader, text_features):
    correct, total = 0, 0
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc="Evaluating", ncols=80):
            images = images.to(device)
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = image_features @ text_features.T
            preds = similarity.argmax(dim=-1).cpu()
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / total

loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
acc = zero_shot_accuracy(loader, text_features)
print(f"Zero-shot accuracy on Stanford Cars: {acc * 100:.2f}%")

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
visualize_predictions(test_dataset, text_ensemble, idx_to_class,
                      num_examples=8,
                      save_path="clip_ViT-B_16_ensemble.png",
                      sample_indices=sample_indices)
visualize_predictions(test_dataset, text_simple, idx_to_class,
                      num_examples=8,
                      save_path="clip_ViT-B_16_simple.png",
                      sample_indices=sample_indices)