import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from facenet_pytorch import InceptionResnetV1
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from dataset import get_dataloaders
import argparse

# Designer-grade aesthetics
sns.set_theme(style="darkgrid")
plt.rcParams['font.family'] = 'sans-serif'

def get_embeddings(model, dataloader, device):
    model.eval()
    embeddings = []
    labels = []
    
    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            # Remove classification head to get raw 512d embeddings
            embeds = model(images)
            embeddings.append(embeds.cpu().numpy())
            labels.append(targets.numpy())
            
    return np.concatenate(embeddings), np.concatenate(labels)

def generate_pairs(embeddings, labels):
    same_pairs, diff_pairs = [], []
    n = len(labels)
    
    for i in range(n):
        for j in range(i + 1, n):
            # Cosine Similarity
            sim = np.dot(embeddings[i], embeddings[j]) / (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
            if labels[i] == labels[j]:
                same_pairs.append(sim)
            else:
                diff_pairs.append(sim)
                
    return np.array(same_pairs), np.array(diff_pairs)

def plot_metrics(same_pairs, diff_pairs, prefix):
    # Labels for ROC
    y_true = np.concatenate([np.ones(len(same_pairs)), np.zeros(len(diff_pairs))])
    y_scores = np.concatenate([same_pairs, diff_pairs])
    
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    # 1. Plot Similarity Distribution
    plt.figure(figsize=(10, 6))
    sns.kdeplot(same_pairs, fill=True, color="cyan", label="Same Identity")
    sns.kdeplot(diff_pairs, fill=True, color="magenta", label="Different Identity")
    plt.title(f"{prefix} Model: Cosine Similarity Distribution", fontsize=16, fontweight='bold')
    plt.xlabel("Cosine Similarity", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend()
    plt.savefig(f"{prefix}_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Plot ROC Curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='cyan', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.title(f"{prefix} Model: ROC Curve", fontsize=16, fontweight='bold')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(f"{prefix}_roc.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Metrics saved for {prefix}. AUC: {roc_auc:.3f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='pretrained', help="Path to fine-tuned .pth file, or 'pretrained'")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, eval_loader, _ = get_dataloaders('dataset/train', 'dataset/eval')

    if args.weights == 'pretrained':
        model = InceptionResnetV1(pretrained='vggface2').to(device)
        prefix = "Baseline"
    else:
        # Load custom weights
        model = InceptionResnetV1(pretrained=None, classify=False).to(device)
        # Handle strict state dict loading for custom architectures
        state_dict = torch.load(args.weights)
        # Filter out the classification layer we added during training
        state_dict = {k: v for k, v in state_dict.items() if 'logits' not in k}
        model.load_state_dict(state_dict, strict=False)
        prefix = "Finetuned"

    print(f"Extracting embeddings using {prefix} weights...")
    embeddings, labels = get_embeddings(model, eval_loader, device)
    
    print("Computing pairwise similarities...")
    same, diff = generate_pairs(embeddings, labels)
    
    print("Generating aesthetic plots...")
    plot_metrics(same, diff, prefix)

if __name__ == "__main__":
    main()