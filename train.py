import torch
import torch.nn as nn
import torch.optim as optim
from facenet_pytorch import InceptionResnetV1
from pathlib import Path
from tqdm import tqdm
from dataset import get_dataloaders

def train_model(epochs=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Engine starting on: {device}")

    train_loader, _, classes = get_dataloaders('dataset/train', 'dataset/eval', batch_size=16)
    num_classes = len(classes)

    # Load pre-trained FaceNet (vggface2 weights)
    # We set classify=True to get a classification head, replacing it with our class count
    model = InceptionResnetV1(pretrained='vggface2', classify=True, num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            loop.set_postfix(loss=loss.item())

    # Save the fine-tuned checkpoint
    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "models/finetuned_facenet.pth")
    print("Fine-tuning complete. Model saved to models/finetuned_facenet.pth")

if __name__ == "__main__":
    train_model()