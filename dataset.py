from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(train_dir, eval_dir, batch_size=32):
    # Designer Note: We add blur and jitter to simulate low quality during training
    train_transforms = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # Eval stays pristine, purely deterministic
    eval_transforms = transforms.Compose([
        transforms.Resize((160, 160)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    eval_dataset = datasets.ImageFolder(eval_dir, transform=eval_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, eval_loader, train_dataset.classes