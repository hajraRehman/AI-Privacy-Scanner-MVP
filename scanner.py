# scanner.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load CIFAR-10 + Pretrained ResNet18
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
])

full_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

# Target model (victim)
model = models.resnet18(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, 10)
model = model.to(device)

# -----------------------------
# 2. Membership Inference Attack (MIA)
# -----------------------------
def train_shadow_model(shadow_train_loader, shadow_val_loader, epochs=5):
    shadow = models.resnet18(pretrained=False)
    shadow.fc = nn.Linear(shadow.fc.in_features, 10)
    shadow = shadow.to(device)
    opt = torch.optim.SGD(shadow.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        shadow.train()
        for x, y in shadow_train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = criterion(shadow(x), y)
            loss.backward()
            opt.step()
    return shadow

def get_posteriors(model, loader):
    model.eval()
    posteriors = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            prob = F.softmax(logits, dim=1)
            posteriors.append(prob.max(1)[0].cpu().numpy())
            labels.append(y.cpu().numpy())
    return np.concatenate(posteriors), np.concatenate(labels)

def mia_attack(target_model, train_loader, val_loader):
    # Split train into shadow_train and shadow_val
    shadow_train_size = len(train_loader.dataset) // 2
    shadow_train, shadow_val = random_split(train_loader.dataset, [shadow_train_size, len(train_loader.dataset) - shadow_train_size])
    shadow_train_loader = DataLoader(shadow_train, batch_size=128)
    shadow_val_loader = DataLoader(shadow_val, batch_size=128)

    shadow_model = train_shadow_model(shadow_train_loader, shadow_val_loader)
    
    # Get posteriors
    train_post, _ = get_posteriors(shadow_model, shadow_train_loader)
    val_post, _ = get_posteriors(shadow_model, shadow_val_loader)

    # Labels: 1 = member, 0 = non-member
    X = np.concatenate([train_post, val_post])
    y = np.concatenate([np.ones(len(train_post)), np.zeros(len(val_post))])

    # Train attack classifier
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression()
    clf.fit(X.reshape(-1, 1), y)

    # Attack target model
    target_train_post, _ = get_posteriors(target_model, train_loader)
    target_val_post, _ = get_posteriors(target_model, val_loader)

    X_target = np.concatenate([target_train_post, target_val_post])
    y_true = np.concatenate([np.ones(len(target_train_post)), np.zeros(len(target_val_post))])
    y_pred = clf.predict(X_target.reshape(-1, 1))
    y_prob = clf.predict_proba(X_target.reshape(-1, 1))[:, 1]

    acc = accuracy_score(y_true, y_pred)
    return acc, np.mean(y_prob[y_true == 1]), np.mean(y_prob[y_true == 0])

# -----------------------------
# 3. Adversarial Attacks: FGSM + PGD
# -----------------------------
def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    return torch.clamp(perturbed_image, 0, 1)

def pgd_attack(model, images, labels, eps=0.03, alpha=0.01, iters=40):
    original_images = images.clone().detach()
    for _ in range(iters):
        images.requires_grad = True
        outputs = model(images)
        loss = F.cross_entropy(outputs, labels)
        model.zero_grad()
        loss.backward()
        adv_images = images + alpha * images.grad.sign()
        eta = torch.clamp(adv_images - original_images, min=-eps, max=eps)
        images = torch.clamp(original_images + eta, min=0, max=1).detach_()
    return images

def test_adversarial(model, loader, eps=0.03):
    model.eval()
    clean_correct = 0
    adv_correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        total += y.size(0)

        # Clean
        out = model(x)
        clean_correct += (out.argmax(1) == y).sum().item()

        # PGD
        x_adv = pgd_attack(model, x, y, eps=eps)
        out_adv = model(x_adv)
        adv_correct += (out_adv.argmax(1) == y).sum().item()

    clean_acc = clean_correct / total
    robust_acc = adv_correct / total
    return clean_acc, robust_acc

# -----------------------------
# 4. Run Full Scan
# -----------------------------
def run_full_scan(model_path):
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # MIA
    mia_acc, member_conf, nonmember_conf = mia_attack(model, train_loader, val_loader)

    # Adversarial
    clean_acc, robust_acc = test_adversarial(model, val_loader, eps=0.03)

    results = {
        "mia_accuracy": round(mia_acc * 100, 1),
        "member_confidence": round(member_conf * 100, 1),
        "nonmember_confidence": round(nonmember_conf * 100, 1),
        "clean_accuracy": round(clean_acc * 100, 1),
        "robust_accuracy": round(robust_acc * 100, 1),
    }
    return results, model