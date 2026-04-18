import copy
import os
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torchvision import transforms, models

# -----------------------------
# Device configuration
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# Image loading and preprocessing
# -----------------------------
imsize = 512 if torch.cuda.is_available() else 256

loader = transforms.Compose([
    transforms.Resize((imsize, imsize)),
    transforms.ToTensor()
])

unloader = transforms.ToPILImage()


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def save_image(tensor, output_path):
    image = tensor.cpu().clone().squeeze(0)
    image = unloader(image.clamp(0, 1))
    image.save(output_path)


def imshow(tensor, title=None):
    image = tensor.cpu().clone().squeeze(0)
    image = unloader(image.clamp(0, 1))
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis("off")


# -----------------------------
# Content Loss
# -----------------------------
class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()

    def forward(self, input_tensor):
        self.loss = F.mse_loss(input_tensor, self.target)
        return input_tensor


# -----------------------------
# Style Loss and Gram Matrix
# -----------------------------
def gram_matrix(input_tensor):
    a, b, c, d = input_tensor.size()
    features = input_tensor.view(a * b, c * d)
    G = torch.mm(features, features.t())
    return G.div(a * b * c * d)


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input_tensor):
        G = gram_matrix(input_tensor)
        self.loss = F.mse_loss(G, self.target)
        return input_tensor


# -----------------------------
# Normalization
# -----------------------------
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1).to(device)
        self.std = torch.tensor(std).view(-1, 1, 1).to(device)

    def forward(self, img):
        return (img - self.mean) / self.std


# -----------------------------
# Load pretrained VGG19
# -----------------------------
cnn = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features.to(device).eval()

cnn_normalization_mean = [0.485, 0.456, 0.406]
cnn_normalization_std = [0.229, 0.224, 0.225]

content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


# -----------------------------
# Build model and losses
# -----------------------------
def get_style_model_and_losses(
    cnn,
    normalization_mean,
    normalization_std,
    style_img,
    content_img,
    content_layers=content_layers_default,
    style_layers=style_layers_default
):
    cnn = copy.deepcopy(cnn)

    normalization = Normalization(normalization_mean, normalization_std).to(device)

    content_losses = []
    style_losses = []

    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:i + 1]

    return model, style_losses, content_losses


# -----------------------------
# Optimizer
# -----------------------------
def get_input_optimizer(input_img):
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    return optimizer


# -----------------------------
# Style transfer function
# -----------------------------
def run_style_transfer(
    cnn,
    normalization_mean,
    normalization_std,
    content_img,
    style_img,
    input_img,
    num_steps=300,
    style_weight=1000000,
    content_weight=1
):
    print("Building the style transfer model...")
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn,
        normalization_mean,
        normalization_std,
        style_img,
        content_img
    )

    optimizer = get_input_optimizer(input_img)

    print("Optimizing...")
    run = [0]
    while run[0] <= num_steps:

        def closure():
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)

            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print(f"Step {run[0]}:")
                print(f"Style Loss : {style_score.item():.4f}")
                print(f"Content Loss: {content_score.item():.4f}")
                print()

            return loss

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)
    return input_img


# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":
    content_path = "content_images/img1.jpg.jpg"
    style_path = "style_images/style_images1.jpg.jpg"
    output_path = "output_images/output1.jpg"

    if not os.path.exists(content_path):
        print("Content image not found:", content_path)
        exit()

    if not os.path.exists(style_path):
        print("Style image not found:", style_path)
        exit()

    os.makedirs("output_images", exist_ok=True)

    content_img = load_image(content_path)
    style_img = load_image(style_path)

    if content_img.size() != style_img.size():
        print("Resizing handled automatically by transforms.")

    input_img = content_img.clone()

    output = run_style_transfer(
        cnn,
        cnn_normalization_mean,
        cnn_normalization_std,
        content_img,
        style_img,
        input_img
    )

    save_image(output, output_path)
    print("Output image saved at:", output_path)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    imshow(content_img, "Content Image")

    plt.subplot(1, 3, 2)
    imshow(style_img, "Style Image")

    plt.subplot(1, 3, 3)
    imshow(output, "Output Image")

    plt.tight_layout()
    plt.show()