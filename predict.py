
from PIL import Image

from torchvision import transforms

from train.model import ShapeCNN_top
import torch
model = ShapeCNN_top()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load("models/shape_CNN_10type_irregular.pth", map_location=device, weights_only=True))

model.eval()
classes = ['circle', 'diamond', 'hexagon', 'oval', 'pentagon', 'rectangle', 'square', 'star', 'trapezoid', 'triangle']

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

image = Image.open("image.png").convert("RGB")
image = transform(image)
image = image.unsqueeze(0)

model = model.to(device)
image = image.to(device)

with torch.no_grad():
    outputs = model(image)
    probabilities = torch.softmax(outputs, dim=1)
    predicted = outputs.argmax(dim=1).item()

print(classes[predicted], probabilities[0, predicted])
