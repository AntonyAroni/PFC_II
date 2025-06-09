import torch
from timm.models import create_model
from models.modules.mobileone import reparameterize_model
from PIL import Image
from torchvision import transforms

# Cargar modelo
model = create_model("fastvit_t8", pretrained=False)

# Ruta del checkpoint (ajústala si está en otra carpeta)
checkpoint = torch.load("fastvit_t8.pth.tar", map_location="cpu")
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# Reparametrizar
model_inf = reparameterize_model(model)

# Imagen de prueba
img = Image.open("test.jpg").convert("RGB")  # asegúrate de tener esta imagen
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])
input_tensor = transform(img).unsqueeze(0)

# Inferencia
with torch.no_grad():
    logits = model_inf(input_tensor)
    pred = logits.argmax(dim=1).item()

print("Predicción:", pred)
