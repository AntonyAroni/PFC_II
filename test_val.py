import json
import urllib.request

# Descargar el índice de clases de ImageNet
url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
imagenet_classes = []
with urllib.request.urlopen(url) as f:
    imagenet_classes = [line.strip().decode("utf-8") for line in f]

print("Predicción:", imagenet_classes[281])  # <- reemplaza con tu índice
