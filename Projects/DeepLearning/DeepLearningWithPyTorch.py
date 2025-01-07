import torch
import urllib.request
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchvision import models
from torchvision.models import AlexNet_Weights

# print(torch.__version__)  # 2.5.0+cu118
# print(torch.cuda.is_available())  # True

url = "https://raw.githubusercontent.com/pytorch/hub/master/images/dog.jpg"
file_name = "dog.jpg"
urllib.request.urlretrieve(url, file_name)
img = Image.open("dog.jpg")
plt.imshow(img)
preprocess = (transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]))
img_tensor = preprocess(img)
# print(img_tensor.shape)  # torch.Size([3, 224, 224])
batch = img_tensor.unsqueeze(0)
# print(batch.shape)  # torch.Size([1, 3, 224, 224])
model = models.alexnet(weights=AlexNet_Weights.DEFAULT)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.eval()
model.to(device)
y = model(batch.to(device))
# print(y.shape)  # torch.Size([1, 1000])
y_max, index = torch.max(y, 1)
# print(index, y_max)  # tensor([258], device='cuda:0') tensor([16.8252], device='cuda:0', grad_fn=<MaxBackward0>)
url = "http://pytorch.tips/imagenet-labels/"
fname = "imagenet_class_labels.txt"
urllib.request.urlretrieve(url, fname)
with open("imagenet_class_labels.txt") as f:
    classes = [line.strip() for line in f.readlines()]
# print(classes[258])  # 258: 'Samoyed, Samoyede',
prop = torch.nn.functional.softmax(y, dim=1)[0] * 100
# print(classes[index[0]], prop[index[0]].item())  # 258: 'Samoyed, Samoyede', 72.4476547241211
_, indeces = torch.sort(y, descending=True)
for idx in indeces[0][:5]:
    print(classes[idx], prop[idx].item())  # ilk 5 tahmini yazdırıyor;
# 258: 'Samoyed, Samoyede', 72.4476547241211
# 104: 'wallaby, brush kangaroo', 13.937847137451172
# 259: 'Pomeranian', 5.874993324279785
# 332: 'Angora, Angora rabbit', 2.2829768657684326
# 279: 'Arctic fox, white fox, Alopex lagopus', 1.2450159788131714
