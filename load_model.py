
import torch

from core.model import Deblur
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

unloader = transforms.ToPILImage()

def imshow(tensor, title=None):
	image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
	image = image.squeeze(0)  # remove the fake batch dimension
	image = unloader(image)
	plt.imshow(image)
	if title is not None:
		plt.title(title)
	plt.pause(100)  # pause a bit so that plots are updated

model = Deblur()
model.load_state_dict(torch.load('.\\models\\best_network.pth'))
img = Image.open('.\\data\\0_IPHONE-SE_S.JPG').convert('RGB')
trans = transforms.Compose([
    transforms.CenterCrop(256),
    transforms.Resize(128),
    transforms.GaussianBlur(21),
    transforms.ToTensor()
])
model.eval()
img = trans(img)
imshow(img)
out = model(img)
imshow(out)
