from torchvision import transforms
from PIL import Image, ImageOps
from torch.autograd import Variable

def load(path):
    shape = (224,224)
    image = Image.open(path).convert("RGB")
    image = ImageOps.fit(image, shape, Image.ANTIALIAS)

    loader = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image = loader(image)
    image = image.unsqueeze(0)

    return Variable(image)
