import torchvision.transforms as transforms
from PIL import Image
from AGNET import *

if __name__ == '__main__':

    net = GNet()
    model_filename = "model_best.pkl"
    net.load_state_dict(torch.load(model_filename))

    img_path = "/Users/ankitakalra/Documents/unshrouded_data/2208385.jpg"
    image = Image.open(img_path)
    image = image.resize((224, 224))
    transform = transforms.ToTensor()
    image = transform(image)
    image = image.resize_(1,3,224,224)

    result = net(image).cpu()

    to_pil_image = transforms.ToPILImage()

    res = to_pil_image(result[0])
    res.show()
    imagelog = torch.log(image+1)

    deshadow = torch.add(imagelog, result)
    deshadow = torch.exp(deshadow)-1
    deshadowed_image = to_pil_image(deshadow[0])
    deshadowed_image.show()
