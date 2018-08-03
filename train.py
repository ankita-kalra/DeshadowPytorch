from shadow_dataset_loader import *
import torch.utils.data as data
from SNet import *
from AGNET import *
import os
from torch.autograd import Variable

# hyper parameters
BATCH_SIZE = 1
BASE_LR = 1e-3
G_LR = 1e-5
NUM_EPOCHS = 5
MODEL_SAVE_RATE=2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

normalization_mean = torch.tensor([0.485, 0.456, 0.406])
normalization_std = torch.Tensor([0.229, 0.224, 0.225])


def make_label(image, label):
    image_log = torch.log(image + 1).to(device)
    label_log = torch.log(label + 1).to(device)
    result = torch.abs(torch.add(image_log, -1, label_log)).to(device)
    result *= 3
    return result


def normalize(input, mean, std):
    m = torch.tensor(mean).view(-1, 1, 1).to(device)
    s = torch.tensor(std).view(-1, 1, 1).to(device)
    output = (input - m) / s
    return output



def local_s_train():

    #Initilization
    preprocess = transforms.Compose([
        transforms.Resize(120),
        transforms.ToTensor(),
    ])

    data_dir = os.path.join('/Users/ankitakalra/', 'Documents/GitHub/DeshadownetPytorch/data/')
    train_folder = CustomShadowPairDataset(os.path.join(data_dir, 'original_image/'), preprocess)
    print(data_dir,train_folder)
    train_loader = torch.utils.data.DataLoader(train_folder,
                                               batch_size=1)

    net = SNet()
    net = net.to(device)

    gparam = list(map(id, net.features.parameters()))
    base_param = filter(lambda p: id(p) not in gparam, net.parameters())

    optimizer = torch.optim.SGD([
        {'params': base_param},
        {'params': net.features.parameters(), 'lr': G_LR}], lr = BASE_LR, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.MSELoss()

    for epoch in range(NUM_EPOCHS):
        for i, data in enumerate(train_loader, 0):
            image, label = data
            image = image.to(device)
            label = label.to(device)
            target = make_label(image, label)

            norm_image = normalize(image, normalization_mean, normalization_std)
            prediction = net(norm_image)
            loss = criterion(prediction, target)

            print('Epoch: %d | iter: %d | train loss: %.10f' % (epoch, i, float(loss)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            model_name = os.path.join('model/model_%d.pkl' % epoch)
            torch.save(net.state_dict(), model_name)

def local_a_train():

    #Initilization
    preprocess = transforms.Compose([
        transforms.Resize(120),
        transforms.ToTensor(),
    ])

    data_dir = os.path.join('/Users/ankitakalra/', 'Documents/GitHub/DeshadownetPytorch/data/')
    train_folder = CustomShadowPairDataset(os.path.join(data_dir, 'original_image/'), preprocess)
    print(data_dir,train_folder)
    train_loader = torch.utils.data.DataLoader(train_folder,
                                               batch_size=1)

    net = ANet()
    net = net.to(device)

    gparam = list(map(id, net.features.parameters()))
    base_param = filter(lambda p: id(p) not in gparam, net.parameters())

    optimizer = torch.optim.SGD([
        {'params': base_param},
        {'params': net.features.parameters(), 'lr': G_LR}], lr = BASE_LR, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.MSELoss()

    for epoch in range(NUM_EPOCHS):
        for i, data in enumerate(train_loader, 0):
            image, label = data
            image = image.to(device)
            label = label.to(device)
            target = make_label(image, label)

            norm_image = normalize(image, normalization_mean, normalization_std)
            prediction = net(norm_image)
            loss = criterion(prediction, target)

            print('Epoch: %d | iter: %d | train loss: %.10f' % (epoch, i, float(loss)))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            model_name = os.path.join('model/model_%d.pkl' % epoch)
            torch.save(net.state_dict(), model_name)



def gpu_train():

    # Initilization
    preprocess = transforms.Compose([
        transforms.Resize(120),
        transforms.ToTensor(),
    ])
    use_gpu = torch.cuda.is_available()
    data_dir = os.path.join(os.path.expanduser("~"), 'Documents/Github/DeshadownetPytorch/data/')
    train_folder = CustomShadowPairDataset(os.path.join(data_dir, 'original_images'), preprocess)
    train_loader = torch.utils.data.DataLoader(train_folder,
                                               batch_size=1, shuffle=True,
                                               num_workers=2, pin_memory=False)
    normalization_mean = torch.tensor([0.485, 0.456, 0.406]).cuda(0)
    normalization_std = torch.Tensor([0.229, 0.224, 0.225]).cuda(0)

    net = SNet()
    if use_gpu:
        net.cuda()

    gparam = list(map(id, net.module.features.parameters()))
    base_param = filter(lambda p: id(p) not in gparam, net.parameters())
    optimizer = torch.optim.SGD([
        {'params': base_param},
        {'params': net.module.features.parameters(), 'lr': G_LR}], lr=BASE_LR, momentum=0.9, weight_decay=5e-4)

    criterion = torch.nn.MSELoss()
    optimizer = nn.DataParallel(optimizer, device_ids=0)

    for epoch in range(NUM_EPOCHS):
        for i, data in enumerate(train_loader, 0):
            image, label = data
            target = make_label(image, label)
            norm_image = normalize(image, normalization_mean, normalization_std)

            img = Variable(norm_image, requires_grad=True).cuda(0)
            tar = Variable(target).cuda(0)

            prediction = net(img)
            loss = criterion(prediction, tar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.module.step()

            print('Epoch: %d | iter: %d | train loss: %.10f' % (epoch, i, float(loss)))

        if epoch % 100 == 0:
            model_name = os.path.join("./model/model_%d.pkl" % epoch)
            torch.save(net, model_name)

local_a_train()
local_s_train()