import copy

import torch
import torch.nn as nn

import torchvision.models as models
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim

class VGG(object):
    def __init__(self, use_gpu):
        super(VGG, self).__init__()
        self.use_gpu = use_gpu
        self.models = []
        self.init_L_models()

    # CNN L-1 to L を得る(モデルの一部分だけ)
    def init_L_models(self):
        print('loading pre-trained vgg model...')
        cnn = models.vgg19(pretrained=True).features
        last_layer_names = ['relu_1', 'relu_3', 'relu_5', 'relu_9', 'relu_13']
        model = nn.Sequential()
        i = 1
        for layer in list(cnn): # pre-trained modelの読み込み
            if isinstance(layer, nn.Conv2d):
                name = "conv_" + str(i)
                model.add_module(name, layer)
            if isinstance(layer, nn.ReLU):
                name = "relu_" + str(i)
                model.add_module(name, layer)
                if name in last_layer_names:
                    self.models.append(model)
                    model = nn.Sequential() # 新しいmodelを作成
                i += 1
            if isinstance(layer, nn.MaxPool2d):
                name = "pool_" + str(i)
                model.add_module(name, nn.AvgPool2d((2,2)))
        print('model load finished')

        if self.use_gpu:
            self.models = []
            for model in self.models[:]:
                self.models.append(model.cuda())

    def get_features(self, input_image):
        out = input_image
        features = []

        if self.use_gpu:
            input_image = input_image.cuda()

        for model in self.models:
            out = model(out)
            features.append(out)

        features = [feature.data.squeeze().cpu().numpy().transpose(1,2,0) for feature in features]

        return features

    def deconv(self, feature, iters, L, optim_name):
        channel_num = 64 * 2**(L-1)
        noise = torch.randn(1, channel_num, feature.shape[0]*2, feature.shape[1]*2).float()

        feature = feature.transpose(2,0,1)
        feature = torch.from_numpy(feature).float()

        if self.use_gpu:
            noise = noise.cuda()
            feature = feature.cuda()

        noise = Variable(noise,requires_grad=True)
        feature = Variable(feature)

        if optim_name == 'LBFGS':
            optimizer = optim.LBFGS([noise],lr=1, max_iter=20, tolerance_grad=1e-4, history_size=4)
        elif optim_name == 'Adam':
            optimizer = optim.Adam([noise],lr=1)

        criterion = nn.MSELoss()
        for i in range(1,iters):
            def closure():
                optimizer.zero_grad()
                output = self.models[L](noise)
                loss = criterion(output, feature)
                loss.backward()
                return loss

            if optim_name == 'LBFGS':
                optimizer.step(closure)
            elif optim_name == 'Adam':
                optimizer.zero_grad()
                output = self.models[L](noise)
                loss = criterion(output, feature)
                loss.backward()
                optimizer.step()

            noise.data.clamp_(0., 1.)
            print(i, loss.cpu().data.numpy())

        noise = noise.cpu().data.squeeze().numpy()
        return noise.transpose(1,2,0)
