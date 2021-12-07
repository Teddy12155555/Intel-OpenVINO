import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import alexnet, vgg16
import torchvision
from PIL import Image

def convolutionize(model, num_classes, input_size=(3, 40, 40)):
    '''Converts the classification layers of VGG & Alexnet to convolutions

    Input:
        model: torch.models
        num_classes: number of output classes
        input_size: size of input tensor to the model

    Returns:
        model: converted model with convolutions
    '''
    features = model.features
    classifier = model.classifier
    print(model.avgpool)

    # create a dummy input tensor and add a dim for batch-size
    x = torch.zeros(input_size).unsqueeze_(dim=0)

    # change the last layer output to the num_classes
    classifier[-1] = nn.Linear(in_features=classifier[-1].in_features,
                               out_features=num_classes)

    # pass the dummy input tensor through the features layer to compute the output size
    for layer in features:
        x = layer(x)

    conv_classifier = []
    for layer in classifier:
        if isinstance(layer, nn.Linear):
            # create a convolution equivalent of linear layer
            conv_layer = nn.Conv2d(in_channels=x.size(1),
                                   out_channels=layer.weight.size(0),
                                   kernel_size=(x.size(2), x.size(3)))

            # transfer the weights
            conv_layer.weight.data.view(-1).copy_(layer.weight.data.view(-1))
            conv_layer.bias.data.view(-1).copy_(layer.bias.data.view(-1))
            layer = conv_layer

        x = layer(x)
        conv_classifier.append(layer)

    # replace the model.classifier with newly created convolution layers
    model.classifier = nn.Sequential(*conv_classifier)

    return model

def visualize(model, input_size=(3, 40, 40)):
    '''Visualize the input size though the layers of the model'''
    x = torch.zeros(input_size).unsqueeze_(dim=0)
    print(x.size())
    for layer in list(model.features):
        x = layer(x)
        print(x.size())

class net(nn.Module):
    def __init__(self, model):
        super(net, self).__init__()
        self.features = nn.Sequential(
                    *list(model.features.children())
        )
    def forward(self, x):
        x = self.features(x)
        return x


if __name__ == "__main__":

    _vgg = torchvision.models.vgg16(pretrained=True)
    resN = torchvision.models.resnet152(pretrained=True)
    cust_vgg = net(_vgg)
    dummy_input = torch.randn(1, 3, 224, 224)
    #torch.onnx.export(cust_vgg, dummy_input, "custom_vgg16.onnx", verbose=True)
    print('---- Done ----')


    
    """
    transform = transforms.Compose([ 
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        ])
    cust_vgg.eval()
    img = Image.open("dog.jpg")
    imgt = transform(img)
    batch_t = torch.unsqueeze(imgt, 0)

    res = cust_vgg(batch_t)
    #print(res)
    """
    
    #torch.onnx.export(cust_vgg, dummy_input, "custom_vgg16.onnx", verbose=True)
    print('---- Done ----')
    #print(cust_vgg(dummy_input))
    visualize(cust_vgg)
    