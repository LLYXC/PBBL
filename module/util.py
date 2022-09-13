import torch.nn as nn
import torchxrayvision as xrv

def get_model(model_tag, num_classes, ldd=False, pretrained_xrv=True, mlp_classifier=False):
    if model_tag == "DenseNet121":
        if pretrained_xrv:
            model = xrv.models.DenseNet(weights="densenet121-res224-all")
        else:
            model = xrv.models.DenseNet()
        if mlp_classifier:

            model.classifier = nn.Sequential(nn.Linear(1024, 512),
                                             nn.ReLU(),
                                             nn.Linear(512, 512),
                                             nn.ReLU(),
                                             nn.Linear(512, num_classes))

            for param in model.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True
        else:
            model.classifier = nn.Linear(1024, num_classes)

        model.upsample = None

        #model.classifier.weight.requires_grad = True
        #model.classifier.bias.requires_grad = True
        return model

    else:
        raise NotImplementedError
