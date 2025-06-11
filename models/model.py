import torch.nn as nn

from torchvision.models import resnet50

class FaceKeypointModel(nn.Module):
    def __init__(self, pretrained=False, requires_grad=True):
        super(FaceKeypointModel, self).__init__()
        if pretrained:
            self.model = resnet50(weights='DEFAULT')
        else:
            self.model = resnet50(weights=None)

        if requires_grad:
            for param in self.model.parameters():
                param.requires_grad = True
            print('Training intermediate layer parameters...')
        else:
            for param in self.model.parameters():
                param.requires_grad = False
            print('Freezing intermediate layer parameters...')

        # change the final layer
        self.model.fc = nn.Linear(in_features=2048, out_features=136)

    def forward(self, x):
        out = self.model(x)
        return out

if __name__ == '__main__':
    model = FaceKeypointModel()
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")