import torch
from mmaction.apis import init_recognizer
from torch import nn 


class SimpleRecog(nn.Module):
    def __init__(
        self,
        config_file : str = "",
        checkpoint_file : str = "",
        device : str = "cpu"
    ):
        super().__init__()

        self.device = torch.device(device)
        self.model = init_recognizer(config_file, checkpoint_file, device = device)

    def forward(self, x):
        # batch_size, channels, width, height = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        # x = x.view(batch_size, -1)

        return self.model(x)


if __name__ == "__main__":
    model = SimpleRecog("/home/nnthao/ActionRecog/actiondetect/mmaction2/configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x5-100e_kinetics400-rgb.py",
                        'src/models/checkpoints/tsn_r50_video_1x1x8_100e_kinetics400_rgb_20200702-568cde33.pth',
                        'cpu')
    x = torch.randn(1,1,3,224,224)
    device = torch.device('cpu')
    x = x.to( torch.device(device))
# print(x.shape)
    output = model.forward(x)
    print(output)
    # _ = SimpleRecog()
