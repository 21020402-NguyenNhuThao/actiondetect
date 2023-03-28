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
        batch_size, channels, width, height = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        x = x.view(batch_size, -1)

        return self.model(x)


if __name__ == "__main__":
    _ = SimpleRecog()
