import torch
from mmaction.apis import inference_recognizer, init_recognizer
from torch import nn 
from mmcv import Config
from mmaction.apis import single_gpu_test
import torch.nn.functional as F
from mmcv.parallel import MMDataParallel
from mmaction.apis import inference_recognizer, init_recognizer
from mmaction.models import build_model

class SimpleRecog(nn.Module):
    def __init__(
        self,
        config_file : str = "",
        checkpoint_file : str = "",
        device : str = ""
    ):
        super().__init__()
        self.cfg = Config.fromfile(config_file)
        self.cfg.model.cls_head.num_classes = 2
        self.cfg.evaluation.save_best='auto'

        self.device = torch.device(device)
        self.model = build_model(self.cfg.model, train_cfg=self.cfg.get('train_cfg'), test_cfg=self.cfg.get('test_cfg'))
        self.model.eval()
    def forward(self, x): 
        # batch_size, channels, width, height = x.size()

        # (batch, 1, width, height) -> (batch, 1*width*height)
        # x = x.view(batch_size, -1)
        # pass the input through the TSN model
        # print(x['imgs'])
        # with torch.no_grad():
        #     result = inference_recognizer(self.model, x)

        # return the predicted class probabilities
        # return F.softmax(torch.tensor(result), dim=-1)
        # print(x) #1,250,3,224,224
        # print(type(x))
        # x = x.view(250,3,224,224)
        # X 4 dimentions B * C * H * W
        
        
        # print(x.shape)
        # x.unsqueeze(0)
        # return self.model.forward(x, return_loss = False)
        #print(x.shape) #batch, chunk, channel, H,W)
        return self.model.forward(x, return_loss = False)
        
        


if __name__ == "__main__":
    _ = SimpleRecog()
