# DeepLearning

In this work, we applied the resnet18 as the prototype and to change a part of the architecture of this model. Because the parameter quantity is required to be less than 5M. We have 4 residual layers and each residual layer has only one residual block. In each residual block, we have the structure like conv→batchnorm→relu→conv→batchnorm→(maybe shortcut). We also use some data augmentation strategies like picture crop and flip. The CrossEntropy is chosen as a loss function. Adam is the optimizer in this model. Then we use CIFAR-10 as the dataset to train the model by running 50 epochs. We finally got the test accuracy 90.84%.
       
Before training the model, please use following code to process image data.

```
## read model file
import torch
from project1_model import project1_model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = project1_model().to(device)
model_path = './project1_model.pt'
model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        
