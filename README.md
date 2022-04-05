# DeepLearning


In this work, we applied the resnet18 as the prototype and changed a part of the architecture of this model. Because the parameter quantity is required to be less than 5M, we have 4 residual layers and each residual layer has only one residual block. In each residual block, we have the structure like conv→batchnorm→relu→conv→batchnorm→(maybe shortcut). We also use some data augmentation strategies like picture crop and flip. The CrossEntropy is chosen as a loss function. Adam is the optimizer in this model. Then we use CIFAR-10 as the dataset to train the model by running 50 epochs. We finally got the test accuracy 90.84%.

To test the accuracy of the trained model, follow these steps:
1. download the codes from this repository, which contains the ResNet model file [**project1_model.py**] and the file containing the trained weights of the ResNet model in project1_model.py, [**project1_model.pt**].
```
git clone https://github.com/mirevas/DeepLearning.git
```
 2. move the test dataset to the DeepLearning directory, for example,
```
mv self_eval.py ./DeepLearning
```
 3. run the test code by using python and it will show the accuracy in the terminal.
```
python self_eval.py
```

To reproduce the final results by running the same training process( Noticed: The results may be slightly different.), follow these steps:
1. Use pip to install pytorch, torchvision, matplotlib and torchsummary
2. Use colab to run project1_model.ipynb file.
3. Use colab to run train.ipynb file to train the model. You are supposed to upload project1_model.py file manually in the second code section. 
4. Train loss, test loss, training accuracy and test accuracy plots, the number of parameters and the message of the ResNet model will be showed after the training process. 
5. You can download the file [project1_model.pt] which contains the trained weights for the test process.

