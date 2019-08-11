# Week Four Assignment
Here the task was to achieve 99.4% validation accuracy with less than 20k parameters.

## Code 1
In the first code block, the only condition we had was that the architecture of the model had to be a vanilla model. The only additions we could use were the 1x1 and Max Pooling.

Number of trainable parameters: 58,538

Training Accuracy: 99.81%

Validation Accuracy: 99.12%

## Code 2
In the second code file, the changes made in the model architecture are: Adding Batch Normalization Layers, Changing the number of channels and changing the batch size. These are the three improvements in the second code file.

Number of trainable parameters: 15,310

Training Accuracy: 99.56%

Validation Accuracy: 99.19%

## Code 3
In the third code file, the changes made are: Addition of Dropout, Changing the Optimizer from Adam to SGD and Changing the learning rate.

Number of trainable parameters: 11,530

Training Accuracy: 96.87%

Validation Accuracy: 98.17%

## Code 4
In the fourth code file, the changes made are: Addition of Dropout layer after each Batch Normalization Layer, Adding callbacks and changing the kernel size.

Number of trainable parameters: 11,516

Training Accuracy: 98.40%

Validation Accuracy: 99.40%