# UCL-for-IDS

## File Structure

    .
    ├── AutonomousDCN           # ADCN baseline - downloaded from their github
    ├── dataloader              # Dataloader
    └── README.md



## Dataloader tutorial

### MNIST Datasets

1. Load in dataset through torch vision

```
from torchvision import datasets, transforms

transform = transforms.ToTensor()

# load the training and test datasets
labeledData   = datasets.MNIST(root='data', train=False,download=True, transform=transform)
unlabeledData = datasets.MNIST(root='data', train=True, download=True, transform=transform)

dataStream = mnistLoader(labeledData, unlabeledData, nEachClassSamples = 500)
```


2. Select dataset task

For PermuteMNIST:

`dataStream.createTask(nTask = 5, taskList = [], taskType = 1)`

For RotateMNIST:

`dataStream.createTask(nTask = 4, taskList = [[0,30],[31,60],[61,90],[91,120]], taskType = 2)`

For SplitMNIST:

`dataStream.createTask(nTask = 5, taskList = [[0,1],[2,3],[4,5],[6,7],[8,9]], taskType = 3)`

3. TODO: CREATE TASK IMBALANCE FOR IDS SCENARIO.