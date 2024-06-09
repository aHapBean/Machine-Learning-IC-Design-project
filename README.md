# CS Machine Learning IC Design project

Authors: [aHapBean](https://github.com/aHapBean), [zzzhr97](https://github.com/zzzhr97), [XiaoyangLiu39](https://github.com/XiaoyangLiu39)

## Environment Setup

### Step 1: Install `abc_py`
First, install the `abc_py` package by following the instructions provided in the [abc_py GitHub repository](https://github.com/krzhu/abc_py).

### Step 2: Install PyTorch
Install PyTorch. We use `torch==1.13.1`, but other versions of PyTorch may also be compatible.

### Step 3: Install PyTorch Geometric Dependencies
Before installing `torch_geometric`, you need to install its dependencies. Follow the instructions on the [PyTorch Geometric website](https://pytorch-geometric.com/whl/).

For `torch==1.13.1`, you can install the following versions of the dependencies:

- `torch-cluster`: `1.6.1+pt113cu116`
- `torch-scatter`: `2.1.1+pt113cu116`
- `torch-sparse`: `0.6.15+pt113cu116`

### Step 4: Install `torch_geometric`
After installing the dependencies, you can install `torch_geometric`. We use `torch_geometric==2.3.1`.

### Step 5: Install Remaining Dependencies
Finally, install the remaining dependencies by running the following command:

```bash
pip install -r requirements.txt
```



## Data preprocessing
需要把test数据中的 mem_ctrl.aig 改成 memctrl.aig.

## TODO

The following content will be added soon.
- Add training guidance
- Add files introduction
- Format the files in the project

## task 1

Model: 
> - GCN (GCN)
> - GAT (PureGAT)
> - GCN + GAT (EnhancedGCN)
> - Deeper GCN + GAT (DeeperEnhancedGCN)

*The `concat` operation is adopted by default in the model.

## task 2

Model: 
> - GCN (GCN)
> - GAT (PureGAT)
> - GCN + GAT (EnhancedGCN)
> - Deeper GCN + GAT (DeeperEnhancedGCN)

*The `concat` operation is adopted by default in the model.

Search algorithm:
> TODO