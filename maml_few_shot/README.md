# Model-Agnostic Meta-Learning Algorithm
This folder's files are based on personalized environment by ***Xiaoyang Liu***. The algorithm is shown as below. 
<div align=center>
    <img src="../image/maml.png" alt="maml" style="width: 50%;">
</div>

## Data Spilt and Extraction
In this environment, we split the training data into training set and validation set. There are 23 IC categories in total, 18 for training and 3 for validation. The other 2 IC are aborted for their huge memory usage(over 5GB for each). In order to save the usage of training time, we decided to firstly extract the graph data and save them to `.pkl` files and then direcly load them on the training stage.

`6get_aig.py` is the file to extract the data. Note that for each IC graph, the variants from it can reach a huge number like ***40K+***, so we had to constrain the number to a smaller one. In practice, for each IC graph, we randomly selected only 1/8 of original data. Then we can get total ***100K+*** samples for training and validation. `8launch.sh` uses multi-processing technique to extract data. The concurrent feature extraction process only needs 2 hours.

## Training Stage
The meta-learning features N-way K-shot task formulation. That is, for each iteration in the training stage, we have `bs_task` as the number of tasks, `bs_support_cate` as N (categories in a task), `bs_support_train` as K (instance number of a category in the support set), `bs_support_per` as total instance number of a certain category in a task, i.e., `bs_support_per` - `bs_support_train` is the instance number in the query set for each category. 

For each task, the model will updates its parameters from $\theta$ to $\theta^\prime$ on the support set, and test the $\theta^\prime$ on the query set. The aggregation of the loss on the query set from all the tasks in a iteration will actually update the $\theta$.

## Evaluation Stage
For the test stage, we only need few instances to update the model. The good initialization and quick adaptation can lead to a good performance on the few shot test set.

