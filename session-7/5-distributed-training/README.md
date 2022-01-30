## Example inspired by the [official PyTorch example](https://github.com/pytorch/examples/blob/master/imagenet/main.py)

# Distributed training
This `train.py` is ready to use for distributed training, which is the fastest way to train when using multiple GPUs (even if using a single node!). This training script can serve as the base for your training scripts if you want to leveraged multiple GPUs.

<!-- To launch the script it is recommended that you use the `distributed.launch` utility from PyTorch explained [here](https://pytorch.org/docs/stable/distributed.html#launch-utility). -->

For training with a single node you can run:

```
>>> python train.py -a resnet50 --dist-url 'tcp://127.0.0.1:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 1 --rank 0
```

For training with multiple nodes, you have to run the following commands in each of the nodes:

Node 0:
```
>>> python train.py -a resnet50 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 0
```

Node 1:
```
>>> python train.py -a resnet50 --dist-url 'tcp://IP_OF_NODE0:FREEPORT' --dist-backend 'nccl' --multiprocessing-distributed --world-size 2 --rank 1
```
...

For more advanced utilization, you can take a look at the [official tutorial](https://pytorch.org/tutorials/beginner/dist_overview.html) and the [official documentation](https://pytorch.org/docs/stable/distributed.html).
