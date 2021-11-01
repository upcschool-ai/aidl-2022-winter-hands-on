# Deep Learning Tips & Tricks

1. To find the correct architecture, always start overfitting and then reduce it. This will also help you debug, if the network is not able to learn the train set.

2. Almost all big neural networks will benefit from `BatchNorm`.

3. For fast and decent results, use Adam as optimizer. If you want the best results, use SGD.

4. Add an scheduler. PyTorch’s [OneCycleLR](https://pytorch.org/docs/stable/optim.html?highlight=onecyclelr#torch.optim.lr_scheduler.OneCycleLR) scheduler is awesome.

5. If you can, do data augmentation to generate more data for free. This is usually one of the best ways to improve a model.

6. Always use `DataLoader` and set `num_workers` to at least 4.

7. When training with GPUs, always check GPU usage by executing `nvidia-smi` in a terminal. If the usage is not close to 100%, you should increase batch size or optimize data loading. GPUs at low % are slower than CPUs!!

If you have any other tip, feel free to do a PR to this repo and I will add it!
