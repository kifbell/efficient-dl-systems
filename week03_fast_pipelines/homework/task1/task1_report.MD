Run experiments:

1. Training without scaling. Use static scaling with `scale = 1`. In this experiment I skip optimizer step if the gradients contain NaNs or Infs.

```
python train.py scaler.scaler_type=static scaler.scale=1 wandb_name=no_scaling
```

2. Training with scaling. Use static scaling with `scale = 2 ** 16`.

```
python train.py scaler.scaler_type=static scaler.scale=65536 wandb_name=static_scaling_65536
```

3. Use dynamic scaling. Use `scale = 2 ** 16`, `growth_factor = 2`, `backoff_factor = 0.2` and `growth_interval = 16`. If non-finite values are encountered in gradients multiply scale by backoff_factor. If non-finite values are encountered in gradients for more than growth interval multiply the scale by growth_factor until they become finite. I update scale factor after each gradient step using `update()` function.

```
python train.py scaler.scaler_type=dynamic scaler.scale=65536 wandb_name=dynamic_scaling_65536
```

We can see that given this scale the gradients are always finite. This is why I have conducted one more experiment with higher initial `scale = 2 ** 32`.
```
python train.py scaler.scaler_type=dynamic scaler.scale=4294967296 wandb_name=dynamic_scaling_4294967296
``` 

Runs link: https://wandb.ai/dog-kirill-belyakov/edl-hw2-task-1?nw=nwuserdogkirillbelyakov


The run contains information on batch accuracy, total accuracy, gradient norm, loss and current scale factor.


Conclusions from charts:
1. Without scaling the models achieves bad accuracy which it is lower than with scaling.
2. Static scaling improves the speed of training significantly. It also achieves higher accuracy (0.985+ according to the task).
3. Dynamic scaling automatically adjusts the scaling factor to be the maximum that preserves finite gradients. It seems that learning with it is faster than learning with static small scale if we do not count the first epochs where gradients were not finite and scale was declining. The final accuracy is almost the same as for static scaling which is 0.985+ that aligns with our goal for the task.


