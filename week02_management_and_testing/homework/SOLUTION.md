# Week 2 home assignment

# Writing tests and fixing bugs


```
tests summary 
7.97s call     tests/test_model.py::test_unet[input_tensor3-40]
1.85s call     tests/test_model.py::test_unet[input_tensor2-30]
0.50s call     tests/test_model.py::test_unet[input_tensor1-20]
0.25s call     tests/test_model.py::test_diffusion
0.19s call     tests/test_model.py::test_unet[input_tensor0-10]

11.35s setup    tests/test_pipeline.py::test_train_on_one_batch[cpu]
3.08s call     tests/test_pipeline.py::test_train_on_one_batch[cpu]
1.35s call     tests/test_pipeline.py::test_train_on_one_batch[cuda]
0.63s setup    tests/test_pipeline.py::test_train_on_one_batch[cuda]
```

- first of all, initialize all the tensors on the same device (gpu)
- loss function have better return detached value -> use item to te `detach + cpu` . I have added `.item()` to the loss before returning it.
- Run unit tests firs. Tensors seemed to have inconsistent dimensions. I printed the dimensions of each tensor in the Unet's forward pass. I have also printed the sum of two tensors before the error. `torch.Size([2, 256, 1, 1]) + torch.Size([2, 256]) = torch.Size([2, 256, 2, 256])` which is interesting, but not what we expect. To tackle this problem just unsqueeze the dimensions of the second tensor. `self.timestep_embedding(t).distributionshape = torch.Size([2, 256]) --> self.timestep_embedding(t).shape = torch.Size([2, 256, 1, 1])`
- in provided code `eps` is sampled from uniform distribution. However, in the original paper it is sampled from normal distribution. I replaced `eps = torch.rand_like(x)` with `eps = torch.randn_like(x)` and the tests passed.
- in provided code the loss do not decrease. The training loop looked obvious, so I went back into the diffusion model code to investigate. I noticed the problem with x_t formula, the variance of x_t should not change, so it's formula was definitely incorrect. I have checked the outputs of `get_schedules` function. The output `sqrt_one_minus_alpha_prod` was exactly the coefficient needed for the correct formula (such formula that does not change the variance). Also this coefficient was not used in model. --> replace the coefficient with the correct one (`one_minus_alpha_over_prod --> sqrt_one_minus_alpha_prod`).
- I implemented the `test_training` function similar to the main.py on subset of the dataset. By implementing this test I found that sample function also had problems with devices, so I fixed it. After it pytest coverage is above 80% (only wandb logging parts of code are not tested).
- added postprocessing parameters to the diffusion model to normalize the pictures back (to return back mean and standard deviation).

# Split preparing dataset and training

I decomposed the code into two parts:
1. Prepare dataset: download it
2. Train model and generate samples

These two stages were used in DVC. They could be done in two simple commands:
```
python prepare_dataset.py
python train.py
```

# Use hydra config

Hydra has one useful function `instantiate` that allows to specify classes names in config. I have used it to transfer model's and dataloader's parameters and even dataset specification into different parts of config. All parameters from the task could be specified and I have added some new ones:
1. `dataset`. Specify the dataset class and dataset parameters
2. `model`. Specify model architecture from classes that are available in modelling
3. `train`. Specify optimizer type, optimizer parameters, number of epochs and dataloader parameters. Also here user can specify output directories and add new augmentations (like flipping).


# Additional improvements

- and hydra & pydantic for strong config validation
- Add checkpoints to save model and optimizer parameters.
- Display the speed of training in pipeline tests.
- Add sample frequency to config to sample once per N epochs.
- Set random seeds for reproducible training.

# Experiments

Main experiment for Task 1 (100 epochs of training with default hyperparameters). I was lucky to train for exactly 100-150 epochs on H100.  if took from 30 to 90 minutes depending on the batch size.

# DVC

As I expressed before I have divided the pipeline into data preparation and training model itself. The first part requires only preparation script and dataset config and produces directory with the dataset. The second part requires the train script and model code. The second part produces model checkpoints.

I have implemented all stages in `dvc.yaml` 

# Summary

All tasks (1, 2, 3, 4) have been completed.


## Runs links


- `dvc exp run --set-param 'train.loop.num_epochs=150' -v`

    https://wandb.ai/dog-kirill-belyakov/diffusion_hw_1/runs/zjrule4r?nw=nwuserdogkirillbelyakov

- `dvc exp run --set-param 'train.loop.batch_size=128' --set-param 'train.loop.num_epochs=100' -v`

    https://wandb.ai/dog-kirill-belyakov/diffusion_hw_1/runs/jwe85nxt?nw=nwuserdogkirillbelyakov

- `dvc exp run --set-param 'train.loop.batch_size=1048' --set-param 'train.loop.num_epochs=150' --set-param 'train.optimizer.lr=0.01' -v`
    
    https://wandb.ai/dog-kirill-belyakov/diffusion_hw_1/runs/2eu32a36?nw=nwuserdogkirillbelyakov
