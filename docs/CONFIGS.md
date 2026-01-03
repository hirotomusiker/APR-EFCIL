# Configs for Adversarial Pseudo Replay


```
incremental:
  init_cls: 10                               <-- Number of initial task classes (t = 0).
  increment: 10                              <-- Number of incremental task classes (t > 0).
  shuffle_cls: True                          <-- Whether to shuffle dataset classes with args.clsseed.

net:
  net_type: "SplitCosNet"                    <-- Network name.
  backbone_type: "resnet18"                  <-- Backbone name.
  for_insubset: False                        <-- If True, a backbone with first convolution stride = 1 and MaxPool stride = 2 is used (for ImageNet-Subset), following FeCAM.

learner:
  learner_type: "APR"                        <-- Incremental learner name.
  amp: False                                 <-- If True, torch mixed precision package is enabled (faster, less GPU memory).

loss:
  cls_lambda: 1.0                            <-- Classification (cross-entropy) loss weight.
  kd_lambda: 10.0                            <-- Knowledge distillation loss weight.

init_optimizer:                              <-- Optimizer used at the initial task (t = 0).
  type: "sgd"                                <-- Optimizer type.
  lr: 0.1                                    <-- Learning rate.
  momentum: 0.9                              <-- Momentum of SGD.
  weight_decay: 5e-4                         <-- Weight decay used for SGD and Adam.
  grad_accum: 1                              <-- Grad accumulation to reduce GPU memory usage.

incremental_optimizer:                       <-- Optimizer used at the initial task (t > 0).
  type: "sgd"                                <-- Optimizer type.
  lr: 0.01                                   <-- Learning rate.
  momentum: 0.9                              <-- Momentum of SGD.
  weight_decay: 2e-4                         <-- Weight decay used for SGD and Adam.
  grad_accum: 1                              <-- Grad accumulation to reduce GPU memory usage.

init_scheduler:                              <-- scheduler for initial task (t = 0)
  type: "cos"                                <-- "cos" or "step"
  total_epochs: 200                          <-- Number of epochs per task
  milestones: []                             <--Epoch list for LR drops in "step"
  lrdecay: 0.1                               <-- LR drop magnitude in "step"

incremental_scheduler:                       <-- scheduler for incremental tasks (t > 0)
  type: "cos"                                <-- "cos" or "step"
  total_epochs: 100                          <-- Number of epochs per task
  milestones: []                             <--Epoch list for LR drops in "step"
  lrdecay: 0.1                               <-- LR drop magnitude in "step"

protos:                                      <-- prototype and covariance settings
  normalize: True                            <-- Whether to normalize covariance.
  svd_k: Null                                <-- SVD parameter. If int (e.g. 8), top-k decomposition is applied to covariance.
  gamma1_eval: 24                            <-- Covariance shrinkage parameter for Mahalanobis metric.

adc:                                         <-- Settings for Adversarial Drift Compensation.
  do_adc: True                               <-- Enable ADC after task.
  adc_alpha: 25                              <-- Magnitude of perturbation.
  adc_sample_limit: 1000                     <-- Candidates for ADC, closest to target prototype.
  input_data_mode: "adcapr"                  <-- Augmentation mode. "adcapr" ensures reproducible augmentation.
  attack: True                               <-- Enable perturbation to the candidates.
  batchsize: 64                              <-- ADC batch size.
  epochs: 9                                  <-- Number of attacks.
  clamp_x: True                              <-- Whether to clamp perturbed samples to the original max and min values.
  trans_epochs: 64                           <-- Number of epochs to train transfer layer.
  trans_lr: 1e-4                             <-- Learning rate to train transfer layer.
  colorjitter: False                         <-- Enable colorjitter augmentation for the candidate samples.
  cifar10policy: False                       <-- Enable CIFAR10Policy augmentation for the candidate samples.
  imagenetpolicy: False                      <-- Enable ImageNetPolicy augmentation for the candidate samples.

apr:                                         <-- Settings for Adversarial Pseudo Replay.
  do_apr: True                               <-- Enable online APR.
  do_reproducible_trans: True                <-- If True, augmentations used for candidate sampling become reproducible.
  perturb: True                              <-- Enable online perturbation (adversarial attack).
  p_batchsize: 64                            <-- Batch size for pseudo-replay samples, loaded along with the new-task samples.
  perturb_alpha: 64                          <-- Magnitude of perturbation.
  loops: 4                                   <-- Number of adversarial attacks.
  clamp_x: False                             <-- Whether to clamp perturbed samples to the original max and min values.
  apr_sample_limit: 200                      <-- Number of APR candidates per old class.
  colorjitter: True                          <-- Enable colorjitter augmentation for the candidate samples.
  cifar10policy: True                        <-- Enable CIFAR10Policy augmentation for the candidate samples.
  imagenetpolicy: False                      <-- Enable ImageNetPolicy augmentation for the candidate samples.
  proto_noise_mag: 1                         <-- Magnitude of target prototype augmentation.

data:
  type: "iCIFAR100"                          <-- Dataset type.
  cifar10policy: True                        <-- Enable CIFAR10Policy AutoAugment.
  cifar10colorjitter: True                   <-- Enable colorjitter.
  train:
    batch_size: 64                           <-- Batch size for training (t>0).
    init_batch_size: 64                      <-- Batch size for training (t=0).
    num_workers: 2                           <-- Number of workers for train dataloader.
  test:
    batch_size: 128                          <-- Batch size for test inference (t>0).
    init_batch_size: 128                     <-- Batch size for test inference (t=0).
    num_workers: 2                           <-- Number of workers for test dataloader.

log:
  interval: 20                               <-- Log interval (in epochs).
  metrics: ["Linear", "Mahalanobis", "NCM"]  <-- Evaluation metrics.
  use_tqdm: False                            <-- Whether to use tqdm bar for an epoch loop.
  ckpt_idx: []                               <-- List of task indices where model checkpoint is saved.
  ckpt_tmpl: "apr_cifar100_task_{}.pth"      <-- Checkpoint output path template. One `{}` should be included, to put task index there.
```
