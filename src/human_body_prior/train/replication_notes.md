> These are some notes I wrote while trying to replicate this, hopefully
> they might help anyone else who wants to try to do that.

Working on replicating the VPoser code to check that my calculation of
their loss function is correct. Also, I want to know what kind of losses
they achieved.

I've cloned the repo locally and downloaded both versions of the model
they've shared on their `SMPL-X` repo.

Using `amass` environment on `vws83`.

The version of the model in the current `human_body_prior` repo appears to
be `V02_05` based on the directory name.

Unpacking `V02_05.zip` into the `V02_05` directory.

I can't just run their training script because I haven't preprocessed AMASS
according to their specifications yet.

Copying their data preparation script from [this random markdown
file](https://github.com/nghorbani/human_body_prior/blob/master/src/human_body_prior/data/README.md).

Copying only a tiny subset of AMASS to make processing faster. Listed below
the splits I'm going to use:

```
amass_splits = {
    'vald': ['HumanEva'],
    'train': ['ACCAD']
}
```

Tried to run the script and configer is not installed. It's a random
`nghorbani` library.

Installed `configer`.

Hit an error executing script:

```
> python prepare_data.py
[eval-vposer] Preparing data for training VPoser.
Creating pytorch dataset at amass/eval-vposer
Using AMASS body parameters from ./amass_tarfiles
Preparing VPoser data for split vald
Found 0 sequences from HumanEva.
Traceback (most recent call last):
  File "prepare_data.py", line 20, in <module>
    prepare_vposer_datasets(vposer_datadir,amass_splits,amass_dir,logger=logger)
  File "/h/gngdb/repos/human_body_prior/src/human_body_prior/data/prepare_data.py", line 127, in prepare_vposer_datasets
    logger('{} datapoints dumped for split {}. ds_meta_pklpath: {}'.format(len(v), split_name, osp.join(vposer_dataset_dir, split_name)))
UnboundLocalError: local variable 'v' referenced before assignment
```

Oh, I think it wanted me to unpack the tar files first.

Unpacked with `llamass`'s `fast_amass_unpack`.

Next error:

```
> python prepare_data.py
[eval-vposer] Preparing data for training VPoser.
Creating pytorch dataset at amass/eval-vposer
Using AMASS body parameters from ./amass_npz
Preparing VPoser data for split train
Found 252 sequences from ACCAD.
46090 datapoints dumped for split train. ds_meta_pklpath: amass/eval-vposer/train
Traceback (most recent call last):
  File "prepare_data.py", line 20, in <module>
    prepare_vposer_datasets(vposer_datadir,amass_splits,amass_dir,logger=logger)
  File "/h/gngdb/repos/human_body_prior/src/human_body_prior/data/prepare_data.py", line 130, in prepare_vposer_datasets
    'amass_splits':amass_splits.toDict(),
AttributeError: 'dict' object has no attribute 'toDict'
```

Rewriting `human_body_prior` code to fix this; don't cast to dict if it's
already a dict.

That seems to have worked? But it was extremely fast so I am suspicious.

The file this script has created appears to contain pytorch files, so I guess
it's fine.

Training uses `pytorch-lightning` so I can't read it as easily.

Looking at documentation for validation.

It looks like the correct way is to use: `trainer.test(<dataloader>)`.

To avoid changing the code the best way to do this is going to be to process
the data again.

There are a lot of paths to fill in manually in the experiment `yaml`
config.

It looks like this version of VPoser is trained on the `SMPL-X` body
model, which I wasn't expecting because the `VPoser` model definition
definitely appears to take the same shape tensor as SMPL. Also, I don't see
a step in the dataset processing that would convert it to SMPL-X. The AMASS
dataset is normally in SMPL-H format unless converted.

Unless they converted their version of AMASS to SMPL-X and then didn't
update the documentation? Or their model definition?

Decided to point that toward the SMPL-H model files.

There's an `amass_dir` AND a `dataset_basedir` in the `.yaml` file, unsure
which is which.

To play this safe I'm going to start by unpacking the exact same amass
parts they list in this YAML file:

```
  amass_splits:
    test:
    - BMLrub_test
    train:
    - CMU
    - BMLrub_train
    vald:
    - BMLrub_vald
```

Unpacking these and then symlinking the directories into the same
directory as the experiment files so I can write a generic config that
doesn't depend on being in the same filesystem.

Oh, I can't because `BMLrub_test`, `BMLrub_train` and `BMLrub_vald` are all
just part of `BMLrub` and I have no idea how that was split.

I guess I'll unpack it and see if it's obvious.

```
> ls npzfiles/BioMotionLab_NTroje/
rub001  rub009  rub018  rub027  rub035  rub043  rub051  rub059  rub067  rub075  rub084  rub092  rub100  rub109
rub002  rub010  rub020  rub028  rub036  rub044  rub052  rub060  rub068  rub076  rub085  rub093  rub101  rub110
rub003  rub011  rub021  rub029  rub037  rub045  rub053  rub061  rub069  rub077  rub086  rub094  rub102  rub111
rub004  rub012  rub022  rub030  rub038  rub046  rub054  rub062  rub070  rub078  rub087  rub095  rub103  rub112
rub005  rub014  rub023  rub031  rub039  rub047  rub055  rub063  rub071  rub079  rub088  rub096  rub104  rub113
rub006  rub015  rub024  rub032  rub040  rub048  rub056  rub064  rub072  rub080  rub089  rub097  rub105  rub114
rub007  rub016  rub025  rub033  rub041  rub049  rub057  rub065  rub073  rub081  rub090  rub098  rub106  rub115
rub008  rub017  rub026  rub034  rub042  rub050  rub058  rub066  rub074  rub083  rub091  rub099  rub108
```

It is not obvious.

I suppose I'll split them equally.

Wrote a script to split them equally:

```
import shutil
import itertools
from pathlib import Path

def split_bml():
    npz = Path("npzfiles")
    bml = npz/"BioMotionLab_NTroje"
    bmldirs = [p for p in enumerate(sorted(bml.iterdir(), key=lambda s: int(s.name[3:])))]
    # forms three groups
    key = lambda s: (s[0])//(len(bmldirs)//3)

    output_dirs = [npz/d for d in ["BMLrub_test", "BMLrub_train", "BMLrub_vald"]]

    for k, g in itertools.groupby(sorted(bmldirs, key=key), key=key):
        for i, d in g:
            output_dir = output_dirs[k]
            output_dir.mkdir(exist_ok=True)
            print(f"moving {d} to {output_dir/d.name}")
            shutil.move(d, output_dir/d.name)
        assert k < 3
    print(f"removing {bml}")
    shutil.rmtree(bml)

if __name__ == '__main__':
    split_bml()
```

Symlinked these directories, including the body models and ran the script to
process the data after changing the configuration to match this `yaml` file.

Tried running the experiment with these settings, immediately hit error requiring `dotmap`.

```
> python V02_05.py
Traceback (most recent call last):
  File "V02_05.py", line 27, in <module>
    from human_body_prior.tools.configurations import load_config
  File "/h/gngdb/repos/human_body_prior/src/human_body_prior/tools/configurations.py", line 23, in <module>
    from dotmap import DotMap
ModuleNotFoundError: No module named 'dotmap'
```

It does a similar thing to `configer`.

Installed `dotmap`.

Next required is `pytorch-lightning` as expected. Installed that.

Next hit an error where the code is looking for a key in the config that doesn't exist:

```
> python V02_05.py
#training_jobs to be done: 1
Traceback (most recent call last):
  File "V02_05.py", line 54, in <module>
    main()
  File "V02_05.py", line 50, in main
    train_vposer_once(job)
  File "/h/gngdb/repos/human_body_prior/src/human_body_prior/train/vposer_trainer.py", line 291, in train_vposer_once
    model = VPoserTrainer(_config)
  File "/h/gngdb/repos/human_body_prior/src/human_body_prior/train/vposer_trainer.py", line 91, in __init__
    self.bm_train = BodyModel(vp_ps.body_model.bm_fname)
  File "/nobackup/gngdb/envs/amass/lib/python3.7/site-packages/dotmap/__init__.py", line 116, in __getattr__
    return self[k]
  File "/nobackup/gngdb/envs/amass/lib/python3.7/site-packages/dotmap/__init__.py", line 93, in __getitem__
    return self._map[k]
KeyError: 'bm_fname'
```

It looks like the `yaml` config bundled with the model archive is not at the
same version as the code on github. I think `bm_path` is the same thing as `bm_fname`.

Next error I'm missing `trimesh` but it's used in the visualizer and I don't
want to visualize anything.

Disabled it in the config and disabled the import.

Next error is from the changes I made:

```
> python V02_05.py
#training_jobs to be done: 1
/nobackup/gngdb/envs/amass/lib/python3.7/site-packages/pytorch_lightning/trainer/connectors/accelerator_connector.py:111: LightningDeprecationWarning: `Trainer(distributed_backend=ddp)` has been deprecated and will be removed in v1.5. Use `Trainer(accelerator=ddp)` instead.
  f"`Trainer(distributed_backend={distributed_backend})` has been deprecated and will be removed in v1.5."
GPU available: True, used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
Traceback (most recent call last):
  File "V02_05.py", line 54, in <module>
    main()
  File "V02_05.py", line 50, in main
    train_vposer_once(job)
  File "/h/gngdb/repos/human_body_prior/src/human_body_prior/train/vposer_trainer.py", line 338, in train_vposer_once
    trainer.test()
  File "/nobackup/gngdb/envs/amass/lib/python3.7/site-packages/pytorch_lightning/trainer/trainer.py", line 695, in test
    "`model` must be provided to `trainer.test()` when it hasn't been passed in a previous run"
pytorch_lightning.utilities.exceptions.MisconfigurationException: `model` must be provided to `trainer.test()` when it hasn't been passed in a previous run
```

Documentation for the `test` method of `trainer` is [here](https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.trainer.trainer.html#pytorch_lightning.trainer.trainer.Trainer.test).

Next error involves data processing:

```
> python V02_05.py
#training_jobs to be done: 1
/nobackup/gngdb/envs/amass/lib/python3.7/site-packages/pytorch_lightning/trainer/connectors/accelerator_connector.py:111: LightningDeprecationWarning: `Trainer(distributed_backend=ddp)` has been deprecated and will be removed in v1.5. Use `Trainer(accelerator=ddp)` instead.
  f"`Trainer(distributed_backend={distributed_backend})` has been deprecated and will be removed in v1.5."
GPU available: True, used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
[V02_05] -- Creating pytorch dataset at ./data/V02_03
[V02_05] -- Using AMASS body parameters from ./amass
[V02_05] -- Preparing VPoser data for split test
[V02_05] -- Found 0 sequences from BMLrub_test.
Traceback (most recent call last):
  File "V02_05.py", line 54, in <module>
    main()
  File "V02_05.py", line 50, in main
    train_vposer_once(job)
  File "/h/gngdb/repos/human_body_prior/src/human_body_prior/train/vposer_trainer.py", line 338, in train_vposer_once
    trainer.test(model=model)
  File "/nobackup/gngdb/envs/amass/lib/python3.7/site-packages/pytorch_lightning/trainer/trainer.py", line 705, in test
    results = self._run(model)
  File "/nobackup/gngdb/envs/amass/lib/python3.7/site-packages/pytorch_lightning/trainer/trainer.py", line 855, in _run
    self.data_connector.prepare_data(model)
  File "/nobackup/gngdb/envs/amass/lib/python3.7/site-packages/pytorch_lightning/trainer/connectors/data_connector.py", line 75, in prepare_data
    model.prepare_data()
  File "/nobackup/gngdb/envs/amass/lib/python3.7/site-packages/pytorch_lightning/utilities/distributed.py", line 48, in wrapped_fn
    return fn(*args, **kwargs)
  File "/h/gngdb/repos/human_body_prior/src/human_body_prior/train/vposer_trainer.py", line 278, in prepare_data
    prepare_vposer_datasets(self.dataset_dir, self.vp_ps.data_parms.amass_splits, self.vp_ps.data_parms.amass_dir, logger=self.text_logger)
  File "/h/gngdb/repos/human_body_prior/src/human_body_prior/data/prepare_data.py", line 127, in prepare_vposer_datasets
    logger('{} datapoints dumped for split {}. ds_meta_pklpath: {}'.format(len(v), split_name, osp.join(vposer_dataset_dir, split_name)))
UnboundLocalError: local variable 'v' referenced before assignment
```

It appears to be running the same function that I just ran in the dataset
preprocessing script that I was instructed to use in the README?

Also, it's not loading from the model checkpoint.

It looks like it's looking for files to prepare in the output directory I
created before?

Tried pointing it at the directory containing the `npz` files and got the same
error.

Relative file path mistake, fixed it and it is running the preprocessing again,
so the README has indeed provided incorrect instructions.

It didn't hit an error, but it also didn't print anything to indicate what
happened in the test loop.

```
> python V02_05.py
#training_jobs to be done: 1
/nobackup/gngdb/envs/amass/lib/python3.7/site-packages/pytorch_lightning/trainer/connectors/accelerator_connector.py:111: LightningDeprecationWarning: `Trainer(distributed_backend=ddp)` has been deprecated and will be removed in v1.5. Use `Trainer(accelerator=ddp)` instead.
  f"`Trainer(distributed_backend={distributed_backend})` has been deprecated and will be removed in v1.5."
GPU available: True, used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
[V02_05] -- Creating pytorch dataset at ./data/ptfiles/V02_03
[V02_05] -- Using AMASS body parameters from ./data/npzfiles
[V02_05] -- Preparing VPoser data for split test
[V02_05] -- Found 1005 sequences from BMLrub_test.
[V02_05] -- 245618 datapoints dumped for split test. ds_meta_pklpath: ./data/ptfiles/V02_03/test
[V02_05] -- Preparing VPoser data for split train
[V02_05] -- Found 2082 sequences from CMU.
[V02_05] -- Found 1052 sequences from BMLrub_train.
[V02_05] -- 2001810 datapoints dumped for split train. ds_meta_pklpath: ./data/ptfiles/V02_03/train
[V02_05] -- Preparing VPoser data for split vald
[V02_05] -- Found 1004 sequences from BMLrub_vald.
[V02_05] -- 336983 datapoints dumped for split vald. ds_meta_pklpath: ./data/ptfiles/V02_03/vald
[V02_05] -- Dumped final pytorch dataset at ./data/ptfiles/V02_03
initializing ddp: GLOBAL_RANK: 0, MEMBER: 1/1
----------------------------------------------------------------------------------------------------
distributed_backend=nccl
All DDP processes registered. Starting ddp with 1 processes
----------------------------------------------------------------------------------------------------

LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1]
```

Printing the output of `trainer.test(model=model)` also produces nothing. It
could be because the model doesn't have a `test_step` defined?

Trying `trainer.validate(model=model)` instead.

Validation loss was printed but no other information:

```
python V02_05.py
#training_jobs to be done: 1
/nobackup/gngdb/envs/amass/lib/python3.7/site-packages/pytorch_lightning/trainer/connectors/accelerator_connector.py:111: LightningDeprecationWarning: `Trainer(distributed_backend=ddp)` has been deprecated and will be removed in v1.5. Use `Trainer(accelerator=ddp)` instead.
  f"`Trainer(distributed_backend={distributed_backend})` has been deprecated and will be removed in v1.5."
GPU available: True, used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
[V02_05] -- VPoser dataset already exists at ./data/ptfiles/V02_03
initializing ddp: GLOBAL_RANK: 0, MEMBER: 1/1
----------------------------------------------------------------------------------------------------
distributed_backend=nccl
All DDP processes registered. Starting ddp with 1 processes
----------------------------------------------------------------------------------------------------

LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0,1]
Validating: 100%|█████████████████████████████████████████████████████████████████▊| 2627/2633 [00:50<00:00, 55.58it/s][V02_05] -- Epoch 0: val_loss:0.47
[V02_05] -- lr is []
--------------------------------------------------------------------------------
DATALOADER:0 VALIDATE RESULTS
{}
--------------------------------------------------------------------------------
[{}]
```

The checkpoint was not loaded, so this is the initial total loss on the
validation set.

Edited the code to access all of the components of the loss and log them. At
random initialisation:

```
[V02_05] -- Epoch 0: val_loss:0.47, v2v:0.23, kl:3.52, geodesic_matrot:2.17, jtr:0.26
```

Loading the provided checkpoint, I'm not sure it's actually using it:

```
[V02_05] -- Epoch 0: val_loss:0.47, v2v:0.23, kl:3.52, geodesic_matrot:2.17, jtr:0.26
```

Running it again with `trainer.validate(model=model, ckpt_path=resume_from_checkpoint)`:

```
[V02_05] -- Epoch 0: val_loss:0.47, v2v:0.23, kl:3.52, geodesic_matrot:2.17, jtr:0.26
```

What if I just let it train? Enabling `trainer.fit()`:

```
> python V02_05.py
#training_jobs to be done: 1
/nobackup/gngdb/envs/amass/lib/python3.7/site-packages/pytorch_lightning/callbacks/model_checkpoint.py:446: UserWarning: Checkpoint directory ./V02_05/snapshots exists and is not empty.
  rank_zero_warn(f"Checkpoint directory {dirpath} exists and is not empty.")
[V02_05] -- Resuming the training from ./V02_05/snapshots/V02_05_epoch=13_val_loss=0.03.ckpt
Loading ./V02_05/snapshots/V02_05_epoch=13_val_loss=0.03.ckpt
/nobackup/gngdb/envs/amass/lib/python3.7/site-packages/pytorch_lightning/trainer/connectors/accelerator_connector.py:111: LightningDeprecationWarning: `Trainer(distributed_backend=ddp)` has been deprecated and will be removed in v1.5. Use `Trainer(accelerator=ddp)` instead.
  f"`Trainer(distributed_backend={distributed_backend})` has been deprecated and will be removed in v1.5."
GPU available: True, used: True
TPU available: False, using: 0 TPU cores
IPU available: False, using: 0 IPUs
Restoring states from the checkpoint file at ./V02_05/snapshots/V02_05_epoch=13_val_loss=0.03.ckpt
[V02_05] -- VPoser dataset already exists at ./data/ptfiles/V02_03
initializing ddp: GLOBAL_RANK: 0, MEMBER: 1/1
----------------------------------------------------------------------------------------------------
distributed_backend=nccl
All DDP processes registered. Starting ddp with 1 processes
----------------------------------------------------------------------------------------------------

Traceback (most recent call last):
  File "V02_05.py", line 54, in <module>
    main()
  File "V02_05.py", line 50, in main
    train_vposer_once(job)
  File "/h/gngdb/repos/human_body_prior/src/human_body_prior/train/vposer_trainer.py", line 363, in train_vposer_once
    trainer.fit(model)
  File "/nobackup/gngdb/envs/amass/lib/python3.7/site-packages/pytorch_lightning/trainer/trainer.py", line 552, in fit
    self._run(model)
  File "/nobackup/gngdb/envs/amass/lib/python3.7/site-packages/pytorch_lightning/trainer/trainer.py", line 868, in _run
    self.checkpoint_connector.restore_model()
  File "/nobackup/gngdb/envs/amass/lib/python3.7/site-packages/pytorch_lightning/trainer/connectors/checkpoint_connector.py", line 142, in restore_model
    self.trainer.training_type_plugin.load_model_state_dict(self._loaded_checkpoint)
  File "/nobackup/gngdb/envs/amass/lib/python3.7/site-packages/pytorch_lightning/plugins/training_type/training_type_plugin.py", line 152, in load_model_state_dict
    self.lightning_module.load_state_dict(checkpoint["state_dict"])
  File "/nobackup/gngdb/envs/amass/lib/python3.7/site-packages/torch/nn/modules/module.py", line 1407, in load_state_dict
    self.__class__.__name__, "\n\t".join(error_msgs)))
RuntimeError: Error(s) in loading state_dict for VPoserTrainer:
        Unexpected key(s) in state_dict: "bm_train.init_v_template", "bm_train.f", "bm_train.shapedirs", "bm_train.exprdirs", "bm_train.init_expression", "bm_train.J_regressor", "bm_train.posedirs", "bm_train.kintree_table", "bm_train.weights", "bm_train.init_trans", "bm_train.init_root_orient", "bm_train.init_pose_body", "bm_train.init_pose_hand", "bm_train.init_pose_jaw", "bm_train.init_pose_eye", "bm_train.init_betas".
```

It appears now it really is trying to load the config, and it's finding that
there are keys that don't match.

It looks like those might be SMPL-X parameters that aren't found when I
initialise the body model using SMPL-H.

Downloaded SMPL-X body models.

Initialised loading from SMPL-X body models.

Printing the state_dict of the initialised body model yields nothing, which doesn't make sense because there are various buffers registered during the initialisation of the body model.

In debugger, the parameters definitely exist in the `nn.Module` for the body model.

It looks like the flag that disables adding those buffers to the state dict is False by default.
Set `persistent_buffer=True` in the code and I can see the buffers now.

More problems loading the checkpoint:

```
Traceback (most recent call last):
  File "V02_05.py", line 54, in <module>
    main()
  File "V02_05.py", line 50, in main
    train_vposer_once(job)
  File "/h/gngdb/repos/human_body_prior/src/human_body_prior/train/vposer_trainer.py", line 364, in train_vposer_once
    trainer.fit(model)
  File "/nobackup/gngdb/envs/amass/lib/python3.7/site-packages/pytorch_lightning/trainer/trainer.py", line 552, in fit
    self._run(model)
  File "/nobackup/gngdb/envs/amass/lib/python3.7/site-packages/pytorch_lightning/trainer/trainer.py", line 870, in _run
    self.checkpoint_connector.restore_callbacks()
  File "/nobackup/gngdb/envs/amass/lib/python3.7/site-packages/pytorch_lightning/trainer/connectors/checkpoint_connector.py", line 175, in restore_callbacks
    "The checkpoint you're attempting to load follows an"
ValueError: The checkpoint you're attempting to load follows an outdated schema. You can upgrade to the current schema by running `python -m pytorch_lightning.utilities.upgrade_checkpoint --file model.ckpt` where `model.ckpt` is your checkpoint file.
```

Ran the suggested command on both checkpoints.

It ran a validation sanity check at train start:

```
[V02_05] -- Epoch 14: val_loss:0.03, v2v:0.01, kl:29.52, geodesic_matrot:0.10, jtr:0.02
```

Running it over the whole validation set:

```
[V02_05] -- Epoch 0: val_loss:0.51, v2v:0.25, kl:3.52, geodesic_matrot:2.17, jtr:0.26
```

That's as much as I want to know right now.
