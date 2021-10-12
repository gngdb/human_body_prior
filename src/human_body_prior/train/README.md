# Train VPoser from Scratch

The original instructions are:

> To train your own VPoser with new configuration duplicate the provided
> **V02_05** folder while setting a new experiment ID and change the settings
> as you desire.  First you would need to download the
> [AMASS](https://amass.is.tue.mpg.de/) dataset, then following the [data
> preparation tutorial](../data/README.md) prepare the data for training.
> Following is a code snippet for training that can be found in the [example
> training
> experiment](https://github.com/nghorbani/human_body_prior/blob/master/src/human_body_prior/train/V02_05/V02_05.py):

But, that will fail because the code runs the preprocessing anyway. All
that matters is unpacking the AMASS tar files to the correct directory and
pointing the experiment at that file. I also hit a number of other problems
trying to replicate this that can be found [here](./replication_notes.md).

I rewrote the experiment config and had to make some edits to the
repository. Here's an example of the directory structure I used:

```
├── __init__.py
├── __pycache__
│   ├── __init__.cpython-37.pyc
│   └── vposer_trainer.cpython-37.pyc
├── README.md
├── replication_notes.md
├── V02_05
│   ├── data -> /nobackup/gngdb/amass_vposer_V02_05
│   │   ├── npzfiles
│   │   ├── ptfiles
│   │   ├── split_BML.py
│   │   ├── tarfiles
│   │   └── V02_03
│   ├── __init__.py
│   ├── prepare_data.py
│   ├── smplh -> /nobackup/gngdb/body_models/smplh
│   │   ├── female
│   │   ├── info.txt
│   │   ├── LICENSE.txt
│   │   ├── male
│   │   ├── neutral
│   │   ├── SMPLH_FEMALE.npz
│   │   ├── SMPLH_MALE.npz
│   │   └── SMPLH_NEUTRAL.npz
│   ├── smplx -> /nobackup/gngdb/body_models/smplx
│   │   ├── SMPLX_FEMALE.npz
│   │   ├── SMPLX_FEMALE.pkl
│   │   ├── SMPLX_MALE.npz
│   │   ├── SMPLX_MALE.pkl
│   │   ├── SMPLX_NEUTRAL.npz
│   │   ├── SMPLX_NEUTRAL.pkl
│   │   ├── smplx_npz.zip
│   │   └── version.txt
│   ├── training_experiments
│   │   └── V02_05
│   ├── V02_05
│   │   ├── code
│   │   ├── snapshots
│   │   ├── tensorboard
│   │   ├── V02_05.log
│   │   └── V02_05.yaml
│   ├── V02_05.log
│   ├── V02_05.py
│   └── V02_05.yaml
└── vposer_trainer.py
```

I ran the validation loop on the trained model with:

```
python V02_05.py
```

Other previous notes:

> The above code uses yaml configuration files to handle experiment settings.
> It loads the default settings in *<expr_id>.yaml* and overloads it with
> your new args. 
> 
> The training code, will dump a log file along with tensorboard readable
> events file.
