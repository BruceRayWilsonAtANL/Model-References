# ResNet50 PyTorch Model Scaling

## NOTE

For those who need to count HPUs in use on a habana node, this hack works:

```bash
echo $(hl-smi | grep "Mib \/" | grep -v 512 | wc -l)
```

## Login

On your development machine:

```bash
ssh -J wilsonb@homes.cels.anl.gov wilsonb@habana-01.ai.alcf.anl.gov
CELS password
```

## Clone ResNet50

```bash
cd ~/DL/github.com/BruceRayWilsonAtANL
git clone https://github.com/BruceRayWilsonAtANL/Model-References.git

cd Model-References/PyTorch/computer_vision/classification/torchvision
```

## Create Venv

```bash
python3.8 -m venv --system-site-packages ~/venvs/habana/resnet50_PT_venv

source ~/venvs/habana/resnet50_PT_venv/bin/activate
```

## Define PYTHON

Set the Python environment variable:

```bash
export PYTHON=$(which python)
```

## Define PYTHONPATH

Set the Python path:

```bash
export PYTHONPATH=~/DL/github.com/BruceRayWilsonAtANL/Model-References/:$(which python)
```

## Environment Variables

Set these:

```bash
export HABANA_LOGS=~/.habana_logs
export MPI_ROOT=/usr/local/openmpi
```

## Change Directory

```bash
cd ~/DL/github.com/BruceRayWilsonAtANL/Model-References/PyTorch/computer_vision/classification/torchvision
```

## Install Requirements

Install required packages using pip:

```bash
python3 -m pip install -r requirements.txt
```

### Training Data

I think all of this was done when I set up the TFRecords.

ImageNet 2012 dataset needs to be organized as per PyTorch requirements. PyTorch requirements are specified in the link below which contains scripts to organize ImageNet data.
https://github.com/soumith/imagenet-multiGPU.torch

#### Set Up

From https://github.com/soumith/imagenet-multiGPU.torch/blob/master/README.md

##### Data processing

**The images do not need to be preprocessed or packaged in any database.** It is preferred to keep the dataset on an [SSD](http://en.wikipedia.org/wiki/Solid-state_drive) but we have used the data loader comfortably over NFS without loss in speed.

We just use a simple convention: SubFolderName == ClassName.
So, for example: if you have classes {cat,dog}, cat images go into the folder dataset/cat and dog images go into dataset/dog

```bash
cd /lambda_stor/data/imagenet
```

The training images for ImageNet are already in appropriate subfolders (like n07579787, n07880968).
You need to get the validation groundtruth and move the validation images into appropriate subfolders.
To do this, download ILSVRC2012_img_train.tar ILSVRC2012_img_val.tar and use the following commands:

```bash
# extract train data
mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
#tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
tar -xvf ILSVRC2012_img_train.tar
find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
# extract validation data
cd ../ && mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash
```

Now you are all set!

## Single-Server Habana Resnet Training Steps

### Using LARS Optimizer

Using LARS optimizer usually requires changing the default values of some hyperparameters and should be manually set for resnet_ctl_imagenet_main.py. The recommended parameters together with their default values are presented below:

| Parameter          | Value      |
| ------------------ | ---------- |
| optimizer          | LARS       |
| base_learning_rate | 2.5 or 9.5*|
| warmup_epochs      | 3          |
| lr_schedule        | polynomial |
| label_smoothing    | 0.1        |
| weight_decay       | 0.0001     |
| single_l2_loss_op  | True       |

*2.5 is the default value for single card (1 HPU) trainings, otherwise, the default is 9.5. These values have been determined experimentally.

### Single Card and Multi-Card Training Examples

**Run training on 1 HPU:**

- ResNet50, lazy mode, BF16 mixed precision, batch Size 256, custom learning rate, Habana dataloader (with hardware decode support on **Gaudi2**), 1 HPU on a single server:

The path **/lambda_stor/habana/data/tensorflow/imagenet** is poorly named.  The files stored at that
location are PyTorch compatible and not TensorFlow compatible.

This works.  20220825

```bash
$PYTHON -u train.py --dl-worker-type HABANA --batch-size 256 --model resnet50 --device hpu --workers 8 --print-freq 20 --dl-time-exclude False --deterministic --data-path /lambda_stor/habana/data/tensorflow/imagenet --epochs 90 --hmp --hmp-bf16 ./ops_bf16_Resnet.txt --hmp-fp32 ./ops_fp32_Resnet.txt  --lr 0.1 --custom-lr-values 0.1 0.01 0.001 0.0001 --custom-lr-milestones 0 30 60 80
```

This works and is the same as above.  20220825

```bash
time $PYTHON -u train.py \
    --dl-worker-type HABANA \
    --batch-size 256 \
    --model resnet50 \
    --device hpu \
    --workers 8 \
    --print-freq 20 \
    --dl-time-exclude False \
    --deterministic \
    --data-path /lambda_stor/habana/data/tensorflow/imagenet \
    --epochs 1 \
    --hmp \
    --hmp-bf16 ./ops_bf16_Resnet.txt \
    --hmp-fp32 ./ops_fp32_Resnet.txt \
    --lr 0.1 \
    --custom-lr-values 0.1 0.01 0.001 0.0001 \
    --custom-lr-milestones 0 30 60 80
```

This works.  20221025.  Using a modified ResNetA that works with standard jpeg files.

```bash
time $PYTHON -u train.py \
    --dl-worker-type HABANA \
    --batch-size 256 \
    --model ResNetA \
    --device hpu \
    --workers 8 \
    --print-freq 20 \
    --dl-time-exclude False \
    --deterministic \
    --data-path /lambda_stor/habana/data/tensorflow/imagenet \
    --epochs 1 \
    --hmp \
    --hmp-bf16 ./ops_bf16_Resnet.txt \
    --hmp-fp32 ./ops_fp32_Resnet.txt \
    --lr 0.1 \
    --custom-lr-values 0.1 0.01 0.001 0.0001 \
    --custom-lr-milestones 0 30 60 80
```

Using a ResNetA.  Oops.  Does not work.  Use my dataloader.

```bash
time $PYTHON -u train.py \
    --dl-worker-type HABANA \
    --batch-size 256 \
    --model ResNetA \
    --device hpu \
    --workers 8 \
    --print-freq 20 \
    --dl-time-exclude False \
    --deterministic \
    --data-path ~/HL/pipeline_resnet_hl \
    --epochs 1 \
    --hmp \
    --hmp-bf16 ./ops_bf16_Resnet.txt \
    --hmp-fp32 ./ops_fp32_Resnet.txt \
    --lr 0.1 \
    --custom-lr-values 0.1 0.01 0.001 0.0001 \
    --custom-lr-milestones 0 30 60 80
```

### One HPU on 1 server, batch 256, 25 epochs, BF16 precision, LARS

Scaling Runs

```bash
time $PYTHON resnet_ctl_imagenet_main.py \
    -dt bf16 \
    --data_loader_image_type bf16 \
    -te 25 \
    -bs 256 \
    --optimizer LARS \
    --base_learning_rate 9.5 \
    --warmup_epochs 3 \
    --lr_schedule polynomial \
    --label_smoothing 0.1 \
    --weight_decay 0.0001 \
    --single_l2_loss_op \
    --data_dir /lambda_stor/data/imagenet/tf_records
```

Statistics:

```console
real    67m37.184s
'avg_exp_per_second': 1594.6571942068174
```

### Two HPUs on 1 server, batch 256, 25 epochs, BF16 precision, LARS

```bash
time mpirun \
    --allow-run-as-root \
    --bind-to core \
    -np 2 \
    --map-by socket:PE=7 \
    --merge-stderr-to-stdout \
       $PYTHON resnet_ctl_imagenet_main.py \
            --dtype bf16 \
            --data_loader_image_type bf16 \
            --use_horovod \
            -te 25 \
            -bs 256 \
            --optimizer LARS \
            --base_learning_rate 9.5 \
            --warmup_epochs 3 \
            --lr_schedule polynomial \
            --label_smoothing 0.1 \
            --weight_decay 0.0001 \
            --single_l2_loss_op \
            --data_dir /lambda_stor/data/imagenet/tf_records
```

Statistics:

```console
real    36m48.760s
'avg_exp_per_second': 2990.6616862467968
```

### Four HPUs on 1 server, batch 256, 25 epochs, BF16 precision, LARS

```bash
time mpirun \
    --allow-run-as-root \
    --bind-to core \
    -np 4 \
    --map-by socket:PE=7 \
    --merge-stderr-to-stdout \
       $PYTHON resnet_ctl_imagenet_main.py \
            --dtype bf16 \
            --data_loader_image_type bf16 \
            --use_horovod \
            -te 25 \
            -bs 256 \
            --optimizer LARS \
            --base_learning_rate 9.5 \
            --warmup_epochs 3 \
            --lr_schedule polynomial \
            --label_smoothing 0.1 \
            --weight_decay 0.0001 \
            --single_l2_loss_op \
            --data_dir /lambda_stor/data/imagenet/tf_records
```

Statistics:

```console
real    19m34.556s
'avg_exp_per_second': 5708.747706392027
```

### Eight HPUs on 1 server, batch 256, 25 epochs, BF16 precision, LARS

```bash
time mpirun \
    --allow-run-as-root \
    --bind-to core \
    -np 8 \
    --map-by socket:PE=7 \
    --merge-stderr-to-stdout \
       $PYTHON resnet_ctl_imagenet_main.py \
            --dtype bf16 \
            --data_loader_image_type bf16 \
            --use_horovod \
            -te 25 \
            -bs 256 \
            --optimizer LARS \
            --base_learning_rate 9.5 \
            --warmup_epochs 3 \
            --lr_schedule polynomial \
            --label_smoothing 0.1 \
            --weight_decay 0.0001 \
            --single_l2_loss_op \
            --data_dir /lambda_stor/data/imagenet/tf_records
```

Statistics:

```console
real    11m11.386s
'avg_exp_per_second': 10484.725513908621
```

## Multi-Server Habana Resnet Training Steps

### One-time per user ssh key set up

On both Habana machines set up the ssh key

#### Habana-01

On **Habana-01**

```bash
mkdir ~/.ssh
cd ~/.ssh
ssh-keygen -t rsa -b 4096
#Accecpt default filename of id_rsa
#Enter passphrase (empty for no passphrase):
#Enter same passphrase again:
cat id_rsa.pub >> authorized_keys
```

```bash
ssh-keyscan -H 140.221.77.101 >> ~/.ssh/known_hosts
```

You should see:

```console
# 140.221.77.101:22 SSH-2.0-OpenSSH_8.2p1 Ubuntu-4ubuntu0.5
# 140.221.77.101:22 SSH-2.0-OpenSSH_8.2p1 Ubuntu-4ubuntu0.5
# 140.221.77.101:22 SSH-2.0-OpenSSH_8.2p1 Ubuntu-4ubuntu0.5
# 140.221.77.101:22 SSH-2.0-OpenSSH_8.2p1 Ubuntu-4ubuntu0.5
# 140.221.77.101:22 SSH-2.0-OpenSSH_8.2p1 Ubuntu-4ubuntu0.5
```

```bash
ssh-keyscan -H 140.221.77.102 >> ~/.ssh/known_hosts
```

You should see:

```console
# 140.221.77.102:22 SSH-2.0-OpenSSH_8.2p1 Ubuntu-4ubuntu0.5
# 140.221.77.102:22 SSH-2.0-OpenSSH_8.2p1 Ubuntu-4ubuntu0.5
# 140.221.77.102:22 SSH-2.0-OpenSSH_8.2p1 Ubuntu-4ubuntu0.5
# 140.221.77.102:22 SSH-2.0-OpenSSH_8.2p1 Ubuntu-4ubuntu0.5
# 140.221.77.102:22 SSH-2.0-OpenSSH_8.2p1 Ubuntu-4ubuntu0.5
```

Verify you can ssh from habana-01 to habana-02 without a password.

Verify vice-versa.

### ResNet50

Ensure $MPI_ROOT is set

```bash
echo $MPI_ROOT
```

If not,

```bash
export MPI_ROOT=/usr/local/openmpi
```

### Tensorflow

Test with:

```bash
cd /path/to/Model-References/TensorFlow/computer_vision/Resnets/resnet_keras
```

Verify a connection with

```bash
mpirun -v --allow-run-as-root  --mca btl_tcp_if_include enp75s0f0,ens1f0 --tag-output --merge-stderr-to-stdout --prefix /usr/local/openmpi --host 140.221.77.101:8,140.221.77.102:8 hostname
```

Almost immediately, you should see

```console
[1,0]<stdout>:habana-01
[1,1]<stdout>:habana-01
[1,2]<stdout>:habana-01
[1,3]<stdout>:habana-01
[1,4]<stdout>:habana-01
[1,5]<stdout>:habana-01
[1,6]<stdout>:habana-01
[1,7]<stdout>:habana-01
[1,8]<stdout>:habana-02
[1,9]<stdout>:habana-02
[1,10]<stdout>:habana-02
[1,11]<stdout>:habana-02
[1,12]<stdout>:habana-02
[1,13]<stdout>:habana-02
[1,14]<stdout>:habana-02
[1,15]<stdout>:habana-02
```

If the above command hangs, there is a communication problem.

### Sixteen HPUs on 2 servers, batch 256, 25 epochs, BF16 precision, LARS

```bash
mpirun -v \
    --allow-run-as-root \
    -np 16 \
    --mca btl_tcp_if_include 192.168.201.0/24 \
    --tag-output \
    --merge-stderr-to-stdout \
    --prefix /usr/local/openmpi \
    -H 140.221.77.101:8,140.221.77.102:8 \
    -x GC_KERNEL_PATH \
    -x HABANA_LOGS \
    -x PYTHONPATH \
    -x HCCL_SOCKET_IFNAME=enp75s0f0,ens1f0 \
    -x HCCL_OVER_TCP=1 \
    $PYTHON resnet_ctl_imagenet_main.py \
        -dt bf16 \
        -dlit bf16 \
        -bs 256 \
        -te 25 \
        --use_horovod \
        --data_dir /lambda_stor/data/imagenet/tf_records/ \
        --optimizer LARS \
        --base_learning_rate 9.5 \
        --warmup_epochs 3 \
        --lr_schedule polynomial \
        --label_smoothing 0.1 \
        --weight_decay 0.0001 \
        --single_l2_loss_op \
        --model_dir=`mktemp -d`
```

Statistics:

```console
2022-09-27 18:23:10.461103
2022-09-27 18:31:34.638972
real    8:24
'avg_exp_per_second': 15481.107839604083

### PyTorch

This section has not been tested.

```bash
cd Model-References/PyTorch/computer_vision/classification/torchvision/
export MASTER_PORT=12355
export MASTER_ADDR=192.168.201.101
mpirun --allow-run-as-root --mca --bind-to core --map-by ppr:4:socket:PE=7 -np 16 --mca btl_tcp_if_include 192.168.201.0/24 --merge-stderr-to-stdout --prefix $MPI_ROOT -H 140.221.77.101:8,140.221.77.102:8 -x GC_KERNEL_PATH -x PYTHONPATH -x MASTER_ADDR -x MASTER_PORT -x HCCL_SOCKET_IFNAME=enp75s0f0,ens1f0 -x HCCL_OVER_TCP=1 $PYTHON -u train.py --batch-size=256 --model=resnet50 --device=hpu --workers=8 --print-freq=1 --deterministic --data-path=/lambda_stor/habana/data/tensorflow/imagenet --epochs=2 --hmp --hmp-bf16 ./ops_bf16_Resnet.txt --hmp-fp32 ./ops_fp32_Resnet.txt --custom-lr-values 0.475
```
