# Power of Two Quantization

### Create a VM in GCP

```bash
$ python3 create_vm.py --project_id="high-performance-ml" --vm_name="sleds" --disk_size=100 --gpu_type="nvidia-tesla-t4" --gpu_count=4 --machine_type="n1-standard-8"
```

### Install dependencies

```bash
$ pip install -r requirements.txt
```
### Test a single training

```
$ export LD_LIBRARY_PATH=
$ export OMP_NUM_THREADS=1
$ torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py --model_type=resnet20 --dataset=cifar --quantizer_type=none --bits=4 --num_epochs=164 --batch_size=128 --lr=0.1
```

### Run train scripts

For a given model and dataset, perform full precision training then all QAT configurations

```bash
$ ./train_launch.sh resnet20 cifar 164 128 0.1
$ ./train_launch.sh resnet32 cifar 164 128 0.1
$ ./train_launch.sh resnet44 cifar 164 128 0.1
$ ./train_launch.sh resnet56 cifar 164 128 0.1
```

### Run test scripts

```bash
# get test results for full precision, PTQ and QAT
$ torchrun --standalone --nnodes=1 --nproc-per-node=4 test.py --model_type=resnet20 --dataset=cifar
```