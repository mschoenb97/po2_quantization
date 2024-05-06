lealearlear# Power of Two Quantization

### Create a VM in GCP

```bash
$ python3 create_vm.py --project_id="high-performance-ml" --vm_name="sleds" --disk_size=100 --gpu_type="nvidia-tesla-t4" --gpu_count=4 --machine_type="n1-standard-8"
```

### Install dependencies

```bash
$ pip install -r requirements.txt

### Download data

```bash
$ python download_data.py --dataset=cifar
$ huggingface-cli login # enter token first
$ python download_data.py --dataset=imagenet
```

### Test a single training run

```
export LD_LIBRARY_PATH=
export OMP_NUM_THREADS=1
torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py --model_type=resnet20 --dataset=cifar --quantizer_type=none --bits=4 --num_epochs=164 --batch_size=128 --lr=0.1
```

### Run train scripts

For a given model and dataset, perform full precision training then all QAT configurations

```bash
./train_launch.sh resnet20 cifar 164 128 0.1
./train_launch.sh resnet32 cifar 164 128 0.1
./train_launch.sh resnet44 cifar 164 128 0.1
./train_launch.sh resnet56 cifar 164 128 0.1
./train_launch.sh mobilenet cifar 164 128 0.1
./train_launch.sh mobilevit cifar 164 128 0.1

# only perform full precision training for imagenet
torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py --model_type=resnet56 --dataset=imagenet --quantizer_type=none --bits=4 --num_epochs=164 --batch_size=128 --lr=0.1
torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py --model_type=mobilenet --dataset=imagenet --quantizer_type=none --bits=4 --num_epochs=164 --batch_size=128 --lr=0.1
torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py --model_type=mobilevit --dataset=imagenet --quantizer_type=none --bits=4 --num_epochs=164 --batch_size=128 --lr=0.1
```

### Run test scripts

```bash
# get test results for full precision, PTQ and QAT
python test.py --model_type=resnet20 --dataset=cifar
python test.py --model_type=resnet32 --dataset=cifar
python test.py --model_type=resnet44 --dataset=cifar
python test.py --model_type=resnet56 --dataset=cifar
python test.py --model_type=mobilenet --dataset=cifar
python test.py --model_type=mobilevit --dataset=cifar

# get test results for full precision and PTQ for imagenet
python test.py --model_type=resnet56 --dataset=imagenet --skip_qat=True
python test.py --model_type=mobiletnet --dataset=imagenet --skip_qat=True
python test.py --model_type=mobilevit --dataset=imagenet --skip_qat=True
```