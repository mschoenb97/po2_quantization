# Power of Two Quantization

### Description and Outline 📐

Here we demonstrate the use of the standard power-of-two quantization formula

$$ PO2(x) = 2 ^ {\mathrm{round} (\log_2(x))} $$

against our improved quantization formula

$$ PO2_+(x) = 2 ^ {\mathrm{round} \left(\log_2\left(\sqrt{8/9} \cdot x\right)\right) }.$$

We use ResNet, MobileNet, and MobileVit models, all of which are available in the `models` 
directory. We also test on CIFAR and ImageNet data, which are available in the `data` and `resnet_data`
directories. The main launch script is `train_launch.sh`, which we will describe how to use below. 


### Create a VM in GCP ☁︎

```bash
python3 create_vm.py --project_id="high-performance-ml" --vm_name="sleds" --disk_size=100 --gpu_type="nvidia-tesla-t4" --gpu_count=4 --machine_type="n1-standard-8"
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Download Data

```bash
python download_data.py --dataset=cifar
huggingface-cli login # enter token first
python download_data.py --dataset=imagenet
```

### Test a Single Training Run 🏃‍♂️

```
export LD_LIBRARY_PATH=
export OMP_NUM_THREADS=1
torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py --model_type=resnet20 --dataset=cifar --quantizer_type=none --bits=4 --num_epochs=164 --batch_size=128 --lr=0.1 --seed=8
```

### Run Train Scripts

For a given model and dataset, perform full precision training then all QAT configurations

```bash
# <model_type> <dataset> <num_epochs> <batch_size> <learning_rate> <num_gpus>
./train_launch.sh resnet20 cifar 164 128 0.1 4
./train_launch.sh resnet32 cifar 164 128 0.1 4
./train_launch.sh resnet44 cifar 164 128 0.1 4
./train_launch.sh resnet56 cifar 164 128 0.1 4
./train_launch.sh mobilenet cifar 164 128 0.1 4
./train_launch.sh mobilevit cifar 164 128 0.1 4

# only perform full precision training for imagenet
torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py --model_type=resnet56 --dataset=imagenet --quantizer_type=none --bits=4 --num_epochs=164 --batch_size=128 --lr=0.1 --seed=8
torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py --model_type=mobilenet --dataset=imagenet --quantizer_type=none --bits=4 --num_epochs=164 --batch_size=128 --lr=0.1 --seed=8
torchrun --standalone --nnodes=1 --nproc-per-node=4 train.py --model_type=mobilevit --dataset=imagenet --quantizer_type=none --bits=4 --num_epochs=164 --batch_size=128 --lr=0.1 --seed=8
```

### Run Test Scripts 👨‍💻

```bash
# get test results for full precision, PTQ and QAT across all seeds
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

### Results

For results, charts, and tables, see `analysis.ipynb`. Here we see mixed results.
Sometimes the improved formulas give us better generalization error and quantization error,
but not always. We intend to run each of the experiments with different seeds in order 
to get more robust results. 
