# Power of Two Quantization

### Create a VM in GCP

```bash
$ python3 create_vm.py --project_id="high-performance-ml" --vm_name="sleds" --disk_size=100 --gpu_type="nvidia-tesla-t4" --gpu_count=4 --machine_type="n1-standard-8"
```

### Install dependencies

```bash
$ pip install -r requirements.txt
```

### Run train scripts

```bash
# first perform full precision training then all QAT configurations
$ ./train_launch.sh resnet20 cifar 164
$ ./train_launch.sh resnet32 cifar 164
$ ./train_launch.sh resnet44 cifar 164
$ ./train_launch.sh resnet56 cifar 164
```

### Run test scripts

```bash
# get test results for full precision, PTQ and QAT
$ python test.py resnet20 cifar
```