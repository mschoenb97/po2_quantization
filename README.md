# Power of Two Quantization

Create a VM in GCP

$ python3 create_vm.py --project_id="high-performance-ml" --vm_name="sleds" --disk_size=100 --gpu_type="nvidia-tesla-t4" --gpu_count=4 --machine_type="n1-standard-8"

Install dependencies

$ pip install -r requirements.txt

Run scripts

$ ./train_launch.sh resnet20 cifar
