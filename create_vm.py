from typing import Dict, List

import fire
from google.cloud import compute_v1
from tqdm import tqdm

"""
Instantiate and run a single GCP instance with a GPU attached
    
Run the following for help:    
$ python3 create_vm.py --help

Example usage:
$ python3 create_vm.py --project_id="high-performance-ml" --vm_name="my-vm"
"""


def scan_zones(project_id: str, gpu_type: str, num_zones_to_check: int) -> List[Dict]:
    res = []
    zones = compute_v1.ZonesClient().list(project=project_id)

    for i, zone in enumerate(
        tqdm(
            zones, total=num_zones_to_check, desc="checking zones for GPU availability"
        )
    ):
        if i == num_zones_to_check:
            break

        request = compute_v1.ListAcceleratorTypesRequest(
            project=project_id, zone=zone.name
        )
        response = compute_v1.AcceleratorTypesClient().list(request=request)

        for gpu in response:
            if gpu.name == gpu_type:
                res.append(
                    {
                        "region": "-".join(gpu.zone.split("/")[-1:][0].split("-")[:-1]),
                        "zone": gpu.zone.split("/")[-1:][0],
                        "gpu_type": gpu.name,
                    }
                )
                break

    return res


def create_single_vm(
    region: str,
    zone: str,
    project_id: str,
    instance_name: str,
    gpu_type: str,
    gpu_count: int,
    machine_type: str,
    disk_source_image: str,
    disk_size: int,
) -> None:
    instance_client = compute_v1.InstancesClient()

    # GPU configuration
    accelerator_config = compute_v1.AcceleratorConfig()
    accelerator_config.accelerator_count = gpu_count
    accelerator_config.accelerator_type = (
        f"projects/{project_id}/zones/{zone}/acceleratorTypes/{gpu_type}"
    )

    # disk configuration
    disk = compute_v1.AttachedDisk(auto_delete=True, boot=True)
    disk.initialize_params = compute_v1.AttachedDiskInitializeParams(
        source_image=disk_source_image,
        disk_size_gb=disk_size,
        disk_type=f"projects/{project_id}/zones/{zone}/diskTypes/pd-balanced",
    )

    # network configuration
    network_interface = compute_v1.NetworkInterface()
    access_config = compute_v1.AccessConfig()
    access_config.name = "External NAT"
    access_config.type_ = "ONE_TO_ONE_NAT"
    network_interface.access_configs = [access_config]
    network_interface.stack_type = "IPV4_ONLY"
    network_interface.subnetwork = (
        f"projects/{project_id}/regions/{region}/subnetworks/default"
    )

    # base instance configuration
    instance = compute_v1.Instance(
        name=instance_name,
        machine_type=f"projects/{project_id}/zones/{zone}/machineTypes/{machine_type}",
        guest_accelerators=[accelerator_config],
        scheduling=compute_v1.Scheduling(
            automatic_restart=True,
            on_host_maintenance="TERMINATE",
            provisioning_model="STANDARD",
        ),
        disks=[disk],
        network_interfaces=[network_interface],
    )

    request = compute_v1.InsertInstanceRequest()
    request.zone = zone
    request.project = project_id
    request.instance_resource = instance

    # wait for the create operation to complete.
    operation = instance_client.insert(request=request)
    operation.result(timeout=300)

    return None


def create_all_vm(
    available_zones: List[Dict],
    project_id: str,
    vm_name: str,
    gpu_type: str,
    gpu_count: int,
    machine_type: str,
    disk_source_image: str,
    disk_size: int,
) -> None:
    for zone_dict in tqdm(
        available_zones,
        total=len(available_zones),
        desc="instantiate VM in first available zones",
    ):
        instance_name = f"{vm_name}"
        region = zone_dict["region"]
        zone = zone_dict["zone"]

        try:
            create_single_vm(
                region,
                zone,
                project_id,
                instance_name,
                gpu_type,
                gpu_count,
                machine_type,
                disk_source_image,
                disk_size,
            )
            print(f"instantiated VM {instance_name} currently running in {zone} 🥳")
            break

        except Exception as e:
            print(e)


def main(
    gpu_type: str = "nvidia-tesla-t4",
    gpu_count: int = 1,
    machine_type: str = "n1-standard-8",
    disk_source_image: str = "projects/ml-images/global/images/c0-deeplearning-common-cu121-v20240306-debian-11",
    disk_size: int = 100,
    num_zones_to_check: int = 30,
    *,
    project_id: str,
    vm_name: str,
) -> None:
    """
    Main function to create and run a single GCP instance.

    :param num_zones_to_check: number of zones to check for GPU availability
    :param project_id: GCP project ID, ie. "high-performance-ml"
    :param vm_name: the name for VM instance
    :param gpu_type: type of GPU to use
    :param gpu_count: number of GPUs to attach to the VM instance
    :param machine_type: the machine type for the instances
    :param disk_source_image: the source image for the boot disk
    :param disk_size: the size of the boot disk in GB
    """

    assert project_id != ""
    assert vm_name != ""

    available_zones = scan_zones(project_id, gpu_type, num_zones_to_check)

    # will only create and run one VM instance
    create_all_vm(
        available_zones,
        project_id,
        vm_name,
        gpu_type,
        gpu_count,
        machine_type,
        disk_source_image,
        disk_size,
    )


if __name__ == "__main__":
    fire.Fire(main)
