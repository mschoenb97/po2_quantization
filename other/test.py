from models import resnet20

model = resnet20()


    # PTQ: Post Training Quantization

    for model_name, model_dict in state_dict.items():

        if not model_dict["is_quantized"]:
            continue

        print(f"Quantizing {model_name} to {bits_to_try} bits")

        model = model_dict["model"]
        test_acc_po2 = quantize_loop(
            state_dict, model_name, PowerOfTwoQuantizer, testloader
        )
        test_acc_po2_plus = quantize_loop(
            state_dict, model_name, PowerOfTwoPlusQuantizer, testloader
        )

        model_dict["test_acc_po2"] = test_acc_po2
        model_dict["test_acc_po2+"] = test_acc_po2_plus

        model_dict["improvement"] = [
            (lambda new, old: (new - old) / old if old != 0 else 0)(new, old)
            for new, old in zip(test_acc_po2_plus, test_acc_po2)
        ]

        test_acc = model_dict["test_acc"]
        test_acc_po2 = " ".join([str(round(100 * acc, 2)) for acc in test_acc_po2])
        test_acc_po2_plus = " ".join(
            [str(round(100 * acc, 2)) for acc in test_acc_po2_plus]
        )
        improvement = " ".join(
            [str(round(100 * imp, 2)) for imp in model_dict["improvement"]]
        )
        print(f"\t{'test_acc':<15} = {test_acc}")
        print(f"\t{'test_acc_po2':<15} = {test_acc_po2}")
        print(f"\t{'test_acc_po2+':<15} = {test_acc_po2_plus}")
        print(f"\t{'improvement':<15} = {improvement}")



def compute_test():
    
    correct, total = 0, 0
    criterion = nn.CrossEntropyLoss()
    model.eval()

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = correct / total


def quantize_loop(
    state_dict: Dict[str, Dict[str, Union[nn.Module, List[int], str]]],
    model_name: str,
    quantizer: Callable,
    testloader: DataLoader,
) -> List[float]:

    fp_model_name = state_dict[model_name]["fp_model"]
    model = state_dict[fp_model_name]["model"]
    bits = state_dict[model_name]["bits"]

    accuracies = []
    model_copy = deepcopy(model)
    quantize_model(model_copy, quantizer, 1, bits)
    accuracies.append(compute_test(model_copy, testloader))

    return accuracies


def quantize_model(model, quantizer, fsr: int = 1, bits: int = 4):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "conv" in name and "layer" in name:
                quantized_param = quantizer.forward(None, param, bits=bits)
                param.copy_(quantized_param)


def quantize_loop(model: nn.Module, bits_to_try: List[int], quantizer):
    accuracies = []
    for bit in bits_to_try:
        model_copy = deepcopy(model)
        quantize_model(model_copy, quantizer, 1, bit - 1)
        accuracies.append(compute_test(model_copy))

    return accuracies
