import matplotlib.pyplot as plt


def plot_quantize(state_dict, ax, model_name, bits_to_try, start):

    plt.style.use("default")
    unquantized_acc = [100 * state_dict[model_name]["test_acc"]] * len(
        bits_to_try[start:]
    )
    quantized_acc = [
        100 * acc for acc in state_dict[model_name]["test_acc_po2"][start:]
    ]
    quantized_plus_acc = [
        100 * acc for acc in state_dict[model_name]["test_acc_po2+"][start:]
    ]

    ax.plot(
        bits_to_try[start:],
        unquantized_acc,
        label="float32",
        color="purple",
        linestyle="--",
    )
    ax.plot(bits_to_try[start:], quantized_acc, label="quantized", color="orange")
    ax.plot(bits_to_try[start:], quantized_plus_acc, label="quantized+", color="green")

    ax.set_xlabel("Bits Used")
    ax.set_ylabel("Percent")
    ax.set_title(f"{model_name}")
    ax.set_xticks(bits_to_try[start:])
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    ax.set_ylim(70, 95)
    ax.legend()


# Commented these out because of a bug
# fig, axs = plt.subplots(2, 2, figsize=(10, 6), dpi=300)
# for i, model_name in enumerate(state_dict.keys()):
#     plot_quantize(axs[i // 2, i % 2], model_name, bits_to_try, 1)

# for ax in axs[0, :]:
#     ax.set_xlabel('')

# for ax in axs[:, 1]:
#     ax.set_ylabel('')

# plt.tight_layout()
# plt.savefig(f'{dir}/quantization.png', bbox_inches='tight')


def plot_improvement(state_dict, ax, model_name, bits_to_try, start):

    plt.style.use("default")
    improvement = [100 * acc for acc in state_dict[model_name]["test_acc_po2+"][start:]]

    ax.plot(
        bits_to_try[start:],
        improvement,
        label="quantized+ improvement",
        color="orange",
    )

    ax.set_xlabel("Bits Used")
    ax.set_ylabel("Percent")
    ax.set_title(f"{model_name}")
    ax.set_xticks(bits_to_try[start:])
    ax.grid(axis="y", linestyle="--", linewidth=0.5)
    ax.set_ylim(-3, 14)
    ax.legend()


# Commented these out because of a bug
# fig, axs = plt.subplots(2, 2, figsize=(10, 6), dpi=300)
# for i, model_name in enumerate(state_dict.keys()):
#     plot_quantize(axs[i // 2, i % 2], model_name, bits_to_try, 1)

# for ax in axs[0, :]:
#     ax.set_xlabel('')

# for ax in axs[:, 1]:
#     ax.set_ylabel('')

# plt.tight_layout()
# plt.savefig(f'{dir}/percent_improvement.png', bbox_inches='tight')

if __name__ == "__main__":

    