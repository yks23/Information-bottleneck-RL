import matplotlib.pyplot as plt


def plot_binary_entropy_curve(xs, ys, out_path):
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys)
    plt.xlabel("p(success)")
    plt.ylabel("H(O)")
    plt.title("Binary Outcome Entropy")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
