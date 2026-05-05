import matplotlib.pyplot as plt


def plot_history(history):
    epochs = range(len(history["loss_train"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, history["loss_train"], label='Training Loss')
    ax1.plot(epochs, history["loss_val"], '--', label='Validation Loss')
    ax1.set_title('Loss History')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(epochs, history["acc_train"], label='Training Acc')
    ax2.plot(epochs, history["acc_val"], label='Validation Acc')
    ax2.set_title('Learning Curves')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()

    plt.show()