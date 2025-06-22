import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import streamlit as st
import matplotlib.pyplot as plt


def relu(x):
    return np.maximum(x, 0)


def drelu(x):
    return (x > 0).astype(float)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(x):
    x2 = sigmoid(x)
    return x2 * (1 - x2)


def softmax(z):
    z_shifted = z - np.max(z, axis=1, keepdims=True)  # prevent overflow
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def squared_loss(output, labels):
    return np.mean((labels - output) ** 2) / 2


def binary_cross_entropy(output, labels):
    eps = 1e-12
    out = np.clip(output, eps, 1 - eps)
    loss = -np.mean(labels * np.log(out) + (1 - labels) * np.log(1 - out))
    return loss


def cross_entropy(y_true, y_pred):
    eps = 1e-8
    return -np.mean(np.sum(y_true * np.log(y_pred + eps), axis=1))


class NN:
    def __init__(self, n1, n2, n3):
        self.weight_input_hidden = np.random.randn(n1, n2) * np.sqrt(2. / n1)
        self.weight_hidden_output = np.random.randn(n2, n3) * np.sqrt(2. / n2)

        self.bias_hidden = np.zeros((1, n2))
        self.bias_output = np.zeros((1, n3))

    def evaluate(self, data, labels):
        prediction = self.forward(data)[3]
        return np.mean((prediction > 0.5).astype(int) == labels)

    def forward(self, data):
        Z1 = np.matmul(data, self.weight_input_hidden) + self.bias_hidden
        A1 = relu(Z1)
        Z2 = np.matmul(A1, self.weight_hidden_output) + self.bias_output
        A2 = softmax(Z2)
        return Z1, A1, Z2, A2

    def backward(self, data, labels, stuff):
        m = data.shape[0]
        dz2 = stuff[3] - labels
        dw2 = stuff[1].T @ dz2 / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        da1 = dz2 @ self.weight_hidden_output.T
        dz1 = da1 * drelu(stuff[0])
        dw1 = data.T @ dz1 / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        return dw1, db1, dw2, db2

    def update_parameters(self, gradient, learning_rate=0.01):
        self.weight_input_hidden -= gradient[0] * learning_rate
        self.bias_hidden -= gradient[1] * learning_rate

        self.weight_hidden_output -= gradient[2] * learning_rate
        self.bias_output -= gradient[3] * learning_rate

    def gradient_descent(self, data, labels, learning_rate=0.01):
        f = self.forward(data)

        gradient = self.backward(data, labels, f)

        self.update_parameters(gradient, learning_rate)
        print(cross_entropy(f[3], labels), np.mean((f[3] > 0.5).astype(int) == labels))

    def stochastic_gradient_descent(self, data, labels, batch_size=32, epochs=1000):
        m = data.shape[0]
        accuracy = []

        for epoch in range(epochs):
            lr = 0.05 * (0.98 ** (epoch // (epochs // 100 + 1)))
            indices = np.random.permutation(m)
            data_shuffled = data[indices]
            labels_shuffled = labels[indices]

            for i in range(0, m, batch_size):
                data_batch = data_shuffled[i:i+batch_size]
                labels_batch = labels_shuffled[i:i+batch_size]

                f = self.forward(data_batch)
                gradient = self.backward(data_batch, labels_batch, f)
                self.update_parameters(gradient, lr)

            A2 = self.forward(data)[3]
            accuracy.append(np.mean((A2 > 0.5).astype(int) == labels))

            if accuracy[-1] == 1:
                break

        return accuracy


def main():
    st.set_page_config(
        page_title="Neural Network Configurator",
        layout="centered",
        initial_sidebar_state="auto"
    )

    st.title("⚙️ Neural Network Configuration")
    st.markdown("""
    Welcome to this interactive configurator for a neural network's training parameters!
    Adjust the sliders below to set the number of hidden nodes, batch size, and number of epochs.
    """)

    st.header("Training Parameters")

    # Slider for Number of Hidden Nodes
    num_hidden_nodes = st.slider(
        "Number of Hidden Nodes",
        min_value=1,
        max_value=100,
        value=48,
        step=1,
        help="Determines the complexity and capacity of the hidden layer in the neural network."
    )

    # Slider for Batch Size
    batch_size = st.slider(
        "Batch Size",
        min_value=1,
        max_value=256,
        value=32,
        step=1,
        help="Number of training examples utilized in one batch."
    )

    # Slider for Number of Epochs
    num_epochs = st.slider(
        "Number of Epochs",
        min_value=1,
        max_value=1000,
        value=50,
        step=1,
        help="Number of complete passes through the entire training dataset."
    )

    st.markdown("---")

    # Display selected parameters
    st.subheader("Selected Configuration:")
    st.write(f"**Hidden Nodes:** `{num_hidden_nodes}`")
    st.write(f"**Batch Size:** `{batch_size}`")
    st.write(f"**Epochs:** `{num_epochs}`")

    st.markdown("---")

    # Button to start training
    if st.button("Start Neural Network Training"):
        st.success("Neural network training initiated!")
        st.write("Training with the following parameters:")
        st.write(f"- Hidden Nodes: `{num_hidden_nodes}`")
        st.write(f"- Batch Size: `{batch_size}`")
        st.write(f"- Epochs: `{num_epochs}`")

        network = NN(64, num_hidden_nodes, 10)
        X, y = load_digits(return_X_y=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        y_train = np.eye(10)[y_train]
        y_test = np.eye(10)[y_test]
        accuracy_training = network.stochastic_gradient_descent(X_train, y_train, batch_size, num_epochs)
        accuracy_test = network.evaluate(X_test, y_test)

        st.write(f"Training completed. Training accuracy: `{accuracy_training[-1]}` and validation accuracy: `{accuracy_test}`")

        st.header("Accuracy over time")

        # --- Create the Matplotlib plot ---
        fig, ax = plt.subplots()  # Create a figure and an axes
        ax.plot(list(range(len(accuracy_training))), accuracy_training, label='Training Loss')
        ax.set_xlabel("Number of epochs")
        ax.set_ylabel("Accuracy")
        ax.set_title("Simulated Training Accuracy Over Time")
        ax.legend()
        ax.grid(True)

        # --- Display the plot in Streamlit ---
        st.pyplot(fig)

    st.markdown("---")
    st.markdown("### How this works:")
    st.markdown("""
    This application uses Streamlit to create an interactive user interface for
    neural network training parameters. You can adjust the sliders to explore
    different common configurations. In a real application, these values would
    be used to initialize and train your neural network model.
    """)

    st.markdown("---")
    st.caption("Built with ❤️ using Streamlit")


main()

