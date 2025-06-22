import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import streamlit as st
import matplotlib.pyplot as plt
import io


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


def float_to_string(f, max_value=16):
    return [str(round((val / max_value), 2)) for val in f]


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

    def stochastic_gradient_descent(self, data, labels, batch_size=32, epochs=1000, progress_callback=None, plot_callback=None):
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

            # Update progress bar
            if progress_callback:
                progress_callback(int((epoch / epochs) * 100))

            # Update the plot
            if plot_callback:
                plot_callback(self.weight_input_hidden, self.weight_hidden_output)

            # Add a small delay to visualize changes
            # time.sleep(0.05)

            if accuracy[-1] == 1:
                break

        return accuracy


def draw_neural_network(num_input, num_hidden, num_output, c1, c2, c3):
    """
    Draws a simplified neural network architecture with input, hidden, and output layers.
    Weights are randomly generated and colored/thickened based on their strength (absolute value)
    and sign (positive/negative).
    """
    bgcolor = "#0E1117"
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor(bgcolor)
    ax.set_aspect('equal')
    ax.set_facecolor(bgcolor)  # Light background matching Streamlit

    # Node radii for drawing
    node_radius = 0.01

    # Positions for nodes
    # Input layer at x=0, Hidden at x=1, Output at x=2
    layer_x_coords = [0, 1, 2]

    # Calculate Y-coordinates for nodes in each layer
    node_span = 0.8
    input_y = np.linspace(node_span, -node_span, num_input) if num_input > 1 else [0]
    hidden_y = np.linspace(node_span, -node_span, num_hidden) if num_hidden > 1 else [0]
    output_y = np.linspace(node_span, -node_span, num_output) if num_output > 1 else [0]

    # Store node positions
    input_nodes = [(layer_x_coords[0], y) for y in input_y]
    hidden_nodes = [(layer_x_coords[1], y) for y in hidden_y]
    output_nodes = [(layer_x_coords[2], y) for y in output_y]

    all_nodes = {
        'input': input_nodes,
        'hidden': hidden_nodes,
        'output': output_nodes
    }
    # Draw nodes
    node_facecolor = 'white'
    node_edgecolor = 'white'
    node_zorder = 2

    for i, (x, y) in enumerate(input_nodes):
        circle = plt.Circle((x, y), node_radius, fc=c1[i], ec=c1[i], zorder=node_zorder)
        ax.add_patch(circle)
        # ax.text(x, y, f'I{i+1}', ha='center', va='center', color='black', fontsize=9, weight='bold')

    for i, (x, y) in enumerate(hidden_nodes):
        circle = plt.Circle((x, y), node_radius, fc=c2[i], ec=c2[i], zorder=node_zorder)
        ax.add_patch(circle)
        # ax.text(x, y, f'H{i+1}', ha='center', va='center', color='black', fontsize=9, weight='bold')

    for i, (x, y) in enumerate(output_nodes):
        circle = plt.Circle((x, y), node_radius, fc=c3[i], ec=c3[i], zorder=node_zorder)
        ax.add_patch(circle)
        ax.text(x + 0.1, y, f'{i}', ha='center', va='center', color='white', fontsize=9)

    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(-1, 1)
    ax.axis('off')  # Hide axes

    ax.set_title("Neural Network Architecture Visualization", fontsize=14, pad=20, color="white")

    return fig


def main():
    st.set_page_config(
        page_title="Neural Network Configurator",
        layout="wide",
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
        value=16,
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

    # Slider for training/test split

    test_size = st.slider(
        "Training/test split",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.01,
        help="Percentage of data points used for validation, remaining data points are used for training."
    )

    st.markdown("---")

    # Display selected parameters
    st.subheader("Selected Configuration:")
    st.write(f"**Hidden Nodes:** `{num_hidden_nodes}`")
    st.write(f"**Batch Size:** `{batch_size}`")
    st.write(f"**Epochs:** `{num_epochs}`")
    st.write(f"**Training/test split:** `{test_size}`")

    # Initialize session state for the current image index
    if 'image_index' not in st.session_state:
        st.session_state.image_index = 0
    if 'trained_network' not in st.session_state:
        st.session_state.trained_network = None

    # Display area for the plot
    # plot_placeholder = st.empty()

    # Load data set
    X, y = load_digits(return_X_y=True)
    digits = load_digits()
    images = digits.images
    targets = digits.target

    num_input_nodes = 64
    num_output_nodes = 10
    network2 = NN(num_input_nodes, num_hidden_nodes, num_output_nodes)

    st.header("Current architecture")

    # Button to show the next image
    col11, col21, col31 = st.columns([0.19, 0.1, 0.71])
    with col21:
        if st.button("Next"):
            # Increment index, loop back to 0 if at the end of the dataset
            st.session_state.image_index = (st.session_state.image_index + 1) % len(images)
            # Trigger a rerun to update the image
            st.rerun()
    with col11:
        if st.button("Previous"):
            # Decrement index
            st.session_state.image_index = (st.session_state.image_index - 1) % len(images)
            # Trigger a rerun to update the image
            st.rerun()

    # --- Display the current image ---
    current_index = st.session_state.image_index
    current_image = images[current_index]
    current_target = targets[current_index]

    # Create a Matplotlib figure for the 8x8 digit image
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)  # Small figure size for 8x8 image
    ax.imshow(current_image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(f"Digit: {current_target} (Image {current_index + 1}/{len(images)})", color='white')
    ax.set_axis_off()  # Hide axes for cleaner image display

    # Set background colors to match dark mode (assuming default Streamlit dark mode)
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # Save figure to buffer
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight')  # Optional: bbox_inches
    buf.seek(0)

    # Create two columns
    col1, col2 = st.columns(2)

    # Display digit
    with col1:
        st.image(buf)

    # Display the plot in Streamlit
    # plot_placeholder.pyplot(fig)
    # plt.close(fig)  # Close the figure to free up memory

    # Placeholders for dynamic content
    # plot_placeholder2 = st.empty()

    # Draw and display the network plot
    # cs2 = [str(i / num_hidden_nodes) for i in range(1, num_hidden_nodes + 1)]
    # cs3 = [str(i / num_output_nodes) for i in range(1, num_output_nodes + 1)]

    network_output = network2.forward(X)
    a1 = network_output[1]
    a2 = network_output[3]
    if st.session_state.trained_network:
        st.write("111111111")
        network_output = st.session_state.trained_network.forward(X)
        a1 = network_output[1]
        a2 = network_output[3]
    nn_fig = draw_neural_network(num_input_nodes, num_hidden_nodes, num_output_nodes,
                                 float_to_string(X[st.session_state.image_index, :]),
                                 float_to_string(a1[st.session_state.image_index, :],
                                                 np.max(a1[st.session_state.image_index, :])),
                                 float_to_string(a2[st.session_state.image_index, :],
                                                 np.max(a2[st.session_state.image_index, :])))
    with col2:
        st.pyplot(nn_fig, use_container_width=False)
        plt.close(nn_fig)  # Close the figure to free up memory after displaying

    st.markdown("---")

    # Button to start training
    if st.button("Start Neural Network Training"):
        st.success("Neural network training initiated!")
        st.write("Training with the following parameters:")
        st.write(f"- Hidden Nodes: `{num_hidden_nodes}`")
        st.write(f"- Batch Size: `{batch_size}`")
        st.write(f"- Epochs: `{num_epochs}`")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        y_train = np.eye(10)[y_train]
        y_test = np.eye(10)[y_test]
        network = NN(num_input_nodes, num_hidden_nodes, num_output_nodes)
        accuracy_training = network.stochastic_gradient_descent(X_train, y_train, batch_size, num_epochs)
        accuracy_test = network.evaluate(X_test, y_test)

        st.session_state.trained_network = network

        st.write(f"Training completed. Training accuracy: `{accuracy_training[-1]}` and validation accuracy: `{accuracy_test}`")

        st.header("Accuracy over time")

        # --- Create the Matplotlib plot ---
        fig2, ax2 = plt.subplots(figsize=(5, 3.5))  # Create a figure and an axes
        ax2.plot(list(range(len(accuracy_training))), accuracy_training, label='Training Loss')
        ax2.set_xlabel("Number of epochs")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Simulated Training Accuracy Over Time")
        ax2.legend()
        ax2.grid(True)

        # --- Display the plot in Streamlit ---
        st.pyplot(fig2, use_container_width=False)
        plt.close(fig2)

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

