import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
import seaborn as sns

# Function to initialize weights
def Q1_initialize_weights(L_pre, L_post):
    w0 = np.sqrt(6 / (L_pre + L_post))
    
    W1 = np.random.uniform(-w0, w0, size=(L_post, L_pre))
    W2 = np.random.uniform(-w0, w0, size=(L_pre, L_post))
    b1 = np.random.uniform(-w0, w0, size=(L_post, 1))
    b2 = np.random.uniform(-w0, w0, size=(L_pre, 1))
    
    We = (W1, W2, b1, b2)
    return We


def sigmoid(X):
    A = 1 / (1 + np.exp(-X))
    dA = A * (1 - A)
    return A, dA


# Function to calculate cost and its partial derivatives
def Q1_aeCost(We, data, params):
    N = data.shape[0]

    W1, W2, b1, b2 = We

    rho = params["rho"]
    lambda_ = params["lambda"]
    beta = params["beta"]

    # Propagate forward
    Z1 = np.dot(W1, data.T) + b1
    A1, dA1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2, dA2 = sigmoid(Z2)

    # Calculate the cost
    rho_hat = np.mean(A1, axis=1, keepdims=True)
    MSE = (1 / (2 * N)) * np.sum((data.T - A2) ** 2)
    tykhonov = (lambda_ / 2) * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    kl_div = beta * np.sum(
        rho * np.log(rho / rho_hat) + (1 - rho) * np.log((1 - rho) / (1 - rho_hat))
    )
    J = MSE + tykhonov + kl_div

    # Propagate backward
    dJ_A2 = (-1 / N) * (data.T - A2)
    dJ_Z2 = dJ_A2 * dA2
    dJ_A1 = np.dot(W2.T, dJ_Z2) + beta * (
        -(rho / rho_hat) + (1 - rho) / (1 - rho_hat)
    )
    dJ_Z1 = dJ_A1 * dA1

    dJ_W1 = np.dot(dJ_Z1, data) + (lambda_ * W1)
    dJ_W2 = np.dot(dJ_Z2, A1.T) + (lambda_ * W2)
    dJ_b1 = np.sum(dJ_Z1, axis=1, keepdims=True)
    dJ_b2 = np.sum(dJ_Z2, axis=1, keepdims=True)
    J_grad = dJ_W1, dJ_W2, dJ_b1, dJ_b2
    
    # Return the cost and its derivatives
    return J, J_grad


# Function to update weights
def Q1_update_weights(We, J_grad, We_momentum, learning_rate, momentum_rate):
    W1, W2, b1, b2 = We
    dJ_W1, dJ_W2, dJ_b1, dJ_b2 = J_grad
    W1_momentum, W2_momentum, b1_momentum, b2_momentum = We_momentum

    # Calculate the deltas
    dW1 = (dJ_W1 * learning_rate) + (W1_momentum * momentum_rate)
    dW2 = (dJ_W2 * learning_rate) + (W2_momentum * momentum_rate)
    db1 = (dJ_b1 * learning_rate) + (b1_momentum * momentum_rate)
    db2 = (dJ_b2 * learning_rate) + (b2_momentum * momentum_rate)

    # Update the weights
    W1 -= dW1
    W2 -= dW2
    b1 -= db1
    b2 -= db2

    We = W1, W2, b1, b2
    We_momentum = dW1, dW2, db1, db2
    
    # Return the updated weights and momentums
    return We, We_momentum


# Function to perform gradient descent
def Q1_gradient_descent(
    We, data, params, epoch_count, batch_size, learning_rate, momentum_rate,
):

    N = data.shape[0]
    We_momentum = (0, 0, 0, 0)
    mini_batch_count = int(N / batch_size)
    costs = []

    # Iterate over the number of epochs
    for epoch in range(1, epoch_count + 1):
        J_total = 0

        # Shuffle the data
        np.random.shuffle(data)

        # Split the data into mini-batches
        mini_batches = np.split(data, mini_batch_count)

        # Iterate over each mini-batch
        for mini_batch in mini_batches:
            J, J_grad = Q1_aeCost(We, mini_batch, params)
            We, We_momentum = Q1_update_weights(
                We, J_grad, We_momentum, learning_rate, momentum_rate
            )
            J_total += J

        # Calculate the average cost for this epoch
        costs.append(J_total / mini_batch_count)

        # Print out the progress
        if ((epoch%10)==0):
            print(f"Epoch: {epoch}, Training Error: {round(costs[-1], 5)}")

    # Return the weights and the costs
    return We, costs


def Q1_plot_weights(We, params, rows, cols, fig=None, axs=None, save=False):
    if axs is None:
        # Create a figure and axes for plotting weights
        fig, axs = plt.subplots(
            rows, cols, figsize=(cols, rows)
        )  # Adjust the spacing between subplots
        fig.subplots_adjust(wspace=0.05, hspace=0.05, left=0, right=1, bottom=0, top=1)

    for i in range(rows):
        for j in range(cols):
            # Plot the weights of each neuron
            axs[i, j].imshow(We[0][i * rows + j].reshape(16, 16), cmap="gray")
            axs[i, j].axis("off")

    if save:
        # Save the figure
        fig.savefig(
            f"Q1_images/weights_Lhid{params['Lhid']}_lambda{str(params['lambda']).replace('.','')}_beta{str(params['beta']).replace('.', '')}_rho{str(params['rho']).replace('.','')}"
        )
        
        
def Q1():
    # [1] Load dataset
    with h5py.File("data1.h5") as hf:
        data_raw = np.array(hf["data"])
    
    # [2] Convert to greyscale and Normalize
    data = (
        0.2126 * data_raw[:, 0] + 0.7152 * data_raw[:, 1] + 0.0722 * data_raw[:, 2]
    )

    data = np.reshape(
        data, (data.shape[0], data.shape[1] * data.shape[2])
    )
    data = data - data.mean(axis=1, keepdims=True)
    std = np.std(data)
    data = np.clip(data, -3 * std, +3 * std)
    data = 0.1 + (data - data.min()) / (data.max() - data.min()) * 0.8

    print(data.shape)
    print(data.min(), data.max())
    
    
    # [3] Restructure data arrays for imshow
    data_gray = np.reshape(data, (data.shape[0], 16, 16))
    data_rgb = data_raw.transpose((0, 2, 3, 1))

    figure_rgb, axes_rgb = plt.subplots(20,10, figsize=(10, 20))
    figure_gray, axes_gray = plt.subplots(
        20, 10, figsize=(10, 20), dpi=200, facecolor="w", edgecolor="k"
    )

    idxs = np.random.choice(np.arange(data_gray.shape[0]), 200, replace=False)
    data_rgb_rand = data_rgb[idxs]
    data_gray_rand = data_gray[idxs]
    for i in range(20):
        for j in range(10):
            axes_rgb[i, j].imshow(data_rgb_rand[i * 10 + j].astype("float64"))
            axes_rgb[i, j].axis("off")

            axes_gray[i, j].imshow(data_gray_rand[i * 10 + j], cmap="gray")
            axes_gray[i, j].axis("off")

    figure_rgb.subplots_adjust(wspace=0.05, hspace=0.05, left=0, right=1, bottom=0, top=1)
    figure_gray.subplots_adjust(wspace=0.05, hspace=0.05, left=0, right=1, bottom=0, top=1)

    figure_rgb.savefig("Q1_images/Q1a_rgb.png")
    figure_gray.savefig("Q1_images/Q1a_gray.png")
    plt.close("all")
    
    
    # [4]
    params = {"Lin": 256, "Lhid": 64, "lambda": 5e-4, "beta": 0.05, "rho": 0.05}
    We = Q1_initialize_weights(params["Lin"], params["Lhid"])
    We_final, costs = Q1_gradient_descent(We, data, params, 500, 32, 0.001, 0.95)
    Q1_plot_weights(We_final, params, 8, 8, save=True)
    
    
    # [5]
    Lhids = (16, 49, 81)
    lambdas = (0, 1e-4, 1e-3)

    fig = plt.figure(figsize=(50, 50))
    subfigs = fig.subfigures(len(Lhids), len(lambdas), wspace=0, hspace=0)

    for Lhid_idx in range(len(Lhids)):
        for lambda_idx in range(len(lambdas)):
            Lhid = Lhids[Lhid_idx]
            lambda_ = lambdas[lambda_idx]

            print(f"\nLhid: {Lhid}, lambda: {lambda_}")
            params = {
                "Lin": 256,
                "Lhid": Lhid,
                "lambda": lambda_,
                "beta": 0.05,
                "rho": 0.02,
            }
            We = Q1_initialize_weights(params["Lin"], params["Lhid"])
            (
                We_final,
                costs,
            ) = Q1_gradient_descent(We, data, params, 500, 32, 0.01, 0.95)
            

            rows = cols = int(np.sqrt(Lhid))
            subfig = subfigs[Lhid_idx][lambda_idx]
            subfig.suptitle(r"$L_{hid}$:" + str(Lhid)+ r", $\lambda$:" + str(lambda_), va='top', fontsize=70)
            axs = subfigs[Lhid_idx][lambda_idx].subplots(rows, cols, gridspec_kw={"wspace": 0.0625, "hspace": 0.0625})
            Q1_plot_weights(We_final, params, rows, cols, axs=axs)

    # fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    fig.savefig("Q1_images/Lhid and Lambda")


# Function to one hot encode Y values for Q2
def Q2_one_hot_encode(Y):
    return np.eye(250)[Y.ravel() - 1]


def softmax(X: np.ndarray):
    exps = np.exp(X - np.amax(X, axis=0))
    A = exps / np.sum(exps, axis=0)
    return A


# Initialize weights for Q2
def Q2_initialize_weights(D, P):
    WE = np.random.normal(0, 0.01, (250, D))
    W1 = np.random.normal(0, 0.01, (P, 3 * D))
    W2 = np.random.normal(0, 0.01, (250, P))
    b1 = np.random.normal(0, 0.01, (P, 1))
    b2 = np.random.normal(0, 0.01, (250, 1))
    We = WE, W1, W2, b1, b2
    return We


# Apply Word Embedding
def Q2_words_to_vector(x_train, word_embedding):
    N = x_train.shape[0]
    D = word_embedding.shape[1]
    input_vector = np.zeros((N, 3 * D))
    for i in range(N):
        word1 = word_embedding[x_train[i, 0] - 1]
        word2 = word_embedding[x_train[i, 1] - 1]
        word3 = word_embedding[x_train[i, 2] - 1]

        input_vector[i] = np.concatenate((word1, word2, word3)).T

    return input_vector


# Simple Cross Entropy
def cross_entropy(Y_true, Y_pred):
    return -np.sum(Y_true * np.log(Y_pred)) / Y_pred.shape[1]


# Calculate Gradients for Q2
def Q2_calculate_gradients(We, X, Y):
    WE, W1, W2, b1, b2 = We

    # Propagate Forward
    A0 = Q2_words_to_vector(X, WE)
    Z1 = np.dot(W1, A0.T) + b1
    A1, dA1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)

    J = cross_entropy(Y.T, A2)

    # Propagate Backward
    dJ_Z2 = A2 - Y.T
    dJ_A1 = np.dot(W2.T, dJ_Z2)
    dJ_Z1 = dJ_A1 * dA1

    dJ_WE = np.dot(W1.T, dJ_Z1)
    dJ_W1 = np.dot(dJ_Z1, A0)
    dJ_W2 = np.dot(dJ_Z2, A1.T)
    dJ_b1 = np.sum(dJ_Z1, axis=1, keepdims=True)
    dJ_b2 = np.sum(dJ_Z2, axis=1, keepdims=True)

    J_grad = dJ_WE, dJ_W1, dJ_W2, dJ_b1, dJ_b2
    return J, J_grad


# Update Weights for Q2
def Q2_update_weights(
    We,
    J_grad,
    We_momentum,
    learning_rate,
    momentum_rate,
    x_train,
    batch_start,
    batch_end,
):
    WE, W1, W2, b1, b2 = We
    dJ_WE, dJ_W1, dJ_W2, dJ_b1, dJ_b2 = J_grad
    WE_momentum, W1_momentum, W2_momentum, b1_momentum, b2_momentum = We_momentum

    # Get the shape of the weights matrix
    word_count = WE.shape[0]
    D = WE.shape[1]

    # Initialize temp gradient matrix for word embeddings
    WE_grad = np.zeros((word_count, D))

    # Iterate through the training data
    triagram = dJ_WE[:, batch_start:batch_end].T

    # Split the triagram into three words
    dword1, dword2, dword3 = np.split(triagram, 3)

    # Update the gradients for each word
    WE_grad[x_train[:, 0] - 1, :] += dword1
    WE_grad[x_train[:, 1] - 1, :] += dword2
    WE_grad[x_train[:, 2] - 1, :] += dword3

    # Calculate the new weights deltas
    dWE = learning_rate * WE_grad + momentum_rate * WE_momentum
    dW1 = learning_rate * dJ_W1 + momentum_rate * W1_momentum
    dW2 = learning_rate * dJ_W2 + momentum_rate * W2_momentum
    db1 = learning_rate * dJ_b1 + momentum_rate * b1_momentum
    db2 = learning_rate * dJ_b2 + momentum_rate * b2_momentum

    # Update the weights
    WE -= dWE
    W1 -= dW1
    W2 -= dW2
    b1 -= db1
    b2 -= db2

    # Return the updated weights and momentum
    We = WE, W1, W2, b1, b2
    We_momentum = dWE, dW1, dW2, db1, db2
    return We, We_momentum


# Apply Gradient Descent for Q2
def Q2_gradient_descent(
    We: tuple,
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_validation: np.ndarray,
    Y_validation: np.ndarray,
    batch_size: int,
    epoch_count: int,
    learning_rate: float,
    momentum_rate: float,
    early_stop: bool = False,
    patience: int = 5,
    min_delta: float = 1e-5,
):
    # Initialize variables
    N = X_train.shape[0]
    mini_batch_count = int(N / batch_size)
    We_momentum = (0, 0, 0, 0, 0)

    cost_best = np.inf
    patience_counter = 0
    costs_train = []
    costs_validation = []

    # Iterate through epochs
    for epoch in range(epoch_count):
        mini_batch_start = 0
        mini_batch_end = batch_size

        sample_order = np.random.permutation(N)
        X_train = X_train[sample_order]
        Y_train = Y_train[sample_order]

        J_train_total = 0

        # Iterate through mini batches
        for _ in range(mini_batch_count):
            mini_batch_x = X_train[mini_batch_start:mini_batch_end]
            mini_batch_y = Y_train[mini_batch_start:mini_batch_end]

            J, J_grads = Q2_calculate_gradients(We, mini_batch_x, mini_batch_y)
            We, We_momentum = Q2_update_weights(
                We,
                J_grads,
                We_momentum,
                learning_rate,
                momentum_rate,
                X_train,
                mini_batch_start,
                mini_batch_end,
            )
            mini_batch_start = mini_batch_end
            mini_batch_end += batch_size
            mini_batch_end = min(mini_batch_end, N)

            J_train_total += J

        # Calculate training cost
        costs_train.append(J_train_total / mini_batch_count)

        # Calculate validation cost (Q2_calculate_gradients is fast enough)
        J_validation, _ = Q2_calculate_gradients(We, X_validation, Y_validation)
        costs_validation.append(J_validation)

        # Check for early stopping
        if len(costs_validation) > 1:
            if costs_validation[-1] + min_delta < cost_best:
                cost_best = costs_validation[-1]
                We_best = We
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"Early stop at epoch: {epoch+1}, Training Error: {round(costs_train[-1], 5)}, Validation Error: {round(costs_validation[-1], 5)}"
                    )
                    break

        # Print out progress
        print(
            f"Epoch: {epoch+1}, Training cost:{round(costs_train[-1],5)}, Validation cost: {round(J_validation,5)} "
        )

    return We_best, costs_train, costs_validation


# Get candidates for a set of X
def Q2_get_candidates(We, X, Y, words):
    WE, W1, W2, b1, b2 = We

    A0 = Q2_words_to_vector(X, WE)
    Z1 = np.dot(W1, A0.T) + b1
    A1, dA1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)

    sorted_Y_idxs = np.argmax(Y, axis=1)
    sorted_A2_idxs = np.flip(np.argsort(A2, axis=0), axis=0)
    random_samples = np.random.randint(0, A2.shape[1], size=(5))
    sorted_A2_ind_random_samples = sorted_A2_idxs[:, random_samples]
    y_random_samples = sorted_Y_idxs[random_samples]
    x_random_samples = X[random_samples]
    for i in range(5):
        print(
            f"{words[x_random_samples[i,0]-1].decode('UTF-8')} {words[x_random_samples[i,1]-1].decode('UTF-8')} {words[x_random_samples[i,2]-1].decode('UTF-8')} {[words[ind].decode('UTF-8') for ind in sorted_A2_ind_random_samples[:10,i].T]}"
        )
        

# Do Q2 things
def Q2():
    # [1]
    with h5py.File("data2.h5") as hf:
        X_train = np.array(hf["trainx"])
        Y_train = np.array(hf["traind"])
        X_validation = np.array(hf["valx"])
        Y_validation = np.array(hf["vald"])
        X_test = np.array(hf["testx"])
        Y_test = np.array(hf["testd"])
        words = np.array(hf["words"])
    print(X_train.shape)
    print(Y_train.shape)
    print(X_validation.shape)
    print(Y_validation.shape)
    print(X_test.shape)
    print(Y_test.shape)
    print(words.shape)
    
    
    # [2]
    print(words)
    
    
    # [3]
    Y_train = Q2_one_hot_encode(Y_train)
    Y_validation = Q2_one_hot_encode(Y_validation)
    Y_test = Q2_one_hot_encode(Y_test)
    print(Y_train.shape)
    print(Y_validation.shape)
    print(Y_test.shape)
    
    
    # [4]
    We = Q2_initialize_weights(32, 256)
    We_final, costs_training, costs_validation = Q2_gradient_descent(
        We, X_train, Y_train, X_validation, Y_validation, 200, 50, 0.001, 0.9
    )
    
    
    # [5]
    We = Q2_initialize_weights(16, 128)
    We_final2, costs_training2, costs_validation2 = Q2_gradient_descent(
        We, X_train, Y_train, X_validation, Y_validation, 200, 50, 0.001, 0.9
    )
    
    
    # [6]
    We = Q2_initialize_weights(8, 64)
    We_final3, costs_training3, costs_validation3 = Q2_gradient_descent(
        We, X_train, Y_train, X_validation, Y_validation, 200, 50, 0.001, 0.9
    )
    
    
    # [7]
    seed = np.random.randint(0, 1e5)

    np.random.seed(seed)
    Q2_get_candidates(We_final, X_test, Y_test, words)
    print()

    np.random.seed(seed)
    Q2_get_candidates(We_final2, X_test, Y_test, words)
    print()

    np.random.seed(seed)
    Q2_get_candidates(We_final3, X_test, Y_test, words)
    
    
    # [8]
    fig, axs = plt.subplots(1, 3, figsize=(20,5))
    fig.subplots_adjust(wspace=0.25)

    axs[0].plot(
        list(range(1, len(costs_training) + 1)),
        costs_training,
        color="blue",
        label="Training",
    )
    axs[0].plot(
        list(range(1, len(costs_validation) + 1)),
        costs_validation,
        color="red",
        label="Validation",
    )
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Cross Entropy")
    axs[0].set_title("Cross Entropy vs Epoch (32, 256)")
    axs[0].legend(loc="best")


    axs[1].plot(
        list(range(1, len(costs_training2) + 1)),
        costs_training2,
        color="blue",
        label="Training",
    )
    axs[1].plot(
        list(range(1, len(costs_validation2) + 1)),
        costs_validation2,
        color="red",
        label="Validation",
    )
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Cross Entropy")
    axs[1].set_title("Cross Entropy vs Epoch (16, 128)")
    axs[1].legend(loc="best")


    axs[2].plot(
        list(range(1, len(costs_training3) + 1)),
        costs_training3,
        color="blue",
        label="Training",
    )
    axs[2].plot(
        list(range(1, len(costs_validation3) + 1)),
        costs_validation3,
        color="red",
        label="Validation",
    )
    axs[2].set_xlabel("Epoch")
    axs[2].set_ylabel("Cross Entropy")
    axs[2].set_title("Cross Entropy vs Epoch (8, 64)")
    axs[2].legend(loc="best")


    fig.savefig("Q2_images/Loss")


# Simple hyperbolic tangent
def tanh(X: np.ndarray):
    A = np.tanh(X)
    dA = 1 - np.power(A, 2)
    return A, dA


#Simple rectified linear unit
def relu(X):
    A = X * (X > 0)
    dA = 1 * (X > 0)
    return A, dA


# Xavier Initialization
def Xavier_initialization(n_pre, n_post):
    w0 = np.sqrt(6 / (n_pre + n_post))
    W = np.random.uniform(-w0, w0, size=(n_post, n_pre))
    b = np.random.uniform(-w0, w0, size=(n_post, 1))
    return W, b


# Initialize weights for Q3a
def Q3a_initialize_weights(n_x: int, n_r: int, n_h: tuple[int, int], n_y: int):
    Wi, _ = Xavier_initialization(n_x, n_r)
    Wr, br = Xavier_initialization(n_r, n_r)
    W1, b1 = Xavier_initialization(n_r, n_h[0])
    W2, b2 = Xavier_initialization(n_h[0], n_h[1])
    Wo, bo = Xavier_initialization(n_h[1], n_y)

    We = {
        "Wi": Wi,
        "Wr": Wr,
        "W1": W1,
        "W2": W2,
        "Wo": Wo,
        "br": br,
        "b1": b1,
        "b2": b2,
        "bo": bo,
    }
    return We


# Forward Prop for Q3a
def Q3a_forward_propagation(We, data):
    Wi = We["Wi"]
    Wr = We["Wr"]
    W1 = We["W1"]
    W2 = We["W2"]
    Wo = We["Wo"]
    br = We["br"]
    b1 = We["b1"]
    b2 = We["b2"]
    bo = We["bo"]

    dim, time, samples = data.shape
    n_r = Wi.shape[0]

    # initialize state variables
    A0_ = np.zeros((n_r, time, samples))
    A0_prev = np.zeros((n_r, samples))
    dA0_ = np.zeros((n_r, time, samples))

    for t in range(time):
        current_data = data[:, t, :]
        Z = np.dot(Wi, current_data) + np.dot(Wr, A0_prev) + br
        A0_[:, t, :], dA0_[:, t, :] = tanh(Z)
        A0_prev = A0_[:, t, :]
    A0 = A0_[:, -1, :]  # final state

    # relu layers
    Z1 = np.dot(W1, A0) + b1
    A1, dA1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2, dA2 = relu(Z2)

    # softmax layer
    Z3 = np.dot(Wo, A2) + bo
    A3 = softmax(Z3)
    
    cache = {
        "A0_": A0_,
        "dA0_": dA0_,
        "Z1": Z1,
        "A1": A1,
        "dA1": dA1,
        "Z2": Z2,
        "A2": A2,
        "dA2": dA2,
        "Z3": Z3,
        "A3": A3,
    }
    return cache


# Calculate gradients for Q3a
def Q3a_calculate_gradients(We, x_train_, y_train):
    Wi = We["Wi"]
    Wr = We["Wr"]
    W1 = We["W1"]
    W2 = We["W2"]
    Wo = We["Wo"]
    br = We["br"]
    b1 = We["b1"]
    b2 = We["b2"]
    bo = We["bo"]

    # Propagate forward
    cache = Q3a_forward_propagation(We, x_train_)

    # Calculate training cost for this batch
    J = cross_entropy(cache["A3"], y_train)

    # Propagate backward
    dJ_Z3 = cache["A3"] - y_train
    dJ_A2 = np.dot(Wo.T, dJ_Z3)
    dJ_Z2 = dJ_A2 * cache["dA2"]
    dJ_A1 = np.dot(W2.T, dJ_Z2)
    dJ_Z1 = dJ_A1 * cache["dA1"]

    dJ_Wo = np.dot(dJ_Z3, cache["A2"].T)
    dJ_bo = np.sum(dJ_Z3, axis=1, keepdims=True)
    dJ_W2 = np.dot(dJ_Z2, cache["A1"].T)
    dJ_b2 = np.sum(dJ_Z2, axis=1, keepdims=True)
    dJ_W1 = np.dot(dJ_Z1, cache["A0_"][:, -1, :].T)
    dJ_b1 = np.sum(dJ_Z1, axis=1, keepdims=True)

    # gradient of the final state
    dJ_A0 = np.dot(W1.T, dJ_Z1)  # dh(t)

    # Propagate backward through time
    dim, T, N = cache["A0_"].shape
    dJ_Wi = 0
    dJ_Wr = 0
    dJ_br = 0
    for t in reversed(range(T)):
        x_train = x_train_[:, t, :]
        dA0 = cache["dA0_"][:, t, :]
        dJ_Z0 = dJ_A0 * dA0  # dZ for given time

        A0_prev = cache["A0_"][:, t - 1, :] if t > 0 else np.zeros((Wi.shape[0], N))
        # Sum gradients from different times
        dJ_Wi = dJ_Wi + np.dot(dJ_Z0, x_train.T)
        dJ_Wr = dJ_Wr + np.dot(dJ_Z0, A0_prev.T)
        dJ_br = dJ_br + np.sum(dJ_Z0, axis=1, keepdims=True)

        # Update dh(t) to dh(t-1) if t > 0
        if t > 0:
            dJ_A0 = np.dot(Wr.T, dJ_Z0)
    
    J_grad = {
        "Wi": dJ_Wi,
        "Wr": dJ_Wr,
        "W1": dJ_W1,
        "W2": dJ_W2,
        "Wo": dJ_Wo,
        "br": dJ_br,
        "b1": dJ_b1,
        "b2": dJ_b2,
        "bo": dJ_bo,
    }
    return J, J_grad


# Update weights for Q3a
def Q3a_update_weights(
    We, J_grad, We_momentum, learning_rate, momentum_rate, batch_size
):
    Wi = We["Wi"]
    Wr = We["Wr"]
    W1 = We["W1"]
    W2 = We["W2"]
    Wo = We["Wo"]
    br = We["br"]
    b1 = We["b1"]
    b2 = We["b2"]
    bo = We["bo"]

    dWi = learning_rate * J_grad["Wi"] / batch_size + momentum_rate * We_momentum["Wi"]
    dWr = learning_rate * J_grad["Wr"] / batch_size + momentum_rate * We_momentum["Wr"]
    dW1 = learning_rate * J_grad["W1"] / batch_size + momentum_rate * We_momentum["W1"]
    dW2 = learning_rate * J_grad["W2"] / batch_size + momentum_rate * We_momentum["W2"]
    dWo = learning_rate * J_grad["Wo"] / batch_size + momentum_rate * We_momentum["Wo"]
    dbr = learning_rate * J_grad["br"] / batch_size + momentum_rate * We_momentum["br"]
    db1 = learning_rate * J_grad["b1"] / batch_size + momentum_rate * We_momentum["b1"]
    db2 = learning_rate * J_grad["b2"] / batch_size + momentum_rate * We_momentum["b2"]
    dbo = learning_rate * J_grad["bo"] / batch_size + momentum_rate * We_momentum["bo"]

    # Update
    Wi -= dWi
    Wr -= dWr
    W1 -= dW1
    W2 -= dW2
    Wo -= dWo
    br -= dbr
    b1 -= db1
    b2 -= db2
    bo -= dbo

    We = {
        "Wi": Wi,
        "Wr": Wr,
        "W1": W1,
        "W2": W2,
        "Wo": Wo,
        "br": br,
        "b1": b1,
        "b2": b2,
        "bo": bo,
    }
    We_momentum = {
        "Wi": dWi,
        "Wr": dWr,
        "W1": dW1,
        "W2": dW2,
        "Wo": dWo,
        "br": dbr,
        "b1": db1,
        "b2": db2,
        "bo": dbo,
    }
    return We, We_momentum


# Gradient Descent for Q3a
def Q3a_gradient_descent(
    We,
    X_train,
    Y_train,
    X_validation,
    Y_validation,
    batch_size,
    epoch_count,
    learning_rate,
    momentum_rate,
    patience=5,
    min_delta=1e-5,
):
    # Initialize variables
    N = X_train.shape[2]
    mini_batch_count = int(N / batch_size)
    We_momentum = (0, 0, 0, 0, 0, 0, 0, 0, 0)
    momentum_dict = {
        "Wi": 0,
        "Wr": 0,
        "W1": 0,
        "W2": 0,
        "Wo": 0,
        "br": 0,
        "b1": 0,
        "b2": 0,
        "bo": 0,
    }

    best_cost = np.inf
    patience_counter = 0
    costs_train = []
    costs_validation = []

    # Iterate through epochs
    for epoch in range(epoch_count):
        mini_batch_start = 0
        mini_batch_end = batch_size

        sample_order = np.random.permutation(N)
        X_train = X_train[:, :, sample_order]
        Y_train = Y_train[:, sample_order]

        J_train_total = 0
        for _ in range(mini_batch_count):
            mini_batch_x = X_train[:, :, mini_batch_start:mini_batch_end]
            mini_batch_y = Y_train[:, mini_batch_start:mini_batch_end]
            J, J_grad = Q3a_calculate_gradients(We, mini_batch_x, mini_batch_y)
            We, momentum_dict = Q3a_update_weights(
                We, J_grad, momentum_dict, learning_rate, momentum_rate, batch_size
            )
            mini_batch_start = mini_batch_end
            mini_batch_end += batch_size
            mini_batch_end = min(mini_batch_end, N)

            J_train_total += J

        # Calculate training cost
        costs_train.append(J_train_total / mini_batch_count)

        # calculate validation cost
        cache_validation = Q3a_forward_propagation(We, X_validation)
        J_validation = cross_entropy(cache_validation["A3"], Y_validation)
        costs_validation.append(J_validation)

        # Check for early stopping
        if len(costs_validation) > 1:
            if costs_validation[-1] + min_delta < best_cost:
                best_cost = costs_validation[-1]
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"Early stop at epoch: {epoch+1}, Training Error: {round(costs_train[-1], 5)}, Validation Error: {round(costs_validation[-1], 5)}"
                    )
                    break

        # Print out progress
        print(
            f"Epoch: {epoch+1}, Training cost:{round(costs_train[-1],5)}, Validation cost: {round(costs_validation[-1], 5)} "
        )

    return We, costs_train, costs_validation


# Calculate confusion matrix
def confusion_matrix(Y_true, Y_pred):
    class_count = len(np.unique(Y_true))
    conf_matrix = np.zeros((class_count,class_count))
    for i in range(len(Y_true)):
        conf_matrix[Y_true[i]][Y_pred[i]] += 1
    return conf_matrix


# Calculate Accuracy
def calculate_accuracy(Y_true, Y_pred):
    count = np.count_nonzero(Y_true == Y_pred)
    return (100 * count / len(Y_true))


# Do Q3a stuff
def Q3a():
    # [1]
    with h5py.File("data3.h5") as hf:
        print(np.array(hf))
        X_tr = np.array(hf["trX"]).transpose(2, 1, 0)
        Y_tr = np.array(hf["trY"]).T
        X_test = np.array(hf["tstX"]).transpose(2, 1, 0)
        Y_test = np.array(hf["tstY"]).T

    assert X_tr.shape[2] == Y_tr.shape[1]
    shuffle_idxs = np.random.permutation(X_tr.shape[2])
    X_tr = X_tr[:, :, shuffle_idxs]
    Y_tr = Y_tr[:, shuffle_idxs]

    training_data_count = int(X_tr.shape[2] * 0.9)
    X_train = X_tr[:,:,:training_data_count]
    X_validation = X_tr[:,:,training_data_count:]
    Y_train = Y_tr[:,:training_data_count]
    Y_validation = Y_tr[:,training_data_count:]
    labels = ["Downstairs", "Jogging", "Sitting", "Standing", "Upstairs", "Walking"]

    print(X_train.shape)
    print(Y_train.shape)
    print(X_validation.shape)
    print(Y_validation.shape)
    print(X_test.shape)
    print(Y_test.shape)
    
    
    # [2]
    We = Q3a_initialize_weights(3, 128, (83, 51), 6)
    We_final, costs_train, costs_validation = Q3a_gradient_descent(
        We, X_train, Y_train, X_validation, Y_validation, 32, 50, 0.0001, 0.95
    )
    
    
    # [3]
    fig, ax = plt.subplots()
    ax.plot(
        list(range(1, len(costs_train) + 1)),
        costs_train,
        color="blue",
        label="Training",
    )
    ax.plot(
        list(range(1, len(costs_validation) + 1)),
        costs_validation,
        color="red",
        label="Validation",
    )
    ax.set_title("Cross Entropy vs Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross Entropy")
    ax.legend(loc="best")

    fig.savefig("Q3a_images/q3a_cost.png")
    
    
    # [4]
    cache_train = Q3a_forward_propagation(We_final, X_train)
    pred_train = np.argmax(cache_train["A3"], axis=0)
    Y_train_idx = np.argmax(Y_train, axis=0)

    accuracy_train = calculate_accuracy(Y_train_idx, pred_train)
    print(f"Train accuracy: {accuracy_train}")

    confusion_train = confusion_matrix(Y_train_idx, pred_train)
    sns.heatmap(confusion_train, annot=True, xticklabels=labels, yticklabels=labels)
    plt.title("Training Set Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Prediction")

    plt.savefig("Q3a_images/q3a_confusion_train")
    
    
    # [5]
    cache_test = Q3a_forward_propagation(We_final, X_test)
    pred_test = np.argmax(cache_test["A3"], axis=0)
    Y_test_idx = np.argmax(Y_test, axis=0)

    accuracy_test = calculate_accuracy(Y_test_idx, pred_test)
    print(f"Test accuracy: {accuracy_test}")

    confusion_test = confusion_matrix(Y_test_idx, pred_test)
    sns.heatmap(confusion_test, annot=True, xticklabels=labels, yticklabels=labels)
    plt.title("Test Set Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Prediction")

    plt.savefig("Q3a_images/q3a_confusion_test")



# Init weights for Q3b
def Q3b_initialize_weights(n_x: int, n_r: int, n_h: tuple[int, int], n_y: int):
    # LTSM Weights
    Wf, bf = Xavier_initialization((n_r + n_x), n_r)
    Wi, bi = Xavier_initialization((n_r + n_x), n_r)
    Wc, bc = Xavier_initialization((n_r + n_x), n_r)
    Wo, bo = Xavier_initialization((n_r + n_x), n_r)

    # MLP Weights
    W1, b1 = Xavier_initialization(n_r, n_h[0])
    W2, b2 = Xavier_initialization(n_h[0], n_h[1])
    W3, b3 = Xavier_initialization(n_h[1], n_y)

    We = {
        "Wf": Wf,
        "Wi": Wi,
        "Wc": Wc,
        "Wo": Wo,
        "bf": bf,
        "bi": bi,
        "bc": bc,
        "bo": bo,
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "W3": W3,
        "b3": b3,
    }
    return We


def Q3b_forward_propagation(We, data):
    Wf = We["Wf"]
    Wc = We["Wc"]
    Wi = We["Wi"]
    Wo = We["Wo"]
    bf = We["bf"]
    bi = We["bi"]
    bc = We["bc"]
    bo = We["bo"]
    W1 = We["W1"]
    W2 = We["W2"]
    W3 = We["W3"]
    b1 = We["b1"]
    b2 = We["b2"]
    b3 = We["b3"]
    
    dim, time, samples = data.shape
    recurrent_layer_size = Wf.shape[0]

    # Initialize related variables
    stacked_input = np.zeros((dim + recurrent_layer_size, time, samples))
    A0 = np.zeros((recurrent_layer_size, samples))
    C_prev = np.zeros((recurrent_layer_size, samples))
    C = np.zeros((recurrent_layer_size, time, samples))
    tanhc = np.zeros((recurrent_layer_size, time, samples))
    Af = np.zeros((recurrent_layer_size, time, samples))
    Ai = np.zeros((recurrent_layer_size, time, samples))
    Ac = np.zeros((recurrent_layer_size, time, samples))
    Ao = np.zeros((recurrent_layer_size, time, samples))
    dtanhc = np.zeros((recurrent_layer_size, time, samples))
    dAf = np.zeros(((recurrent_layer_size, time, samples)))
    dAi = np.zeros(((recurrent_layer_size, time, samples)))
    dAc = np.zeros(((recurrent_layer_size, time, samples)))
    dAo = np.zeros(((recurrent_layer_size, time, samples)))

    for t in range(time):
        stacked_input[:, t, :] = np.row_stack((A0, data[:, t, :]))
        curr_stacked = stacked_input[:, t, :]

        Af[:, t, :], dAf[:, t, :] = sigmoid(np.matmul(Wf, curr_stacked) + bf)
        Ai[:, t, :], dAi[:, t, :] = sigmoid(np.matmul(Wi, curr_stacked) + bi)
        Ac[:, t, :], dAc[:, t, :] = tanh(np.matmul(Wc, curr_stacked) + bc)
        Ao[:, t, :], dAo[:, t, :] = sigmoid(np.matmul(Wo, curr_stacked) + bo)

        C[:, t, :] = Af[:, t, :] * C_prev + Ai[:, t, :] * Ac[:, t, :]
        tanhc[:, t, :], dtanhc[:, t, :] = tanh(C[:, t, :])
        A0 = tanhc[:, t, :] * Ao[:, t, :]
        C_prev = C[:, t, :]

    # relu layers
    Z1 = np.matmul(W1, A0) + b1
    A1, d_activation_1 = relu(Z1)
    Z2 = np.matmul(W2, A1) + b2
    A2, d_activation_2 = relu(Z2)

    # softmax layer
    Z3 = np.matmul(W3, A2) + b3
    A3 = softmax(Z3)

    
    cache = {
        "Z1": Z1,
        "A1": A1,
        "dA1": d_activation_1,
        "Z2": Z2,
        "A2": A2,
        "dA2": d_activation_2,
        "Z3": Z3,
        "A3": A3,
        "cache": (stacked_input, A0, C, tanhc, Af, Ai, Ac, Ao, dtanhc, dAf, dAc, dAi, dAo),
    }
    return cache


def Q3b_calculate_gradients(We, x_train, y_train):
    W1 = We["W1"]
    W2 = We["W2"]
    W3 = We["W3"]
    b1 = We["b1"]
    b2 = We["b2"]
    b3 = We["b3"]
    Wf = We["Wf"]
    Wi = We["Wi"]
    Wc = We["Wc"]
    Wo = We["Wo"]
    bf = We["bf"]
    bi = We["bi"]
    bc = We["bc"]
    bo = We["bo"]

    # Forward propagate for training
    cache = Q3b_forward_propagation(We, x_train)
    (
        stacked_input,
        A0,
        C,
        tanhc,
        h_forget,
        h_input,
        h_candidate,
        h_output,
        d_activation_tanhc,
        d_activation_h_forget,
        d_activation_h_candidate,
        d_activation_h_input,
        d_activation_h_output,
    ) = cache["cache"]

    # Calculate training cost for this batch
    batch_training_cost = cross_entropy(cache["A3"], y_train)

    # Backward propagate until recurrent layer
    dJ_Z3 = cache["A3"] - y_train
    dJ_W3 = np.matmul(dJ_Z3, cache["A2"].T)
    dJ_b3 = np.sum(dJ_Z3, axis=1, keepdims=True)
    dJ_A2 = np.matmul(W3.T, dJ_Z3)
    dJ_Z2 = dJ_A2 * cache["dA2"]

    dJ_W2 = np.matmul(dJ_Z2, cache["A1"].T)
    dJ_b2 = np.sum(dJ_Z2, axis=1, keepdims=True)
    dJ_A1 = np.matmul(W2.T, dJ_Z2)
    dJ_Z1 = dJ_A1 * cache["dA1"]
    dJ_W1 = np.matmul(dJ_Z1, A0.T)  # W1 takes final state
    dJ_b1 = np.sum(dJ_Z1, axis=1, keepdims=True)

    # gradient of the final state
    d_state = np.matmul(W1.T, dJ_Z1)  # dh(t)
    d_z = d_state

    # Sizes to be used
    dim, time, sample = h_forget.shape

    # initialize gradients regarding lstm layer
    dJ_Wf = 0
    dJ_Wi = 0
    dJ_Wc = 0
    dJ_Wo = 0
    dJ_bf = 0
    dJ_bi = 0
    dJ_bc = 0
    dJ_bo = 0

    for t in reversed(range(time)):
        stacked_curr = stacked_input[:, t, :]

        C_prev = C[:, t - 1, :] if t > 0 else np.zeros((dim, sample))
        dc = d_z * h_output[:, t, :] * d_activation_tanhc[:, t, :]
        dhf = dc * C_prev * d_activation_h_forget[:, t, :]
        dhi = dc * h_candidate[:, t, :] * d_activation_h_input[:, t, :]
        dhc = dc * h_input[:, t, :] * d_activation_h_candidate[:, t, :]
        dho = d_z * tanhc[:, t, :] * d_activation_h_output[:, t, :]

        # Sum gradients
        dJ_Wf += np.matmul(dhf, stacked_curr.T)
        dJ_Wi += np.matmul(dhi, stacked_curr.T)
        dJ_Wc += np.matmul(dhc, stacked_curr.T)
        dJ_Wo += np.matmul(dho, stacked_curr.T)
        dJ_bf += np.sum(dhf, axis=1, keepdims=True)
        dJ_bi += np.sum(dhi, axis=1, keepdims=True)
        dJ_bc += np.sum(dhc, axis=1, keepdims=True)
        dJ_bo += np.sum(dho, axis=1, keepdims=True)
        
        dxf = np.matmul(Wf.T[:dim, :], dhf)
        dxc = np.matmul(Wc.T[:dim, :], dhc)
        dxi = np.matmul(Wi.T[:dim, :], dhi)
        dxo = np.matmul(Wo.T[:dim, :], dho)

        d_z = dxf + dxc + dxi + dxo

    J_grad = {
        "W1": dJ_W1,
        "W2": dJ_W2,
        "W3": dJ_W3,
        "b1": dJ_b1,
        "b2": dJ_b2,
        "b3": dJ_b3,
        "Wf": dJ_Wf,
        "Wi": dJ_Wi,
        "Wc": dJ_Wc,
        "Wo": dJ_Wo,
        "bf": dJ_bf,
        "bi": dJ_bi,
        "bc": dJ_bc,
        "bo": dJ_bo,
    }
    return batch_training_cost, J_grad


def Q3b_update_parameters(
    We, J_grad, learning_rate, We_momentum, momentum_rate
):
    W1 = We["W1"]
    W2 = We["W2"]
    W3 = We["W3"]
    b1 = We["b1"]
    b2 = We["b2"]
    b3 = We["b3"]
    Wf = We["Wf"]
    Wi = We["Wi"]
    Wc = We["Wc"]
    Wo = We["Wo"]
    bf = We["bf"]
    bi = We["bi"]
    bc = We["bc"]
    bo = We["bo"]

    dW1 = learning_rate * J_grad["W1"] + momentum_rate * We_momentum["W1"]
    dW2 = learning_rate * J_grad["W2"] + momentum_rate * We_momentum["W2"]
    dW3 = learning_rate * J_grad["W3"] + momentum_rate * We_momentum["W3"]
    db1 = learning_rate * J_grad["b1"] + momentum_rate * We_momentum["b1"]
    db2 = learning_rate * J_grad["b2"] + momentum_rate * We_momentum["b2"]
    db3 = learning_rate * J_grad["b3"] + momentum_rate * We_momentum["b3"]
    dWf = learning_rate * J_grad["Wf"] + momentum_rate * We_momentum["Wf"]
    dWi = learning_rate * J_grad["Wi"] + momentum_rate * We_momentum["Wi"]
    dWc = learning_rate * J_grad["Wc"] + momentum_rate * We_momentum["Wc"]
    dWo = learning_rate * J_grad["Wo"] + momentum_rate * We_momentum["Wo"]
    dbf = learning_rate * J_grad["bf"] + momentum_rate * We_momentum["bf"]
    dbi = learning_rate * J_grad["bi"] + momentum_rate * We_momentum["bi"]
    dbc = learning_rate * J_grad["bc"] + momentum_rate * We_momentum["bc"]
    dbo = learning_rate * J_grad["bo"] + momentum_rate * We_momentum["bo"]

    # Update
    W1 -= dW1
    W2 -= dW2
    W3 -= dW3
    b1 -= db1
    b2 -= db2
    b3 -= db3
    Wf -= dWf
    Wi -= dWi
    Wc -= dWc
    Wo -= dWo
    bf -= dbf
    bi -= dbi
    bc -= dbc
    bo -= dbo

    We = {
        "Wf": Wf,
        "Wi": Wi,
        "Wc": Wc,
        "Wo": Wo,
        "bf": bf,
        "bi": bi,
        "bc": bc,
        "bo": bo,
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2,
        "W3": W3,
        "b3": b3,
    }
    momentum_dict = {
        "W1": dW1,
        "W2": dW2,
        "W3": dW3,
        "b1": db1,
        "b2": db2,
        "b3": db3,
        "Wf": dWf,
        "Wi": dWi,
        "Wc": dWc,
        "Wo": dWo,
        "bf": dbf,
        "bi": dbi,
        "bc": dbc,
        "bo": dbo,
    }
    return We, momentum_dict


def Q3b_gradient_descent(
    We,
    X_train,
    Y_train,
    X_validation,
    Y_validation,
    batch_size,
    epoch_count,
    learning_rate,
    momentum_rate,
    patience=5,
    min_delta=1e-5,
):
    N = X_train.shape[2]
    mini_batch_count = int(N / batch_size)
    We_momentum = {
        "W1": 0,
        "W2": 0,
        "W3": 0,
        "b1": 0,
        "b2": 0,
        "b3": 0,
        "Wf": 0,
        "Wi": 0,
        "Wc": 0,
        "Wo": 0,
        "bf": 0,
        "bi": 0,
        "bc": 0,
        "bo": 0,
    }

    best_cost = np.inf
    patience_counter = 0
    costs_train = []
    costs_validation = []
    for epoch in range(epoch_count):
        mini_batch_start = 0
        mini_batch_end = batch_size

        sample_order = np.random.permutation(N)
        X_train = X_train[:, :, sample_order]
        Y_train = Y_train[:, sample_order]

        J_train_total = 0
        for _ in range(mini_batch_count):
            mini_batch_x = X_train[:, :, mini_batch_start:mini_batch_end]
            mini_batch_y = Y_train[:, mini_batch_start:mini_batch_end]
            J, grads = Q3b_calculate_gradients(We, mini_batch_x, mini_batch_y)
            We, We_momentum = Q3b_update_parameters(
                We, grads, learning_rate, We_momentum, momentum_rate
            )
            mini_batch_start = mini_batch_end
            mini_batch_end += batch_size
            mini_batch_end = min(mini_batch_end, N)

            J_train_total += J

        # Calculate training cost
        costs_train.append(J_train_total / mini_batch_count)

        # Calculate validation cost
        cache = Q3b_forward_propagation(We, X_validation)
        J_validation = cross_entropy(cache["A3"], Y_validation)
        costs_validation.append(J_validation)

        # Check for early stopping
        if len(costs_validation) > 1:
            if costs_validation[-1] + min_delta < best_cost:
                best_cost = costs_validation[-1]
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"Early stop at epoch: {epoch+1}, Training Error: {round(costs_train[-1], 5)}, Validation Error: {round(costs_validation[-1], 5)}"
                    )
                    break

        # Print out progress
        print(
            f"Epoch: {epoch+1}, Training cost:{round(costs_train[-1],5)}, Validation cost: {round(costs_validation[-1], 5)} "
        )

    return We, costs_train, costs_validation


def Q3b():
    with h5py.File("data3.h5") as hf:
        print(np.array(hf))
        X_tr = np.array(hf["trX"]).transpose(2, 1, 0)
        Y_tr = np.array(hf["trY"]).T
        X_test = np.array(hf["tstX"]).transpose(2, 1, 0)
        Y_test = np.array(hf["tstY"]).T

    assert X_tr.shape[2] == Y_tr.shape[1]
    shuffle_idxs = np.random.permutation(X_tr.shape[2])
    X_tr = X_tr[:, :, shuffle_idxs]
    Y_tr = Y_tr[:, shuffle_idxs]

    training_data_count = int(X_tr.shape[2] * 0.9)
    X_train = X_tr[:,:,:training_data_count]
    X_validation = X_tr[:,:,training_data_count:]
    Y_train = Y_tr[:,:training_data_count]
    Y_validation = Y_tr[:,training_data_count:]
    labels = ["Downstairs", "Jogging", "Sitting", "Standing", "Upstairs", "Walking"]

    print(X_train.shape)
    print(Y_train.shape)
    print(X_validation.shape)
    print(Y_validation.shape)
    print(X_test.shape)
    print(Y_test.shape)
    
    
    We = Q3b_initialize_weights(3, 128, (83, 51), 6)
    We_final, costs_training, costs_validation = Q3b_gradient_descent(
        We, X_train, Y_train, X_validation, Y_validation, 32, 50, 0.0001, 0.95
    )
    
    
    fig, ax = plt.subplots()
    ax.plot(
        list(range(1, len(costs_training) + 1)),
        costs_training,
        color="blue",
        label="Training",
    )
    ax.plot(
        list(range(1, len(costs_validation) + 1)),
        costs_validation,
        color="red",
        label="Validation",
    )
    ax.set_title("Cross Entropy vs Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross Entropy")
    ax.legend(loc="best")

    fig.savefig("Q3b_images/q3b_cost.png")
    
    
    cache_train = Q3b_forward_propagation(We_final, X_train)
    pred_train = np.argmax(cache_train["A3"], axis=0)
    Y_train_idx = np.argmax(Y_train, axis=0)

    accuracy_train = calculate_accuracy(Y_train_idx, pred_train)
    print(f"Train accuracy: {accuracy_train}")

    confusion_train = confusion_matrix(Y_train_idx, pred_train)
    sns.heatmap(confusion_train, annot=True, xticklabels=labels, yticklabels=labels)
    plt.title("Training Set Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Prediction")

    plt.savefig("Q3b_images/q3b_confusion_train")
    
    
    cache_test = Q3b_forward_propagation(We_final, X_test)
    pred_test = np.argmax(cache_test["A3"], axis=0)
    Y_test_idx = np.argmax(Y_test, axis=0)

    accuracy_test = calculate_accuracy(Y_test_idx, pred_test)
    print(f"Test accuracy: {accuracy_test}")

    confusion_test = confusion_matrix(Y_test_idx, pred_test)
    sns.heatmap(confusion_test, annot=True, xticklabels=labels, yticklabels=labels)
    plt.title("Test Set Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Prediction")

    plt.savefig("Q3b_images/q3b_confusion_test")
    
    
def Q3c_initialize_weights(n_x, n_r, n_h: tuple[int, int], n_y):
    w0 = np.sqrt(6 / (n_x + n_r))
    w0rr = np.sqrt(6 / (n_r + n_r))
    
    Wr, br = Xavier_initialization(n_x, n_r)
    Ur, _ = Xavier_initialization(n_r, n_r)
    Wz, bz = Xavier_initialization(n_x, n_r)
    Uz, _ = Xavier_initialization(n_r, n_r)
    Wh, bh = Xavier_initialization(n_x, n_r)
    Uh, _ = Xavier_initialization(n_r, n_r)
    
    W1, b1 = Xavier_initialization(n_r, n_h[0])
    W2, b2 = Xavier_initialization(n_h[0], n_h[1])
    W3, b3 = Xavier_initialization(n_h[1], n_y)

    We = {
        "Wr": Wr,
        "Wz": Wz,
        "Wh": Wh,
        "Ur": Ur,
        "Uz": Uz,
        "Uh": Uh,
        "br": br,
        "bz": bz,
        "bh": bh,
        "W1": W1,
        "W2": W2,
        "W3": W3,
        "b1": b1,
        "b2": b2,
        "b3": b3,
    }
    return We


def Q3c_forward_propagation(We, X):
    Wr = We["Wr"]
    Wz = We["Wz"]
    Wh = We["Wh"]
    br = We["br"]
    bz = We["bz"]
    bh = We["bh"]
    Ur = We["Ur"]
    Uz = We["Uz"]
    Uh = We["Uh"]
    W1 = We["W1"]
    W2 = We["W2"]
    W3 = We["W3"]
    b1 = We["b1"]
    b2 = We["b2"]
    b3 = We["b3"]
    
    dim,T,N = X.shape
    recurrent_layer_size = Wz.shape[0]

    #Initialize related variables
    A = np.zeros((recurrent_layer_size,T,N))
    A_prev = np.zeros((recurrent_layer_size, N))
    Z = np.zeros((recurrent_layer_size,T,N))
    dZ = np.zeros((recurrent_layer_size,T,N))
    R = np.zeros((recurrent_layer_size,T,N))
    dR = np.zeros((recurrent_layer_size,T,N))
    H = np.zeros((recurrent_layer_size,T,N))
    dH = np.zeros((recurrent_layer_size,T,N))
  

    for t in range(T):
        x_cur = X[:,t,:]
        R[:,t,:], dR[:,t,:] = sigmoid(np.matmul(Wr,x_cur) + np.matmul(Ur,A_prev) + br)
        Z[:,t,:], dZ[:,t,:] = sigmoid(np.matmul(Wz,x_cur) + np.matmul(Uz,A_prev) + bz)
        H[:,t,:], dH[:,t,:] = tanh(np.matmul(Wh,x_cur) + np.matmul(Uh,(R[:,t,:] * A_prev)) + bh)
        A[:,t,:] = (1 - Z[:,t,:]) * A_prev + Z[:,t,:] * H[:,t,:]
        A_prev = A[:,t,:]
        
    A0 = A[:, -1, :]

    # relu layers
    Z1 = np.matmul(W1, A0) + b1
    A1, dA1 = relu(Z1)
    Z2 = np.matmul(W2, A1) + b2
    A2, dA2 = relu(Z2)

    # softmax layer
    Z3 = np.matmul(W3, A2) + b3
    A3 = softmax(Z3)

    cache = {
        "Z1": Z1,
        "A1": A1,
        "dA1": dA1,
        "Z2": Z2,
        "A2": A2,
        "dA2": dA2,
        "Z3": Z3,
        "A3": A3,
        "cache": (A,Z,dZ,R,dR,H,dH),
    }
    return cache


def Q3_calculate_gradients(We, x_train, y_train):
    W1 = We["W1"]
    W2 = We["W2"]
    W3 = We["W3"]
    b1 = We["b1"]
    b2 = We["b2"]
    b3 = We["b3"]
    Wz = We["Wz"]
    Wr = We["Wr"]
    Wh = We["Wh"]
    Uz = We["Uz"]
    Ur = We["Ur"]
    Uh = We["Uh"]
    bz = We["bz"]
    br = We["br"]
    bh = We["bh"]

    # Forward propagate for training
    cache = Q3c_forward_propagation(We, x_train)
    (h, z, d_activation_z, r, d_activation_r, h_, d_activation_h_) = cache["cache"]
    h_prev = h[:, -1, :]

    # Calculate training cost for this batch
    J = cross_entropy(cache["A3"], y_train)

    # Backward propagate until recurrent layer
    dZ3 = cache["A3"] - y_train
    dW3 = np.matmul(dZ3, cache["A2"].T)
    db3 = np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.matmul(W3.T, dZ3)
    dZ2 = dA2 * cache["dA2"]
    dW2 = np.matmul(dZ2, cache["A1"].T)
    db2 = np.sum(dZ2, axis=1, keepdims=True)
    dA1 = np.matmul(W2.T, dZ2)
    dZ1 = dA1 * cache["dA1"]
    dW1 = np.matmul(dZ1, h_prev.T)  # W1 takes final state
    db1 = np.sum(dZ1, axis=1, keepdims=True)

    # gradient of the final state
    d_state = np.matmul(W1.T, dZ1)  # dh(t)
    d_z = d_state  # this d_z is not related with the parameter of z

    # Sizes to be used
    dim, time, sample = h.shape

    # initialize gradients regarding lstm layer
    dWz = 0
    dWr = 0
    dWh = 0
    dbz = 0
    dbr = 0
    dbh = 0
    dUz = 0
    dUr = 0
    dUh = 0

    for t in reversed(range(time)):
        x_cur = x_train[:, t, :]

        h_prev = h[:, t - 1, :] if t > 0 else np.zeros((dim, sample))
        dz = d_z * d_activation_z[:, t, :] * (h_[:, t, :] - h_prev)
        dh_ = d_z * d_activation_h_[:, t, :] * z[:, t, :]
        dr = np.matmul(Uh.T, dh_) * h_prev * d_activation_r[:, t, :]

        dWz += np.matmul(dz, x_cur.T)
        dUz += np.matmul(dz, h_prev.T)
        dbz += np.sum(dz, axis=1, keepdims=True)

        dWh += np.matmul(dh_, x_cur.T)
        dUh += np.matmul(dh_, h_prev.T)
        dbh += np.sum(dh_, axis=1, keepdims=True)

        dWr += np.matmul(dr, x_cur.T)
        dUr += np.matmul(dr, h_prev.T)
        dbr += np.sum(dr, axis=1, keepdims=True)

        # update d_z
        d1 = d_z * (1 - z[:, t, :])
        d2 = np.matmul(Uz.T, dz)
        d3 = np.matmul(Uh.T, dh_) * (
            r[:, t, :] + h_prev * np.matmul(Ur.T, d_activation_r[:, t, :])
        )
        d_z = d1 + d2 + d3

    J_grad = {
        "W1": dW1,
        "W2": dW2,
        "W3": dW3,
        "b1": db1,
        "b2": db2,
        "b3": db3,
        "Wz": dWz,
        "Wr": dWr,
        "Wh": dWh,
        "bz": dbz,
        "br": dbr,
        "bh": dbh,
        "Uz": dUz,
        "Ur": dUr,
        "Uh": dUh,
    }
    return J, J_grad


def Q3c_update_parameters(
    We, J_grad, We_momentum, learning_rate, momentum_rate
):
    W1 = We["W1"]
    W2 = We["W2"]
    W3 = We["W3"]
    b1 = We["b1"]
    b2 = We["b2"]
    b3 = We["b3"]
    Wz = We["Wz"]
    Wr = We["Wr"]
    Wh = We["Wh"]
    bz = We["bz"]
    br = We["br"]
    bh = We["bh"]
    Uz = We["Uz"]
    Ur = We["Ur"]
    Uh = We["Uh"]

    dW1 = learning_rate * J_grad["W1"] + momentum_rate * We_momentum["W1"]
    dW2 = learning_rate * J_grad["W2"] + momentum_rate * We_momentum["W2"]
    dW3 = learning_rate * J_grad["W3"] + momentum_rate * We_momentum["W3"]
    db1 = learning_rate * J_grad["b1"] + momentum_rate * We_momentum["b1"]
    db2 = learning_rate * J_grad["b2"] + momentum_rate * We_momentum["b2"]
    db3 = learning_rate * J_grad["b3"] + momentum_rate * We_momentum["b3"]
    dWz = learning_rate * J_grad["Wz"] + momentum_rate * We_momentum["Wz"]
    dWr = learning_rate * J_grad["Wr"] + momentum_rate * We_momentum["Wr"]
    dWh = learning_rate * J_grad["Wh"] + momentum_rate * We_momentum["Wh"]
    dbz = learning_rate * J_grad["bz"] + momentum_rate * We_momentum["bz"]
    dbr = learning_rate * J_grad["br"] + momentum_rate * We_momentum["br"]
    dbh = learning_rate * J_grad["bh"] + momentum_rate * We_momentum["bh"]
    dUz = learning_rate * J_grad["Uz"] + momentum_rate * We_momentum["Uz"]
    dUr = learning_rate * J_grad["Ur"] + momentum_rate * We_momentum["Ur"]
    dUh = learning_rate * J_grad["Uh"] + momentum_rate * We_momentum["Uh"]

    # Update
    W1 = W1 - dW1
    W2 = W2 - dW2
    W3 = W3 - dW3
    b1 = b1 - db1
    b2 = b2 - db2
    b3 = b3 - db3
    Wz = Wz - dWz
    Wr = Wr - dWr
    Wh = Wh - dWh
    bz = bz - dbz
    br = br - dbr
    bh = bh - dbh
    Uz = Uz - dUz
    Ur = Ur - dUr
    Uh = Uh - dUh

    We = {
        "Wz": Wz,
        "Wr": Wr,
        "Wh": Wh,
        "Uz": Uz,
        "Ur": Ur,
        "Uh": Uh,
        "bz": bz,
        "br": br,
        "bh": bh,
        "W1": W1,
        "W2": W2,
        "W3": W3,
        "b1": b1,
        "b2": b2,
        "b3": b3,
    }
    We_momentum = {
        "Wz": dWz,
        "Wr": dWr,
        "Wh": dWh,
        "bz": dbz,
        "br": dbr,
        "bh": dbh,
        "Uz": dUz,
        "Ur": dUr,
        "Uh": dUh,
        "W1": dW1,
        "W2": dW2,
        "W3": dW3,
        "b1": db1,
        "b2": db2,
        "b3": db3,
    }
    return We, We_momentum


def Q3c_gradient_descent(
    We,
    X_train,
    Y_train,
    X_validation,
    Y_validation,
    batch_size,
    epoch_count,
    learning_rate,
    momentum_rate,
    patience=5,
    min_delta=1e-5,
):
    N = X_train.shape[2]
    mini_batch_count = int(N / batch_size)
    We_momentum = {
        "Wz": 0,
        "Wr": 0,
        "Wh": 0,
        "bz": 0,
        "br": 0,
        "bh": 0,
        "Uz": 0,
        "Ur": 0,
        "Uh": 0,
        "W1": 0,
        "W2": 0,
        "W3": 0,
        "b1": 0,
        "b2": 0,
        "b3": 0,
    }

    best_cost = np.inf
    patience_counter = 0
    costs_train = []
    costs_validation = []

    # Iterate through epochs
    for epoch in range(epoch_count):
        mini_batch_start = 0
        mini_batch_end = batch_size

        sample_order = np.random.permutation(N)
        X_train = X_train[:, :, sample_order]
        Y_train = Y_train[:, sample_order]

        J_train_total = 0
        for _ in range(mini_batch_count):
            mini_batch_x = X_train[:, :, mini_batch_start:mini_batch_end]
            mini_batch_y = Y_train[:, mini_batch_start:mini_batch_end]
            J, grads = Q3_calculate_gradients(We, mini_batch_x, mini_batch_y)
            We, We_momentum = Q3c_update_parameters(
                We, grads, We_momentum, learning_rate, momentum_rate
            )
            mini_batch_start = mini_batch_end
            mini_batch_end += batch_size
            mini_batch_end = min(mini_batch_end, N)

            J_train_total += J

        # Calculate training cost
        costs_train.append(J_train_total / mini_batch_count)

        # Forward propagate for validation
        cache = Q3c_forward_propagation(We, X_validation)
        J_validation = cross_entropy(cache["A3"], Y_validation)
        costs_validation.append(J_validation)

        # Check for early stopping
        if len(costs_validation) > 1:
            if costs_validation[-1] + min_delta < best_cost:
                best_cost = costs_validation[-1]
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"Early stop at epoch: {epoch+1}, Training Error: {round(costs_train[-1], 5)}, Validation Error: {round(costs_validation[-1], 5)}"
                    )
                    break

        # Print out progress
        print(
            f"Epoch: {epoch+1}, Training cost:{round(costs_train[-1],5)}, Validation cost: {round(costs_validation[-1], 5)} "
        )

    return We, costs_train, costs_validation


def Q3c():
    with h5py.File("data3.h5") as hf:
        print(np.array(hf))
        X_tr = np.array(hf["trX"]).transpose(2, 1, 0)
        Y_tr = np.array(hf["trY"]).T
        X_test = np.array(hf["tstX"]).transpose(2, 1, 0)
        Y_test = np.array(hf["tstY"]).T

    assert X_tr.shape[2] == Y_tr.shape[1]
    shuffle_idxs = np.random.permutation(X_tr.shape[2])
    X_tr = X_tr[:, :, shuffle_idxs]
    Y_tr = Y_tr[:, shuffle_idxs]

    training_data_count = int(X_tr.shape[2] * 0.9)
    X_train = X_tr[:,:,:training_data_count]
    X_validation = X_tr[:,:,training_data_count:]
    Y_train = Y_tr[:,:training_data_count]
    Y_validation = Y_tr[:,training_data_count:]
    labels = ["Downstairs", "Jogging", "Sitting", "Standing", "Upstairs", "Walking"]

    print(X_train.shape)
    print(Y_train.shape)
    print(X_validation.shape)
    print(Y_validation.shape)
    print(X_test.shape)
    print(Y_test.shape)
    
    
    We = Q3c_initialize_weights(3, 128, (83, 81), 6)
    We_final, costs_validation, costs_training = Q3c_gradient_descent(
        We, X_train, Y_train, X_validation, Y_validation, 32, 50, 0.0001, 0.95
    )
    
    
    fig, ax = plt.subplots()
    ax.plot(
        list(range(1, len(costs_training) + 1)),
        costs_training,
        color="blue",
        label="Training",
    )
    ax.plot(
        list(range(1, len(costs_validation) + 1)),
        costs_validation,
        color="red",
        label="Validation",
    )
    ax.set_title("Cross Entropy vs Epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross Entropy")
    ax.legend(loc="best")

    fig.savefig("Q3c_images/q3c_cost.png")
    
    
    cache_train = Q3c_forward_propagation(We_final, X_train)
    pred_train = np.argmax(cache_train["A3"], axis=0)
    Y_train_idx = np.argmax(Y_train, axis=0)

    accuracy_train = calculate_accuracy(Y_train_idx, pred_train)
    print(f"Train accuracy: {accuracy_train}")

    confusion_train = confusion_matrix(Y_train_idx, pred_train)
    sns.heatmap(confusion_train, annot=True, xticklabels=labels, yticklabels=labels)
    plt.title("Training Set Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Prediction")

    plt.savefig("Q3a_images/q3a_confusion_train")
    
    
    cache_test = Q3c_forward_propagation(We_final, X_test)
    pred_test = np.argmax(cache_test["A3"], axis=0)
    Y_test_idx = np.argmax(Y_test, axis=0)

    accuracy_test = calculate_accuracy(Y_test_idx, pred_test)
    print(f"Test accuracy: {accuracy_test}")

    confusion_test = confusion_matrix(Y_test_idx, pred_test)
    sns.heatmap(confusion_test, annot=True, xticklabels=labels, yticklabels=labels)
    plt.title("Test Set Confusion Matrix")
    plt.ylabel("Actual")
    plt.xlabel("Prediction")

    plt.savefig("Q3c_images/q3c_confusion_test")
    
    
    
if __name__ == "__main__":
    Q1()
    Q2()
    Q3a()
    Q3b()
    Q3c()