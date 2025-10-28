import numpy as np
# -----------------------------------------------------------
# VARIABLES SETUP
# ------------------------------------------------------------
# global parameters
d = 10  # d of feature vectors
d_prime = 8  # dimension after Q/K/V transformation
L = 3

alphabet_w_size = 5 # |Sigma + omega|
W_enc = np.random.rand(alphabet_w_size, d)

# ('a', 'b', 'c', 'omega')
# the indices of the symbols in W_enc:
# 0 -> 'a', 1 -> 'b', 2 -> 'c', 3 -> 'omega', 4 -> 'other'
input_word_indices = [0, 1, 3] # "abw"
n = len(input_word_indices)

# |X_0|: n x d
X_0 = W_enc[input_word_indices, :]

# Q^l, K^l, V^l: d x d'
Q_1 = np.random.rand(d, d_prime)
K_1 = np.random.rand(d, d_prime)
V_1 = np.random.rand(d, d_prime)

# O^l: d' x d
O_1 = np.random.rand(d_prime, d)

# activation function
def sigma(M):
    # relu
    return np.maximum(0, M)

threshold = 0.5

# ----------------------------------------------------------------
# ATTENTION CALCULATION
# ----------------------------------------------------------------

# A^{l}=LArgMax(X_{l-1}Q^{l}(X_{l-1}K^{l})^{T})
def leftmost_argmax_row_wise(M):
    n, _ = M.shape
    A = np.zeros(M.shape)

    for i in range(n):
        row = M[i, :]
        max_val = np.max(row)

        # Find the index of the leftmost occurrence of the maximum value
        j_max_leftmost = np.where(row == max_val)[0][0]

        A[i, j_max_leftmost] = 1

    return A

def compute_attention_matrix(X_prev, Q_l, K_l):
    # 1. Calculating the attention scores matrix
    Query_Vectors = X_prev @ Q_l
    Key_Vectors = X_prev @ K_l
    AttentionScores = Query_Vectors @ Key_Vectors.T

    # 2. Applying LArgMax row-wise to get A^l 
    A_l = leftmost_argmax_row_wise(AttentionScores)

    return A_l

# Example calculation for A_1
A_1 = compute_attention_matrix(X_0, Q_1, K_1)
print(f"Attention Matrix A_1 (n x n):\n{A_1}")

# -----------------------------------------------------------------
# COMPUTING NEW FEATURE VECTORS
# ----------------------------------------------------------------
def compute_new_feature_vectors(A_l, X_prev, V_l, O_l, activation_func):
    # X_{l}=\sigma(A^{l}X_{l-1}V^{l}O^{l})+X_{l-1}

    # A^l X_{l-1}: (n x n) @ (n x d) = (n x d)
    # selects the feature vector x_j that position i attends to (j)
    Attended_X = A_l @ X_prev

    # Attended_X V^l: (n x d) @ (d x d') = (n x d')
    Attended_V = Attended_X @ V_l

    # Attended_V O^l: (n x d') @ (d' x d) = (n x d)
    Output_Vector_Update = Attended_V @ O_l

    # applying activation function
    Output = activation_func(Output_Vector_Update)

    X_l = Output + X_prev

    return X_l

# Example calculation for X_1
X_1 = compute_new_feature_vectors(A_1, X_0, V_1, O_1, sigma)
print(f"\nNew Feature Vectors Matrix X_1 (n x d):\n{X_1}")

# -----------------------------------------------------------------
# CLASSIFICATION
# --------------------------------------------------------------------
def classify_output(X_L, classification_threshold):
    n, d = X_L.shape

    # get the first element of the last row
    output_value = X_L[n-1, 0]

    if output_value > classification_threshold:
        return True # output is True if X_L[n, 1] > threshold
    else:
        return False

# Example classification using X_1 as if it were the final layer X_L
classification_result = classify_output(X_1, threshold)
print(f"\nClassification (for 1 layer): {classification_result}")