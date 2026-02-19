import numpy as np
# -----------------------------------------------------------
# VARIABLES SETUP
# ------------------------------------------------------------
# global parameters
d = 3  # d of feature vectors
d_prime = 2  # dimension after Q/K/V transformation
L = 3

alphabet_w_size = 5 # |Sigma + omega|
W_enc = np.random.rand(alphabet_w_size, d)

# ('a', 'b', 'c', 'omega')
# the indices of the symbols in W_enc:
# 0 -> 'a', 1 -> 'b', 2 -> 'c', 3 -> 'omega', 4 -> 'other'
input_word_indices = [0, 1, 1, 2, 3] # "abbcw"
n = len(input_word_indices)

# |X_0|: n x d
# X_0 = W_enc[input_word_indices, :]
# 
# # Q^l, K^l, V^l: d x d'
# Q_1 = np.random.rand(d, d_prime)
# K_1 = np.random.rand(d, d_prime)
# V_1 = np.random.rand(d, d_prime)
# 
# # O^l: d' x d
# O_1 = np.random.rand(d_prime, d)

# Define X0: abbc
X_0 = np.array([[0,0,1],
               [0,1,0],
               [0,1,0],
               [1,0,0]])

# Define X0: bacb
X_0 = np.array([[0,1,0],
               [0,0,1],
               [1,0,0],
               [0,1,0]])

# Define matrices Q, K, V, O
Q_1 = np.array([[2, 0],
              [0, 2],
              [0, 0]])

K_1 = np.array([[0, 0],
              [0, 2],
              [2, 0]])

V_1 = np.array([[1, 0],
              [0, 0],
              [0, 1]])

O_1 = np.array([[2, 2, 2],
              [3, 1, 0]])

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

    # get the first element of the first row
    output_value = X_L[0, 0]

    if output_value > classification_threshold:
        return True # output is True if X_L[n, 1] > threshold
    else:
        return False

# Example classification using X_1 as if it were the final layer X_L
classification_result = classify_output(X_1, threshold)
print(f"\nClassification (for 1 layer): {classification_result}")


# 1. Alphabet vectors (X_{l-1})
alphabet_vectors = {
    'a': np.array([0, 1, 0]),
    'b': np.array([0, 0, 1]),
    'c': np.array([1, 0, 0])
}

# 2. Matrices Q, K, V, O
Q_1 = np.array([[2, 0], [0, 2], [0, 0]])
K_1 = np.array([[0, 0], [0, 2], [2, 0]])
V_1 = np.array([[1, 0], [0, 0], [0, 1]])
O_1 = np.array([[2, 2, 2], [3, 1, 0]])

def extract_layer_rules(vectors, Q, K, V, O):
    symbols = list(vectors.keys())
    rules = []

    for trigger_sym in symbols:
        x_q = vectors[trigger_sym]
        scores = {}

        # Alpha for every possible target
        for target_sym in symbols:
            x_k = vectors[target_sym]
            # alpha = (x_source * Q) * (x_target * K)
            score = (x_q @ Q) @ (x_k @ K).T
            scores[target_sym] = score

        # Sort targets by score descending, equivalence classes
        sorted_targets = sorted(scores.items(), key=lambda item: item[1], reverse=True)

        print(f"\nTrigger symbol '{trigger_sym}' scores: {scores}")

        for target_sym, score in sorted_targets:
            # Calculate the resulting vector if trigger_sym attends to target_sym
            # Formula: x_new = x_old + ReLU(x_target * V * O)
            x_target = vectors[target_sym]
            update = np.maximum(0, (x_target @ V) @ O)
            new_vector = vectors[trigger_sym] + update

            print(f"  If '{trigger_sym}' attends to '{target_sym}' (score {score}), new vector: {new_vector}")

    return rules

extract_layer_rules(alphabet_vectors, Q_1, K_1, V_1, O_1)


#1st layer a b c 
#2nd laeyr vab, vbc
#3rd vabbc
#4th layer v(8 letters history)