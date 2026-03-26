import torch
import json
import numpy as np
import pandas as pd
from transformer import SimplifiedTransformer

def lazy_verification():
    model_path = r'../best_model.pt'
    json_path = r'formula_33.json'

    # 1. Load vocab
    with open(json_path, 'r') as f:
        alphabet = json.load(f)['metadata']['alphabet']
    symbol_to_idx = {symbol: idx for idx, symbol in enumerate(alphabet)}
    symbol_to_idx['omega'] = len(alphabet)

    # 2. Setup pyTorch model
    model = SimplifiedTransformer(vocab_size=len(alphabet)+1, d_model=64, d_prime=32, num_layers=3)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # 3. Dynamic test word
    df = pd.read_csv('dataset_33.csv')

    # Generate a random number between 0 and the length of the dataset
    random_idx = np.random.randint(0, len(df))
    random_sequence = str(df.iloc[random_idx]['sequence'])

    test_word = random_sequence.split(',') + ['omega']
    print(f"Testing Word (from Row {random_idx}): {test_word}")

    indices = [symbol_to_idx[s] for s in test_word]
    tensor_input = torch.tensor([indices], dtype=torch.long)

    # 4. LAZY TRACE (Simulating rules ONLY for this word)
    with torch.no_grad():
        # Get PyTorch matrices
        X_pytorch = model.W_enc[tensor_input] #Shape [1, seq_len, d_model]

        # Symbolic history names
        history_names = test_word.copy()

        # Extract numpy matrices
        W_enc = model.W_enc.detach().numpy()
        X_numpy = W_enc[indices] # Shape [seq_len, d_model]

        for l in range(3): # Up to layer 3
            print(f"\n--- Layer {l+1} ---")

            #pt
            Q_pt = model.Q_layers[l](X_pytorch)
            K_pt = model.K_layers[l](X_pytorch)
            V_pt = model.V_layers[l](X_pytorch)
            scores_pt = torch.bmm(Q_pt, K_pt.transpose(1, 2))
            A_pt = model.leftmost_argmax(scores_pt)
            attended_pt = torch.bmm(A_pt, V_pt)
            output_pt = model.O_layers[l](attended_pt)
            X_pytorch = torch.relu(output_pt) + X_pytorch

            #np
            Q_np = model.Q_layers[l].weight.detach().numpy().T
            K_np = model.K_layers[l].weight.detach().numpy().T
            V_np = model.V_layers[l].weight.detach().numpy().T
            O_np = model.O_layers[l].weight.detach().numpy().T

            new_history_names = []
            new_X_numpy = np.zeros_like(X_numpy)

            # For each position in our specific word
            for i in range(len(test_word)):
                trigger_name = history_names[i]
                trigger_vec = X_numpy[i]

                # Calculate scores against all symbols IN THE SEQUENCE
                scores = []
                for j in range(len(test_word)):
                    target_vec = X_numpy[j]
                    score = (trigger_vec @ Q_np) @ (target_vec @ K_np).T
                    scores.append(score)

                # LTL Leftmost Hardmax Logic: Find max score, pick the first one
                max_score = max(scores)
                target_idx = scores.index(max_score)

                target_name = history_names[target_idx]
                target_vec = X_numpy[target_idx]

                # Symbolic update: x_new = x_old + ReLU(target * V * O)
                update = np.maximum(0, (target_vec @ V_np) @ O_np)
                new_vec = trigger_vec + update
                new_X_numpy[i] = new_vec

                # Create history name
                if l == 0:
                    new_name = f"v_{trigger_name},{target_name}"
                else:
                    target_history = target_name.split('_', 1)[1] if '_' in target_name else target_name
                    new_name = f"{trigger_name}_{target_history}"

                new_history_names.append(new_name)
                print(f"Pos {i}: {trigger_name} attends to {target_name} --> {new_name}")

            history_names = new_history_names
            X_numpy = new_X_numpy

    # 5. The check
    print("\nVERIFICATION RESULTS(for position 0")

    pt_final_vec = X_pytorch[0, 0, :].numpy()
    np_final_vec = X_numpy[0]

    is_match = np.allclose(pt_final_vec, np_final_vec, atol=1e-5)

    print(f"Final History Name: {history_names[0]}")
    print(f"PyTorch Vector (First 5): {pt_final_vec[:5]}")
    print(f"Symbolic Vector (First 5): {np_final_vec[:5]}")
    print(f"\nDo they match excatly? --> {'YES!' if is_match else 'NO'}")

if __name__ == "__main__":
    lazy_verification()