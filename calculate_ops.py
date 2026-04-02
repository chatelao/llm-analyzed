L = 126
d_model = 16384
d_ff = 53248
h = 128
h_kv = 8
d_head = 128
V = 128256
N = 10**6

# --- linear (Proj + MLP) ---
# Weights
w_proj = 2 * d_model**2 + 2 * d_model * d_head * h_kv
w_mlp = 3 * d_model * d_ff
weights_linear_total = L * (w_proj + w_mlp)

# Proj Activations
proj_fetch = 2 * N * d_model # Read X, Read Attn_Out
proj_store = 2 * N * d_model + 2 * N * d_head * h_kv # Write Q,K,V, Write O_proj

# MLP Activations
mlp_fetch = N * (d_model + 3 * d_ff) # Read X, Read Gate,Up, Read Intermediate
mlp_store = N * (d_model + 3 * d_ff) # Write Gate,Up, Write Intermediate, Write Down

linear_fetch = weights_linear_total + L * (proj_fetch + mlp_fetch)
linear_store = L * (proj_store + mlp_store)

# --- Attention Mechanism ---
# Fetch Q, K, V activations + Score matrix S (twice: for softmax and SV)
attn_mech_fetch = L * (N * (d_model + 2 * d_head * h_kv) + 2 * h * N**2)
# Store S (twice: from QKT and softmax) + Store Context
attn_mech_store = L * (2 * h * N**2 + N * d_model)

# --- Output Head ---
w_output = d_model * V
output_fetch = w_output + N * d_model
output_store = N * V

print(f"Linear Fetch: {linear_fetch:.2e}")
print(f"Linear Store: {linear_store:.2e}")
print(f"Attention Mech Fetch: {attn_mech_fetch:.2e}")
print(f"Attention Mech Store: {attn_mech_store:.2e}")
print(f"Output Fetch: {output_fetch:.2e}")
print(f"Output Store: {output_store:.2e}")
