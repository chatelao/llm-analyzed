
# Model parameters for Llama 3.1 405B
L = 126
d_model = 16384
d_ff = 53248
h = 128
h_kv = 8
V = 128256
N = 1000000
d_head = d_model // h

# Step A: Attention Projection (Q, K, V, O)
# Weights
w_q = d_model**2
w_k = d_model * (d_head * h_kv)
w_v = d_model * (d_head * h_kv)
w_o = d_model**2
w_attn_proj = w_q + w_k + w_v + w_o

macs_a = N * w_attn_proj

# Activations
act_q = N * d_model
act_k = N * (d_head * h_kv)
act_v = N * (d_head * h_kv)
act_attn_out = N * d_model # Output of Attention Mechanism (Input to O-Proj)
act_x = N * d_model # Input to the whole layer
act_o = N * d_model # Output of O-Proj

# Fetch: Weights + Input + Attn_out (for O-proj)
fetch_a = w_attn_proj + act_x + act_attn_out
# Store: Q, K, V + Final O
store_a = act_q + act_k + act_v + act_o

# Step B: Attention Mechanism
# QK^T -> Scores
macs_b1 = N**2 * d_model # (N x d) * (d x N) per head? No, sum over heads.
# Actually MACs for QK^T: N * N * d_head * h = N^2 * d_model
# Softmax and context: Scores * V -> Context
macs_b2 = N**2 * d_model

macs_b = macs_b1 + macs_b2

# Memory I/O
act_scores = h * N**2
# Fetch: Q, K + Scores (for V-mult) + V
fetch_b = act_q + act_k + act_scores + act_v
# Store: Scores + AttnOut
store_b = act_scores + act_attn_out

# Step C: MLP (SwiGLU)
# Weights: Gate, Up, Down
w_mlp = 3 * d_model * d_ff
macs_c = N * w_mlp

act_mlp_in = N * d_model
act_mlp_hidden = N * d_ff # Intermediate activation after Gate/Up
act_mlp_out = N * d_model

# Fetch: Weights + Input + Hidden (for Down-proj)
fetch_c = w_mlp + act_mlp_in + act_mlp_hidden
# Store: Hidden + Output
store_c = act_mlp_hidden + act_mlp_out

# Step D: Output Layer
w_output = d_model * V
macs_d = N * d_model * V

act_final_in = N * d_model
act_logits = N * V

# Fetch: Weights + Input
fetch_d = w_output + act_final_in
# Store: Logits
store_d = act_logits

# Totals across all layers
total_macs_linear = L * (macs_a + macs_c)
total_macs_attn = L * macs_b
total_macs_output = macs_d
total_macs = total_macs_linear + total_macs_attn + total_macs_output

total_fetch_linear = L * (fetch_a + fetch_c)
total_fetch_attn = L * fetch_b
total_fetch_output = fetch_d
total_fetch = total_fetch_linear + total_fetch_attn + total_fetch_output

total_store_linear = L * (store_a + store_c)
total_store_attn = L * store_b
total_store_output = store_d
total_store = total_store_linear + total_store_attn + total_store_output

print(f"Results for N = {N}:")
print(f"Total MACs: {total_macs:.6e}")
print(f"Total Fetch: {total_fetch:.6e}")
print(f"Total Store: {total_store:.6e}")
print(f"Total I/O: {total_fetch + total_store:.6e}")

print("\nBreakdown per Component (Totals across L layers):")
print(f"Linear (Proj + MLP): Fetch={total_fetch_linear:.6e}, Store={total_store_linear:.6e}")
print(f"Attention (quadr.): Fetch={total_fetch_attn:.6e}, Store={total_store_attn:.6e}")
print(f"Output Head: Fetch={total_fetch_output:.6e}, Store={total_store_output:.6e}")
