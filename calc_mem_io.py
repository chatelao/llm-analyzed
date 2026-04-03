
# Model parameters for Llama 3.1 405B
L = 126
d_model = 16384
d_ff = 53248
h = 128
h_kv = 8
V = 128256
N = 1000000
d_head = d_model // h

# Step A: Attention Projection
# Q, K, V, O projections
w_attn_proj = 2 * d_model**2 + 2 * (d_model * d_head * h_kv)
macs_a = N * w_attn_proj
fetch_a = w_attn_proj + N * d_model
store_a = N * d_model

# Step B: Attention Mechanism
macs_b = 2 * (N**2 * d_model)
fetch_b = N * d_model + h * N**2
store_b = h * N**2 + N * d_model

# Step C: MLP (SwiGLU)
w_mlp = 3 * d_model * d_ff
macs_c = N * w_mlp
fetch_c = w_mlp + N * d_model
store_c = N * d_model

# Step D: Output Layer
w_output = d_model * V
macs_d = N * d_model * V
fetch_d = w_output + N * d_model
store_d = N * V

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
print(f"Total MACs: {total_macs:.2e}")
print(f"Total Fetch: {total_fetch:.2e}")
print(f"Total Store: {total_store:.2e}")
print(f"Total I/O: {total_fetch + total_store:.2e}")

print("\nBreakdown per Component (Totals across L layers):")
print(f"Linear (Proj + MLP): Fetch={total_fetch_linear:.2e}, Store={total_store_linear:.2e}")
print(f"Attention (quadr.): Fetch={total_fetch_attn:.2e}, Store={total_store_attn:.2e}")
print(f"Output Head: Fetch={total_fetch_output:.2e}, Store={total_store_output:.2e}")
