import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Theoretical roofline

# TFLOP/s, TByte/s
hardware_configs = {
    '910B2': (376, 1.6),
    '910B4': (280, 0.8),
    '910B4_real': (240, 0.65),
    'A100': (312, 1.935),
    'A6000': (164, 0.7)
}

device_model = '910B4_real'

# [M, K] x [K, N] -> [M, N]

def compute_bound_perf(m, n, k, comp):
    # output unit: us (1e-6 sec)
    flops = 2 * m * n * k
    t = flops / comp / 1e9
    return t

def memory_bound_perf(m, n, k, bw):
    # output unit: us (1e-6 sec)
    io = (m*k + n*k + m*n)*2  # FP16
    t = io / bw / 1e9
    return t

# llama2-7b config
# Q, K, V, OUT in Attention layer
attn_proj_shape = {
    'k': 4096,
    'n': 4096
}

# up, gate, down in MLP layer
mlp_proj_shape = {
    'k': 4096,
    'n': 14336
}


def time_compute_bound(proj_shape, m_list, comp):
    k = proj_shape['k']
    n = proj_shape['n']
    m = m_list
    return compute_bound_perf(m, n, k, comp)

def time_memory_bound(proj_shape, m_list, bw):
    k = proj_shape['k']
    n = proj_shape['n']
    m = m_list
    return memory_bound_perf(m, n, k, bw)

def time_roofline(proj_shape, m_list, hardware_config):
    time_comp = time_compute_bound(proj_shape, m_list, hardware_config[0])
    time_mem = time_memory_bound(proj_shape, m_list, hardware_config[1])
    # NOTE: take the slower time; here the unit is time(sec), not FLOPs/s
    time_roofline = np.maximum(time_comp, time_mem)
    return time_roofline

def time_roofline_sdpa_compute_bound(m, s_kv, proj_shape, bs, hardware_config):
    # Always compute bound until very large sequence
    s_q = m/bs
    n = proj_shape['n']
    flops = 2 * m * (s_q + s_kv) * n
    comp = hardware_config[0]
    t = flops / comp / 1e9
    return t

def time_roofline_sdpa_memory_bound(m, s_kv, proj_shape, bs, hardware_config):
    # Always compute bound until very large sequence
    s_q = m/bs
    n = proj_shape['n']
    bw = hardware_config[1]
    io = (bs*n*(2*s_kv + 4*s_q))*2  # FP16
    t = io / bw / 1e9
    return t

def time_roofline_sdpa(m, s_kv, proj_shape, bs, hardware_config):
    time_mem = time_roofline_sdpa_memory_bound(m, s_kv, proj_shape, bs, hardware_config)
    time_comp = time_roofline_sdpa_compute_bound(m, s_kv, proj_shape, bs, hardware_config)
    return np.maximum(time_comp, time_mem)

df = pd.read_csv("./spec_forward_timing_batch8_kv1024.csv")
query_lengths = df.query_length.values

bs = 8
m_list = np.array([q*bs for q in query_lengths])
print(m_list)
context_len = 1024

time_roofline_mlp_proj = time_roofline(mlp_proj_shape, m_list, hardware_configs[device_model])
time_roofline_attn_proj = time_roofline(attn_proj_shape, m_list, hardware_configs[device_model])
time_roofline_fa = time_roofline_sdpa(m_list, context_len, attn_proj_shape, bs, hardware_configs[device_model])

print(time_roofline_mlp_proj)
print(time_roofline_attn_proj)
print(time_roofline_fa)

total_mlp = [3*x for x in time_roofline_mlp_proj]
total_att_proj = [4*x for x in time_roofline_attn_proj]
matmul_total = [a + b for a, b in zip(total_mlp, total_att_proj)]

num_layers = 32

sns.set_style("whitegrid")


plt.rcParams.update({
    'font.size': 14,          # Increase overall font size
    'axes.labelsize': 16,     # Axis label font size
    'axes.titlesize': 18,     # Title font size
    'legend.fontsize': 14,    # Legend font size
    'xtick.labelsize': 12,    # X tick label font size
    'ytick.labelsize': 12,    # Y tick label font size
    # 'lines.linewidth': 2.5,   # Line width for plot lines
    # 'axes.linewidth': 1.5,    # Axis line width
})


# Recompute more detailed for sharper lines
x_points = list(range(1, 129))
m_list = np.array([q*bs for q in x_points])
print(m_list)
time_roofline_mlp_proj = time_roofline(mlp_proj_shape, m_list, hardware_configs[device_model])
time_roofline_attn_proj = time_roofline(attn_proj_shape, m_list, hardware_configs[device_model])


fig, ax_left = plt.subplots()

# Setting up right y-axis
ax_right = ax_left.twinx()

# Seaborn color palette
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

print(len(query_lengths))
print(len(time_roofline_fa))
print(len(matmul_total))

ax_left.stackplot(
    query_lengths, [t*num_layers for t in time_roofline_fa], [t*num_layers for t in matmul_total],
    alpha=0.5
    )

ax_left.legend(['FA', 'Linear'], loc='upper left')

ax_right.plot(x_points, time_roofline_attn_proj, colors[2])
ax_right.plot(x_points, time_roofline_mlp_proj, colors[3])
ax_right.legend(['Q/K/V proj', 'Up/Down proj'], title="Single operations per\nlayer (Right y axis)",  loc='upper right')

ax_left.set_xlabel('Speculation length ($s_q$)')
ax_left.set_ylabel('Forward pass time (ms)')
ax_right.set_ylabel('Single-operation time (ms)')
ax_left.set_ylim(0,110)

ax_left.grid(True)
ax_right.grid(False)

plt.tight_layout()

plt.savefig("theoretical_stacked.png", dpi=100)



# VERY LONG CONTEXT
bs = 8
m_list = np.array([q*bs for q in query_lengths])
context_len = 10000

time_roofline_mlp_proj = time_roofline(mlp_proj_shape, m_list, hardware_configs[device_model])
time_roofline_attn_proj = time_roofline(attn_proj_shape, m_list, hardware_configs[device_model])
time_roofline_fa = time_roofline_sdpa(m_list, context_len, attn_proj_shape, bs, hardware_configs[device_model])

print(time_roofline_mlp_proj)
print(time_roofline_attn_proj)
print(time_roofline_fa)

total_mlp = [3*x for x in time_roofline_mlp_proj]
total_att_proj = [4*x for x in time_roofline_attn_proj]
matmul_total = [a + b for a, b in zip(total_mlp, total_att_proj)]

num_layers = 32

plt.figure()

# Seaborn color palette
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
print(colors)


print(len(query_lengths))
print(len(time_roofline_fa))
print(len(matmul_total))
plt.stackplot(
    query_lengths, [t*num_layers for t in time_roofline_fa], [t*num_layers for t in matmul_total],
    alpha=0.5
    )

plt.legend(['FA', 'Linear'], loc='upper left')

plt.xlabel('Speculation length ($s_q$)')
plt.ylabel('Forward pass time (ms)')
plt.ylim(0,110)

plt.grid(True)

plt.tight_layout()

plt.savefig("theoretical_stacked_10k.png", dpi=100)