import matplotlib.pyplot as plt

# Data
batch_size = [1, 4, 8, 16, 32, 48, 64]
autoregressive_throughput = [54.49, 190.23, 352.20, 618.95, 766.76, 1053.54, 892.54]
eagle3_throughput = [89.01, 292.16, 458.29, 720.60, 710.99, 891.97, 772.23]
sssd_throughput = [109.88, 336.04, 569.02, 916.10, 1093.17, 1210.40, 943.36]

# Compute latency = batch_size / throughput
latency_ar  = [b / t for b, t in zip(batch_size, autoregressive_throughput)]
latency_e3  = [b / t for b, t in zip(batch_size, eagle3_throughput)]
latency_sss = [b / t for b, t in zip(batch_size, sssd_throughput)]

# Plot
plt.figure(figsize=(8, 5))

# Autoregressive
plt.plot(autoregressive_throughput, latency_ar,
         marker='o', linestyle='-', label='Autoregressive')
for i, (x, y) in enumerate(zip(autoregressive_throughput, latency_ar)):
    plt.annotate(str(batch_size[i]), (x, y),
                 textcoords="offset points", xytext=(5,5), ha='left')

# Eagle3
plt.plot(eagle3_throughput, latency_e3,
         marker='o', linestyle='-', label='Eagle3')
for i, (x, y) in enumerate(zip(eagle3_throughput, latency_e3)):
    plt.annotate(str(batch_size[i]), (x, y),
                 textcoords="offset points", xytext=(5,5), ha='left')

# SSSD
plt.plot(sssd_throughput, latency_sss,
         marker='o', linestyle='-', label='SSSD')
for i, (x, y) in enumerate(zip(sssd_throughput, latency_sss)):
    plt.annotate(str(batch_size[i]), (x, y),
                 textcoords="offset points", xytext=(5,5), ha='left')

# Labels & legend
plt.xlabel('Throughput (tok/s)')
plt.ylabel('Latency (s/tok)')
plt.title('Latency vs. Throughput for Different Speculative Methods')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig("bel_plt.png")