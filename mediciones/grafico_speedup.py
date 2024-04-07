import matplotlib.pyplot as plt

# Number of threads
no_threads = [1, 2, 4, 8, 16, 32, 48, 96]

# Speedup
speedup_ideal = [1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 48.0, 96.0]
speedup = [1.0, 2.0, 4.0, 7.91, 15.13, 29.57, 41.73, 69.67]

# Create graph
plt.figure(figsize=(10, 6))

# Graph of speedup
plt.plot(no_threads, speedup, marker='o', label='Speedup')

# Graph of ideal speedup
plt.plot(no_threads, speedup_ideal, marker='s', label='Speedup ideal')

for i in range(len(no_threads)):
    plt.text(no_threads[i], speedup[i], f"{speedup[i]:.2f}", fontsize=9, ha='center', va='bottom')

# Add of labels and title
plt.xlabel('Número de hilos')
plt.ylabel('Speedup')
plt.title('Speedup y Speedup ideal en función del número de hilos')

# Add legend
plt.legend()

plt.grid(True)
plt.show()
