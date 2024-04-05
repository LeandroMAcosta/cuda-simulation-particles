import matplotlib.pyplot as plt

# Number of threads
no_threads = [1, 2, 4, 8, 16, 32, 48, 96]

# Efficience
efficience = [1.0, 1.0, 1.0, 0.99, 0.95, 0.92, 0.87, 0.73]

# Create graph
plt.figure(figsize=(10, 6))

# Graph of speedup
plt.plot(no_threads, efficience, marker='s', label='Eficiencia')

for i in range(2, len(no_threads)):
    plt.text(no_threads[i], efficience[i], f"{efficience[i]:.2f}", fontsize=9, ha='center', va='bottom')

# Add of labels and title
plt.xlabel('Número de hilos')
plt.ylabel('Eficiencia')
plt.title('Eficiencia en función del número de hilos')

# Add legend
plt.legend()

plt.grid(True)
plt.show()
