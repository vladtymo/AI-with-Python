# 1
import numpy as np
import matplotlib.pyplot as plt
 
x = np.linspace(-2 * np.pi, 2 * np.pi, 500)
 
f = np.sin(x)
g = np.cos(x)
h = np.sin(x) + np.cos(x)
 
 
plt.figure(figsize=(10, 6))
plt.plot(x, f, label='f(x) = sin(x)', color='blue')
plt.plot(x, g, label='g(x) = cos(x)', color='green')
plt.plot(x, h, label='h(x) = sin(x) + cos(x)', color='red')
 
plt.title('Графіки функцій f(x), g(x), h(x)')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
 
plt.show()
 
 
# 2
np.random.seed(42)
 
group_A = np.random.normal(loc=70, scale=10, size=100)
group_B = np.random.normal(loc=80, scale=5, size=100)
group_C = np.random.normal(loc=65, scale=15, size=100)
 
data = [group_A, group_B, group_C]
labels = ['Група A', 'Група B', 'Група C']
 
plt.figure(figsize=(8, 6))
plt.boxplot(data, labels=labels, patch_artist=True,
            boxprops=dict(facecolor="lightblue"),
            medianprops=dict(color="red"))
 
plt.title("Box-plot оцінок для трьох груп студентів")
plt.ylabel("Оцінка")
plt.grid(True)
plt.show()
 
 
# 3
x = np.random.uniform(0, 1, 100)
y = np.random.uniform(0, 1, 100)
 
 
plt.figure(figsize=(6, 6))
plt.scatter(x, y, color='green', alpha=0.6)
 
plt.xlabel("X значення")
plt.ylabel("Y значення")
plt.title("Точкова діаграма з рівномірним розподілом")
plt.grid(True)
plt.show()