import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('assets/employees.csv')
df = pd.DataFrame(data)

print("Average salary:", df['Salary'].mean())
print("Salary standard deviation:", f"{df['Salary'].std():,.2f}$") 
print("Minimum age:", df['Age'].min())
print("Maximum age:", df['Age'].max())


plt.figure(figsize=(10, 6))
plt.scatter(df['Age'], df['Salary'], color='red')
plt.plot(
    [df['Age'].min(), df['Age'].max()],
    [df['Salary'].min(), df['Salary'].max()],
    color='gray',
)

plt.title('Salary vs. Age Relationship')
plt.xlabel('Age')
plt.ylabel('Salary')

plt.grid(True)
plt.show()