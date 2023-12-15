import numpy as np
import matplotlib.pyplot as plt

# p1 = [(50, 78.92), (100, 75.70), (200, 70.88)]
# p2 = [(50, 80.24), (100, 76.98), (200, 72.55)]

# p1 = [(50, 55.94), (100, 108.05), (200, 210.74)]
# p2 = [(50, 66.61), (100, 127.21), (200, 246.73)]

p1 = [(50, 3.96), (100, 5.42), (200, 8.43)]
p2 = [(50, 6.91), (100, 9.06), (200, 13.88)]

x1, y1 = zip(*p1)
x2, y2 = zip(*p2)

coef1 = np.polyfit(x1, y1, 2)
coef2 = np.polyfit(x2, y2, 2)

po1 = np.poly1d(coef1)
po2 = np.poly1d(coef2)

xv1 = np.linspace(0, 300, 100)
yv1 = po1(xv1)

xv2 = np.linspace(0, 300, 100)
yv2 = po2(xv2)

plt.plot(xv1, yv1, label='baseline')
plt.scatter(x1, y1, color='red')
plt.plot(xv2, yv2, label='extension')
plt.scatter(x2, y2, color='red')
plt.xlabel('# of places')
# plt.ylabel('Accuracy (%)')
# plt.title('Predicted Accuracy Plot')
# plt.ylabel('Training time (s)')
# plt.title('Predicted Training Time Plot')
plt.ylabel('Testing time (s)')
plt.title('Predicted Testing Time Plot')
plt.legend()
plt.show()

