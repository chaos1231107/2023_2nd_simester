import numpy as np
from sklearn.linear_model import SGDRegressor
import time
import matplotlib.pyplot as plt

model = SGDRegressor(max_iter=1000, alpha=0.01, learning_rate='constant', eta0=0.01, random_state=42)

x = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])  # 시간 데이터
y = np.array([20.0, 21.5, 22.7, 23.8, 25.1])  # 해당 시간의 온도데이터

# 모델 초기화 및 학습
model.fit(x, y)

regression_data = list(y)  # 기존 데이터를 리스트에 추가

x_new = len(x) + 1

while True:
    # 현재 시간대의 데이터를 업데이트하고 모델을 학습
    x_new += 1
    model.partial_fit(np.array([[x_new]]), model.predict(np.array([[x_new]])))

    next_temperature = model.predict(np.array([[x_new]]))
    print(f"다음 예상 온도: {next_temperature[0]:.2f} °C")

    # 회귀된 값을 리스트에 저장
    regression_data.append(next_temperature[0])

    if next_temperature >= 40:
        break
    time.sleep(1)

# 회귀된 직선 그리기
x_plot = np.array([[1.0], [6.0]])  # 그래프를 그릴 범위 설정
y_plot = model.predict(x_plot)

plt.plot(range(1, len(regression_data) + 1), regression_data, label='predict data', marker='x')
plt.plot(x_plot, y_plot, color='red', linestyle='--', label='regression line')

plt.scatter(range(1, len(y) + 1), y, color='blue', label='real temperature', marker='o')

plt.legend()
plt.show()
