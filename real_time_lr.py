import numpy as np
from sklearn.linear_model import SGDRegressor
import time


model = SGDRegressor(max_iter=1000, alpha=0.01, learning_rate='constant', eta0=0.01, random_state=42)


X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])  # 시간 데이터
y = np.array([20.0, 21.5, 22.7, 23.8, 25.1])  # 해당 시간의 온도데이터

# 모델 초기화 및 학습
model.fit(X, y)
#
while True:
# 현재 시간대의 데이터를 업데이트하고 모델을 학습
    X_new = np.array([[len(X) + 1]])
    # next_time = len(X) + 1

    model.partial_fit(X_new, model.predict(X_new))

    # 데이터 업데이트
    X = np.vstack([X, X_new])

    next_temperature = model.predict(X_new)
    print(f"다음 예상 온도: {next_temperature[0]:.2f} °C")
    if next_temperature >= 40:
        break
    time.sleep(1)
