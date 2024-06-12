import datetime, warnings, scipy
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import math
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Read csv file
df = pd.read_csv("Delhi/aqi_calculated.csv", parse_dates=["Datetime"])

# log transformation to deal with skewed data
aqi = np.log1p(df[["AQI_calculated"]].values)
aqi.shape

# create new dataframe to compare the original vs log transform data
dist_df = pd.DataFrame({"AQI": df["AQI_calculated"].values, "log_AQI": aqi[:, 0]})

# histogram plot original vs log transform data
plt.figure(figsize=(12, 5))
dist_df.hist()
plt.savefig("histogram.png", format="png")

# split into train and test sets
# 80% for training, 20% for testing
train_size = int(len(aqi) * 0.8)
test_size = len(aqi) - train_size
train, test = aqi[0:train_size, :], aqi[train_size : len(aqi), :]
print(len(train), len(test))


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i : (i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


# reshape into X=t and Y=t+1
look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

print("Shape of trainX :", trainX.shape)
print("Shape of trainY :", trainY.shape)
print("Shape of testX :", testX.shape)
print("Shape of testY :", testY.shape)

# reshape the input array to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

print("Shape of trainX :", trainX.shape)
print("Shape of testX :", testX.shape)

# Build the LSTM model
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(1, look_back)))
model.add(Dropout(0.1))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(16))
model.add(Dense(1))

# Compile the model
model.compile(optimizer="adam", loss="mean_squared_error")

# Train the model
model.fit(trainX, trainY, batch_size=1, epochs=20)

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

train_loss = model.evaluate(trainX, trainY)
test_loss = model.evaluate(testX, testY)
print(f"Training Loss: {train_loss}")
print(f"Testing Loss: {test_loss}")


# Mean Absolute Error
mae = mean_absolute_error(testY, testPredict)
# Root Mean Squared Error
rmse = np.sqrt(mean_squared_error(testY, testPredict))
# R-squared
r2 = r2_score(testY, testPredict)

print(f"MAE: {mae}")
print(f"RMSE: {rmse}")
print(f"R²: {r2}")

# Loss Curves
# plt.figure(figsize=(12,5))
# plt.plot(history.history['loss'], label='Train Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
# plt.savefig("loss_curves", format="png")


# invert predictions
trainPredict = np.expm1(trainPredict)
trainY = np.expm1(trainY)
testPredict = np.expm1(testPredict)
testY = np.expm1(testY)

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY, trainPredict[:, 0]))
print("Train Score: %.2f RMSE" % (trainScore))
testScore = math.sqrt(mean_squared_error(testY, testPredict[:, 0]))
print("Test Score: %.2f RMSE" % (testScore))

test_series = pd.Series(testY)

# state of model performance
if testScore < test_series.std():
    print("\n[ Model performance is GOOD enough ]")
    print("\nRMSE of test prediction < Standard deviation of test dataset")
    print("%.2f" % (testScore), "<", "%.2f" % (test_series.std()))
else:
    print("\n[ Model performance is NOT GOOD enough ]")
    print("\nRMSE of test prediction > Standard deviation of test dataset")
    print("%.2f" % (testScore), ">", "%.2f" % (test_series.std()))

train_aqi = pd.DataFrame(trainPredict, columns=["AQI"])
test_aqi = pd.DataFrame(testPredict, columns=["AQI"])
print(len(train_aqi))
print(len(test_aqi))

# Save predicted result
train_aqi.to_csv("Delhi/train_aqi.csv", index=False)
test_aqi.to_csv("Delhi/test_aqi.csv", index=False)

# Draw graph
# shift train predictions for plotting
trainPredictPlot = np.empty_like(aqi)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back : len(trainPredict) + look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(aqi)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict) + (look_back * 2) + 1 : len(aqi) - 1, :] = testPredict

# plot original dataset and predictions
time_axis = np.linspace(0, aqi.shape[0] - 1, 15)
time_axis = np.array([int(i) for i in time_axis])
time_axisLab = np.array(df.index, dtype="datetime64[D]")

fig = plt.figure()
ax = fig.add_axes([0, 0, 2.1, 2])
ax.plot(np.expm1(aqi), label="Original Dataset")
ax.plot(trainPredictPlot, color="green", label="Train Prediction")
ax.plot(testPredictPlot, color="red", label="Test Prediction")
ax.set_xticks(time_axis)
ax.set_xticklabels(time_axisLab[time_axis], rotation=45)
ax.set_xlabel("\nDate", fontsize=27, fontweight="bold")
ax.set_ylabel("AQI", fontsize=27, fontweight="bold")
ax.legend(loc="best", prop={"size": 20})
ax.tick_params(size=10, labelsize=15)
# ax.set_xlim([-1, 1735])

ax1 = fig.add_axes([2.3, 1.3, 1, 0.7])
ax1.plot(np.expm1(aqi), label="Original Dataset")
ax1.plot(testPredictPlot, color="red", label="Test Prediction")
ax1.set_xticks(time_axis)
ax1.set_xticklabels(time_axisLab[time_axis], rotation=45)
ax1.set_xlabel("Date", fontsize=27, fontweight="bold")
ax1.set_ylabel("AQI", fontsize=27, fontweight="bold")
ax1.tick_params(size=10, labelsize=15)
ax1.set_xlim([38000, 49000])

plt.savefig("result.png", format="png")
