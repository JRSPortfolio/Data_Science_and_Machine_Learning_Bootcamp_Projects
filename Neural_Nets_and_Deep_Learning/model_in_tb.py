import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import TensorBoard
import tensorboard


df = pd.read_csv('Neural_Nets_and_Deep_Learning/lending_club_loan_engineered.csv', index_col = 0)

X = df.drop('loan_repaid', axis = 1)
y = df['loan_repaid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)


def make_fit_model(title: str, neurons: list, metrics: list, b_size: int, eps: int):
    model_car = title
    log_dir = f'/tflogs_fit/nndl_project/{title}'
    board = TensorBoard(log_dir = log_dir, histogram_freq = 1, write_graph = True, update_freq = 'epoch')
    
    model = Sequential()
    
    for n in neurons:
        model.add(Dense(n, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = metrics)
    model.fit(x = X_train, y = y_train, epochs = eps, validation_data = (X_test, y_test), batch_size = b_size, callbacks = [board])

    predictions = model.predict(X_test)
    predicted_classes = (predictions > 0.5).astype('int32')
    print(title)
    print(f'Classification Report:\n{classification_report(y_test, predicted_classes)}')
    print(f'Confusion Matrix:\n{confusion_matrix(y_test, predicted_classes)}')
    print('____________________\n____________________')

# metrics = ['accuracy', 'precision', 'recall', 'binary_accuracy']
# make_fit_model('512_256_128_64_32_16_8_ep12_bs128', [512, 256, 128, 64, 32, 16, 8], metrics, 128, 12)
# make_fit_model('160_80_40_20_10_5_bs128_ep12', [160, 80, 40, 20, 10, 5], metrics, 128, 12)
# make_fit_model('200_150_100_50_25_10_bs128_ep12', [200, 150, 100, 50, 25, 10], metrics, 128, 12)
# make_fit_model('400_300_200_100_50_25_10_ep12_bs256', [400, 300, 200, 100, 50, 25, 10], metrics, 256, 12)
# make_fit_model('80_60_40_20_10_5_ep12_bs64', [80, 60, 40, 20, 10, 5], metrics, 64, 12)

# metrics = ['accuracy', 'precision', 'recall', 'binary_accuracy']
# make_fit_model('512_256_128_64_32_16_8_ep5_bs128', [512, 256, 128, 64, 32, 16, 8], metrics, 128, 5)
# make_fit_model('512_256_128_64_32_16_8_ep5_bs128_acc-ba', [512, 256, 128, 64, 32, 16, 8], ['accuracy', 'binary_accuracy'], 128, 5)
# make_fit_model('512_256_128_64_32_16_8_ep12_bs128_r', [512, 256, 128, 64, 32, 16, 8], ['recall'], 128, 5)
# make_fit_model('80_60_40_20_10_5_ep4_bs64_acc-ba', [80, 60, 40, 20, 10, 5], ['accuracy', 'binary_accuracy'], 64, 4)
# make_fit_model('80_60_40_20_10_5_ep4_bs64_r', [80, 60, 40, 20, 10, 5], ['recall'], 64, 4)
# make_fit_model('80_60_40_20_10_5_ep4_bs64', [80, 60, 40, 20, 10, 5], metrics, 64, 4)

# make_fit_model('512_256_128_64_32_16_8_ep12_bs128_pr', [512, 256, 128, 64, 32, 16, 8], ['precision', 'recall'], 128, 5)
# make_fit_model('80_60_40_20_10_5_ep4_bs64_pr', [80, 60, 40, 20, 10, 5], ['precision', 'recall'], 64, 4)

# make_fit_model('80_40_20_10_5_ep15_bs32_apr', [80, 40, 20, 10, 5], ['accuracy', 'precision', 'recall'], 32, 15)
# make_fit_model('100_50_25_10_5_ep15_bs32_apr', [100, 50, 25, 10, 5], ['accuracy', 'precision', 'recall'], 32, 15)
# make_fit_model('100_50_10_5_ep15_bs32_apr', [100, 50, 10, 5], ['accuracy', 'precision', 'recall'], 32, 15)
# make_fit_model('128_64_32_16_8_4_ep15_bs32_apr', [128, 64, 32, 16, 8, 4], ['accuracy', 'precision', 'recall'], 32, 15)

# make_fit_model('80_40_20_10_5_ep4_bs32_apr', [80, 40, 20, 10, 5], ['accuracy', 'precision', 'recall'], 32, 4)
# make_fit_model('128_64_32_16_8_4_ep4_bs32_apr', [128, 64, 32, 16, 8, 4], ['accuracy', 'precision', 'recall'], 32, 4)

metrics = ['accuracy', 'precision', 'recall', 'binary_accuracy']
make_fit_model('512_128_32_8_ep50_accpr128', [512, 128, 32, 8], ['accuracy', 'precision', 'recall'], 128, 50)
make_fit_model('512_128_32_8_ep50_accba128', [512, 128, 32, 8], ['accuracy', 'binary_accuracy'], 128, 50)
make_fit_model('80_40_5_ep4_ep50_accpr64', [80, 40, 5], ['accuracy', 'precision', 'recall'], 64, 50)
make_fit_model('80_40_5_ep4_ep50_accba64', [80, 40, 5], ['accuracy', 'binary_accuracy'], 64, 50)
