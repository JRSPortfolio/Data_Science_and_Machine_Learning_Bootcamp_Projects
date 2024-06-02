import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.models import Sequential #type: ignore
from tensorflow.keras.layers import Dense #type: ignore
from tensorflow.keras.optimizers import Adam  #type: ignore
from tensorflow.keras.callbacks import TensorBoard #type: ignore
from tensorflow.keras.metrics import  BinaryAccuracy, FalseNegatives, FalsePositives, TrueNegatives, TruePositives, Precision, Recall, AUC #type: ignore
from tensorflow.keras.callbacks import Callback #type: ignore
import tensorflow as tf
import tensorboard
from tensorboard.plugins.hparams import api as hp
from datetime import datetime as dt


def get_data_splits():
    df = pd.read_csv('Neural_Nets_and_Deep_Learning/lending_club_loan_engineered.csv', index_col = 0)

    X = df.drop('loan_repaid', axis = 1)
    y = df['loan_repaid']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 101)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.fit_transform(X_test)
    
    return X_train, X_test, y_train, y_test

class MakeParams(dict):
    def __init__(self, epochs: int, units: list, l_rate: list, b_size: list):
        super(MakeParams, self).__init__()
        self['epochs'] = epochs
        self['num_units'] = hp.HParam(f'Dense_Layer_{units}', hp.Discrete(units))
        self['l_rate'] = hp.HParam('Learnin_Rate', hp.Discrete(l_rate))    
        self['b_size'] = hp.HParam('Batch_Size', hp.Discrete(b_size))

def tune_model_params(X_train, X_test, y_train, y_test, params: MakeParams):    
    metrics = [TruePositives(name = 'true_positives'), FalsePositives(name = 'false_positives'), TrueNegatives(name = 'true_negatives'), FalseNegatives(name = 'false_negatives'),
               BinaryAccuracy(name = 'accuracy'), Precision(name = 'precision'), Recall(name = 'recall'), AUC(name = 'auc')]

    log_dir = f"Neural_Nets_and_Deep_Learning/tflogs_fit/{dt.now().strftime('%d-%m-%Y_%H%M')}"
    
    num_units = params['num_units'].domain.values
    print(num_units)
    epochs = params['epochs']
        
    for b_size in params['b_size'].domain.values:
        for rate in params['l_rate'].domain.values:
            loss, accuracy, precision, recall, hparams = make_model(log_dir, b_size, rate, num_units, epochs, metrics, X_train, y_train, X_test, y_test)
            eval_model(log_dir, b_size, rate, len(num_units), epochs, loss, accuracy, precision, recall, hparams)
                 
def make_model(logdir: str, batch_size, learning_rate, num_units, epochs, metrics: list, X_train, y_train, X_test, y_test):
    model = Sequential()
    layer = 0
    hparams = {}
    for units in num_units:
        hparams[f'Layer_{str(layer)}'] = units 
        layer += 1
        model.add(Dense(units, activation = 'relu'))       
    model.add(Dense(1, activation = 'sigmoid'))
    
    hparams[f'Layer_{str(layer)}'] = 1
    board = TensorBoard(log_dir = f'{logdir}/cb/bs{batch_size}_lr{learning_rate}_l{units}', histogram_freq = 1, write_graph = True, update_freq = 'epoch')
    
    model.compile(loss = 'binary_crossentropy', optimizer = Adam(learning_rate = learning_rate), metrics = metrics)
    model.fit(x = X_train, y = y_train, epochs = epochs, validation_data = (X_test, y_test), batch_size = batch_size, callbacks = [board])
    
    loss, _, _, _, _, accuracy, precision, recall, _ = model.evaluate(X_test, y_test)
    return loss, accuracy, precision, recall, hparams

def eval_model(logdir, batch_size, learning_rate, num_units, epochs, loss, accuracy, precision, recall, hparams):
    file_writer = tf.summary.create_file_writer(f'{logdir}/fw/bs{batch_size}_lr{learning_rate}_l{num_units}')
    with file_writer.as_default():
        hyperp = {'Neurons' : num_units,
                  'Epochs' : epochs,
                  'Learnin_Rate' : learning_rate,
                  'Batch_Size' : batch_size}
        hparams.update(hyperp)
        hp.hparams(hparams)

        tf.summary.scalar('Loss', loss, step = 1)
        tf.summary.scalar('Accuracy', accuracy, step = 1)
        tf.summary.scalar('Precision', precision, step = 1)
        tf.summary.scalar('Recall', recall, step = 1)
    
    
def predict_eval_model(model: Sequential, X_test, y_test):
    predictions = model.predict(X_test)
    predicted_classes = (predictions > 0.5).astype('int32')
    print(f'Classification Report:\n{classification_report(y_test, predicted_classes)}')
    print(f'Confusion Matrix:\n{confusion_matrix(y_test, predicted_classes)}')
    print('____________________\n____________________')


def main():
    X_train, X_test, y_train, y_test = get_data_splits()
    
    params = MakeParams(5, [10, 20, 30], [0.01, 0.001], [128])
    tune_model_params(X_train, X_test, y_train, y_test, params)

if __name__ == '__main__':
    main()
    # predict_eval_model(model, X_test, y_test)
