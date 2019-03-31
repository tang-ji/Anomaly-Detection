import keras
from keras import regularizers
from keras.activations import softplus
from keras.layers import *
from keras.models import Model
from keras.losses import binary_crossentropy

import pydotplus
import os
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

""" 
===============================================
Tree regularization network class
===============================================
"""
class TreeMLP(object):
    def __init__(self, in_count, out_count, hidden_sizes, strength=0.01, APL_mode=0):
        self.mlp = MLP(in_count, hidden_sizes, out_count, APL_mode = APL_mode)
        self.sur = surrogate(self.mlp.num_weights, 1)
        self.strength = strength
        self.tree = []

    def loss(self, y_true,y_pred):    
        path_length = self.sur.predict(np.array([[i for l in self.mlp.model.get_weights() for i in l.flatten()]])).ravel()[0]
        return binary_crossentropy(y_true,y_pred) + self.strength * path_length

    def train(self, X_train, y_train, iters_retrain=5, epochs_mlp=3, epochs_sur=5, batch_size=256, feature_names=None, validation_data=None, class_names=None):
        for i in range(iters_retrain):
            self.mlp.model.compile(optimizer='adam', loss=self.loss, metrics=['accuracy'])
            print('training MLP net... [%d/%d]' % (i + 1, iters_retrain))
            self.mlp.train(X_train, y_train, epochs=epochs_mlp, batch_size=batch_size, validation_data=validation_data)
            print('training surrogate net... [%d/%d]' % (i + 1, iters_retrain))
            self.sur.train(np.array(self.mlp.saved_weights), self.mlp.saved_apl, batch_size=16, epochs=epochs_sur)
            self.tree.append(self.mlp.fit_tree(X_train, y_train))
            if not os.path.isdir('./tree'):
                os.mkdir('./tree')
            nodes = visualize(self.tree[-1], './tree/tree' + str(i) + '.pdf',False,feature_names, class_names)
            acc = accuracy_score(self.tree[-1].predict(validation_data[0]), [np.argmax(x) for x in validation_data[1]])
            
            leaf_indices = self.tree[-1].apply(validation_data[0])
            leaf_counts = np.bincount(leaf_indices)
            leaf_i = np.arange(self.tree[-1].tree_.node_count)
            apl = np.dot(leaf_i, leaf_counts) / float(validation_data[0].shape[0])
            if self.mlp.APL_mode == 1:
            	apl = np.sum(tree.decision_path(validation_data[0])) / float(validation_data[0].shape[0])
            log = open('./tree/log.txt', 'a')
            log.write('tree'+ str(i) + ': accuracy {:.5f}; number of nodes '.format(acc) + str(nodes) + '; APL {:.2f}\n'.format(apl))
            log.close()

    def predict(self, X):
    	return self.mlp.predict(X)


class MLP(object):
    def __init__(self, in_count, hidden_sizes, out_count, APL_mode=0):
        input1 = Input((in_count, ), dtype='float32')
        X = input1
        for i in hidden_sizes:
            X = Dense(i, activation='sigmoid')(X)
        X = Dense(out_count, activation='softmax')(X)
        self.model = Model(inputs=input1, outputs=X)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        self.num_weights = len([i for l in self.model.get_weights() for i in l.flatten()])
        self.in_count = in_count
        self.out_count = out_count
        self.saved_weights = []
        self.saved_apl = []
        self.last_loss = 0.0
        self.APL_mode = APL_mode

    def fit_tree(self, X_train, y_train):
        """Train decision tree to track path length."""
        y_train_hat = self.model.predict(X_train)
        # y_train_hat_int = np.rint(y_train_hat).astype(int)
        y_pred = [np.argmax(x) for x in y_train_hat]
        self.tree = DecisionTreeClassifier(min_samples_leaf=500)
        self.tree.fit(X_train, y_pred)
        return self.tree

    def average_path_length(self, X_train, y_train):
        tree = self.fit_tree(X_train, y_train)
        path_length = 0.0
        #Compute average path length
        if self.APL_mode == 0:
	        X = X_train
	        leaf_indices = tree.apply(X)
	        leaf_counts = np.bincount(leaf_indices)
	        leaf_i = np.arange(tree.tree_.node_count)
	        path_length = np.dot(leaf_i, leaf_counts) / float(X.shape[0])

        else:
          	path_length = np.sum(tree.decision_path(X_train)) / float(X_train.shape[0])

        return path_length

    def train(self, X_train, y_train, batch_size=128, epochs=2, validation_data=None):
        num = min(len(X_train) // batch_size, 30)
        class AplHistory(keras.callbacks.Callback):
            def __init__(self, mlp):
                self.mlp = mlp
                self.step = 0
                self.log = 0
                self.epochs = 0
            def on_batch_end(self, batch, logs={}):
                if self.epochs > epochs - 3:
                    apl = self.mlp.average_path_length(X_train, y_train)
                    self.mlp.saved_weights.append([i for l in self.mlp.model.get_weights() for i in l.flatten()])
                    self.mlp.saved_apl.append(apl)
                    self.mlp.last_loss = apl
                    self.log += 1
                    print(' APL:{:.2f}'.format(apl))
                self.step += 1
            def on_epoch_end(self, batch, logs={}):
                self.epochs += 1
        
        self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[AplHistory(self)], validation_data=validation_data)

    def predict(self, X):
        return self.model.predict(X)

# Surrogate function
class surrogate(object):
    def __init__(self, in_count, out_count):
        input1 = Input((in_count, ), dtype='float32')
        X = input1
        X = Dense(100, activation='tanh', kernel_regularizer=regularizers.l2(0.0001))(X)
        X = Dropout(0.5)(X)
        X = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(0.0001))(X)
        X = Dense(50, activation='tanh', kernel_regularizer=regularizers.l2(0.0001))(X)
        output1 = Dense(out_count, activation='softplus')(X)
        self.model = Model(inputs=input1, outputs=output1)
        self.model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
        self.in_count = in_count
        self.out_count = out_count

    def train(self, X_train, y_train, batch_size=1, epochs=10):
        self.model.fit(X_train, y_train, batch_size = batch_size, epochs = epochs, shuffle=True, verbose=1)

    def predict(self, X):
        return self.model.predict(X)

""" 
===============================================
l1 regularization network class
===============================================
"""

class L1MLP(object):
    def __init__(self, in_count, out_count, hidden_sizes, strength=0.01, APL_mode=0):
        self.mlp = MLP1(in_count, hidden_sizes, out_count, APL_mode = APL_mode)
        self.strength = strength
        self.tree = []

    def loss(self, y_true,y_pred):    
        return binary_crossentropy(y_true,y_pred)

    def train(self, X_train, y_train, iters_retrain=5, epochs_mlp=3, epochs_sur=5, batch_size=256, feature_names=None, validation_data=None, class_names=None):
        for i in range(iters_retrain):
            self.mlp.model.compile(optimizer='adam', loss=self.loss, metrics=['accuracy'])
            print('training MLP net... [%d/%d]' % (i + 1, iters_retrain))
            self.mlp.train(X_train, y_train, epochs=epochs_mlp, batch_size=batch_size, validation_data=validation_data)
            self.tree.append(self.mlp.fit_tree(X_train, y_train))
            if not os.path.isdir('./tree'):
                os.mkdir('./tree')
            nodes = visualize(self.tree[-1], './tree/tree' + str(i) + '.pdf',False,feature_names, class_names)
            acc = accuracy_score(self.tree[-1].predict(validation_data[0]), [np.argmax(x) for x in validation_data[1]])
            
            leaf_indices = self.tree[-1].apply(validation_data[0])
            leaf_counts = np.bincount(leaf_indices)
            leaf_i = np.arange(self.tree[-1].tree_.node_count)
            apl = np.dot(leaf_i, leaf_counts) / float(validation_data[0].shape[0])
            if self.mlp.APL_mode == 1:
            	apl = np.sum(tree.decision_path(validation_data[0])) / float(validation_data[0].shape[0])
            log = open('./tree/log.txt', 'a')
            log.write('tree'+ str(i) + ': accuracy {:.5f}; number of nodes '.format(acc) + str(nodes) + '; APL {:.2f}\n'.format(apl))
            log.close()

    def predict(self, X):
    	return self.mlp.predict(X)

class MLP1(object):
    def __init__(self, in_count, hidden_sizes, out_count, APL_mode=0):
        input1 = Input((in_count, ), dtype='float32')
        X = input1
        for i in hidden_sizes:
            X = Dense(i, activation='sigmoid', kernel_regularizer=regularizers.l1(0.00001))(X)
        X = Dense(out_count, activation='softmax')(X)
        self.model = Model(inputs=input1, outputs=X)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        self.num_weights = len([i for l in self.model.get_weights() for i in l.flatten()])
        self.in_count = in_count
        self.out_count = out_count
        self.APL_mode = APL_mode

    def fit_tree(self, X_train, y_train):
        """Train decision tree to track path length."""
        y_train_hat = self.model.predict(X_train)
        # y_train_hat_int = np.rint(y_train_hat).astype(int)
        y_pred = [np.argmax(x) for x in y_train_hat]
        self.tree = DecisionTreeClassifier(min_samples_leaf=500)
        self.tree.fit(X_train, y_pred)
        return self.tree

    def average_path_length(self, X_train, y_train):
        tree = self.fit_tree(X_train, y_train)
        path_length = 0.0
        #Compute average path length
        if self.APL_mode == 0:
	        X = X_train
	        leaf_indices = tree.apply(X)
	        leaf_counts = np.bincount(leaf_indices)
	        leaf_i = np.arange(tree.tree_.node_count)
	        path_length = np.dot(leaf_i, leaf_counts) / float(X.shape[0])

        else:
          	path_length = np.sum(tree.decision_path(X_train)) / float(X_train.shape[0])

        return path_length

    def train(self, X_train, y_train, batch_size=128, epochs=2, validation_data=None):
        self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=validation_data)

    def predict(self, X):
        return self.model.predict(X)


""" 
===============================================
l2 regularization network class
===============================================
"""

class L2MLP(object):
    def __init__(self, in_count, out_count, hidden_sizes, strength=0.01, APL_mode=0):
        self.mlp = MLP2(in_count, hidden_sizes, out_count, APL_mode = APL_mode)
        self.strength = strength
        self.tree = []

    def loss(self, y_true,y_pred):    
        return binary_crossentropy(y_true,y_pred)

    def train(self, X_train, y_train, iters_retrain=5, epochs_mlp=3, epochs_sur=5, batch_size=256, feature_names=None, validation_data=None, class_names=None):
        for i in range(iters_retrain):
            self.mlp.model.compile(optimizer='adam', loss=self.loss, metrics=['accuracy'])
            print('training MLP net... [%d/%d]' % (i + 1, iters_retrain))
            self.mlp.train(X_train, y_train, epochs=epochs_mlp, batch_size=batch_size, validation_data=validation_data)
            self.tree.append(self.mlp.fit_tree(X_train, y_train))
            if not os.path.isdir('./tree'):
                os.mkdir('./tree')
            nodes = visualize(self.tree[-1], './tree/tree' + str(i) + '.pdf',False,feature_names, class_names)
            acc = accuracy_score(self.tree[-1].predict(validation_data[0]), [np.argmax(x) for x in validation_data[1]])
            
            leaf_indices = self.tree[-1].apply(validation_data[0])
            leaf_counts = np.bincount(leaf_indices)
            leaf_i = np.arange(self.tree[-1].tree_.node_count)
            apl = np.dot(leaf_i, leaf_counts) / float(validation_data[0].shape[0])
            if self.mlp.APL_mode == 1:
            	apl = np.sum(tree.decision_path(validation_data[0])) / float(validation_data[0].shape[0])
            log = open('./tree/log.txt', 'a')
            log.write('tree'+ str(i) + ': accuracy {:.5f}; number of nodes '.format(acc) + str(nodes) + '; APL {:.2f}\n'.format(apl))
            log.close()

    def predict(self, X):
    	return self.mlp.predict(X)

class MLP2(object):
    def __init__(self, in_count, hidden_sizes, out_count, APL_mode=0):
        input1 = Input((in_count, ), dtype='float32')
        X = input1
        for i in hidden_sizes:
            X = Dense(i, activation='sigmoid', kernel_regularizer=regularizers.l2(0.0000001))(X)
        X = Dense(out_count, activation='softmax')(X)
        self.model = Model(inputs=input1, outputs=X)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        self.num_weights = len([i for l in self.model.get_weights() for i in l.flatten()])
        self.in_count = in_count
        self.out_count = out_count
        self.APL_mode = APL_mode

    def fit_tree(self, X_train, y_train):
        """Train decision tree to track path length."""
        y_train_hat = self.model.predict(X_train)
        # y_train_hat_int = np.rint(y_train_hat).astype(int)
        y_pred = [np.argmax(x) for x in y_train_hat]
        self.tree = DecisionTreeClassifier(min_samples_leaf=500)
        self.tree.fit(X_train, y_pred)
        return self.tree

    def average_path_length(self, X_train, y_train):
        tree = self.fit_tree(X_train, y_train)
        path_length = 0.0
        #Compute average path length
        if self.APL_mode == 0:
	        X = X_train
	        leaf_indices = tree.apply(X)
	        leaf_counts = np.bincount(leaf_indices)
	        leaf_i = np.arange(tree.tree_.node_count)
	        path_length = np.dot(leaf_i, leaf_counts) / float(X.shape[0])

        else:
          	path_length = np.sum(tree.decision_path(X_train)) / float(X_train.shape[0])

        return path_length

    def train(self, X_train, y_train, batch_size=128, epochs=2, validation_data=None):
        self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=validation_data)

    def predict(self, X):
        return self.model.predict(X)

""" 
===============================================
Tree visualization functions
===============================================
"""

def make_graph_minimal(graph,fs):
    nodes = graph.get_nodes()
    for node in nodes:
        old_label = node.get_label()
        label = prune_label(old_label,fs)
        if label is not None:
            node.set_label(label)
    return graph, len(nodes)


def prune_label(label,fs):
    if label is None:
        return None
    if len(label) == 0:
        return None
    label = label[1:-1]
    splitted_label = label.split('\\n')
    parts = [part for part in splitted_label
             if 'gini =' not in part]
    return '"' + '\\n'.join(parts) + '"'

def visualize(tree, save_path, fs, feature_names=None, class_names=None):
    dot_data = export_graphviz(tree, out_file=None, proportion=True,
                               filled=True, rounded=False,class_names=class_names, feature_names=feature_names)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph, nodes = make_graph_minimal(graph,fs)  # remove extra text

    if not save_path is None:
        graph.write_pdf(save_path)
    return nodes
