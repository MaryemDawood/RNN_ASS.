#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np


def prepare_data():
    text = ["I", "love", "deep", "learning"]
    word_to_idx = {w: i for i, w in enumerate(text)}
    idx_to_word = {i: w for i, w in enumerate(text)}
    vocab_size = len(text)
    
    X = np.array([word_to_idx["I"], word_to_idx["love"], word_to_idx["deep"]])
    y = word_to_idx["learning"]
    
    return X, y, word_to_idx, idx_to_word, vocab_size

def one_hot_encode(index, vocab_size):
    vec = np.zeros(vocab_size)
    vec[index] = 1
    return vec

def prepare_inputs(X_indices, vocab_size):
    X_encoded = np.array([one_hot_encode(i, vocab_size) for i in X_indices])
    return X_encoded


def initialize_parameters(vocab_size, hidden_size):
    np.random.seed(42)
    parameters = {
        "Wxh": np.random.randn(hidden_size, vocab_size) * 0.01,
        "Whh": np.random.randn(hidden_size, hidden_size) * 0.01,
        "Why": np.random.randn(vocab_size, hidden_size) * 0.01,
        "bh": np.zeros((hidden_size, 1)),
        "by": np.zeros((vocab_size, 1))
    }
    return parameters


def rnn_forward(X_encoded, parameters):
    Wxh, Whh, Why = parameters["Wxh"], parameters["Whh"], parameters["Why"]
    bh, by = parameters["bh"], parameters["by"]
    
    h = np.zeros((Whh.shape[0], 1))  
    
    for x in X_encoded:
        x = x.reshape(-1, 1) 
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
        
    y_pred = np.dot(Why, h) + by  
    return y_pred, h


def compute_loss_and_gradients(y_pred, y_true, h, X_encoded, parameters):
    Wxh, Whh, Why = parameters["Wxh"], parameters["Whh"], parameters["Why"]
    
    exp_scores = np.exp(y_pred)
    probs = exp_scores / np.sum(exp_scores)
    loss = -np.log(probs[y_true, 0])
    
    d_y = probs
    d_y[y_true] -= 1
    
    dWhy = np.dot(d_y, h.T)
    dby = d_y
    
   
    dh = np.dot(Why.T, d_y) * (1 - h * h)  
    
    
    dWxh = np.dot(dh, X_encoded[-1].reshape(1, -1))
    dWhh = np.dot(dh, h.T)
    dbh = dh
    
    grads = {"dWxh": dWxh, "dWhh": dWhh, "dWhy": dWhy, "dbh": dbh, "dby": dby}
    
    return loss, grads



def update_parameters(parameters, grads, learning_rate):
    for param in parameters:
        parameters[param] -= learning_rate * grads["d" + param]
    return parameters



def train_rnn(X_encoded, y_true, vocab_size, hidden_size=5, epochs=500, learning_rate=0.1):
    parameters = initialize_parameters(vocab_size, hidden_size)
    
    for epoch in range(epochs):
        y_pred, h = rnn_forward(X_encoded, parameters)
        loss, grads = compute_loss_and_gradients(y_pred, y_true, h, X_encoded, parameters)
        
        parameters = update_parameters(parameters, grads, learning_rate)
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss:.4f}")
            
    return parameters



def predict(X_encoded, parameters, idx_to_word):
    y_pred, _ = rnn_forward(X_encoded, parameters)
    predicted_idx = np.argmax(y_pred)
    predicted_word = idx_to_word[predicted_idx]
    return predicted_word



def main():
    X_indices, y_true, word_to_idx, idx_to_word, vocab_size = prepare_data()
    X_encoded = prepare_inputs(X_indices, vocab_size)
    
    parameters = train_rnn(X_encoded, y_true, vocab_size)
    
    prediction = predict(X_encoded, parameters, idx_to_word)
    print(f"\nPredicted fourth word: {prediction}")

if __name__ == "__main__":
    main()


# In[ ]:




