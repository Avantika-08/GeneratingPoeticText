# Tensorflow is a free and open source software library for ML and AI.
# It is used for a range of tasks but particular focus is on
# -> training and inference of deep neural networks
# -> used to execute the operations in the graph
# APPLICATIONS:
# -> Image Classification, Object Detection, Fraud Detection, NLP

#Keras is a deep learning API that can be used in python to built and train neural networks.



import random
import numpy as np
import tensorflow as tf

# Sequential is for creating neural networks
from tensorflow.keras.models import Sequential

# Optimizer for compilation of model
from tensorflow.keras.optimizers import RMSprop

# LSTM(Long Short Term Memory) a recurrent layer with memory
# Dense Layer for hidden layers
# Activation Layer for output layer
from tensorflow.keras.layers import Activation, Dense,LSTM


# Loading the data
filepath=tf.keras.utils.get_file('shakespeare.txt','https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
text=open(filepath,'rb').read().decode(encoding='utf-8').lower()


# Preparing the data
text=text[500000:800000]
characters=sorted(set(text))
#print(characters)

# Enumerate assigns one number to each character in the set
char_to_index=dict((c,i)for i, c in enumerate(characters))
print(char_to_index)
index_to_char=dict((i,c)for i, c in enumerate(characters))
print(index_to_char)


# Training the data 
SEQ_LENGTH=40
STEP_SIZE=3

# Predict the next character
sentences=[]
next_char=[]
# print(text)
# print('\n\n')
for i in range(0,len(text)-SEQ_LENGTH,STEP_SIZE):
    sentences.append(text[i:i+SEQ_LENGTH])
    next_char.append(text[i+SEQ_LENGTH])
# print(sentences)
# print(next_char)

# Converting training data into a numerical format
x=np.zeros((len(sentences),SEQ_LENGTH,len(characters)),dtype=bool)
y=np.zeros((len(sentences),len(characters)),dtype=bool)

for i,sentence in enumerate(sentences):
    for t,character in enumerate(sentence):
        x[i,t,char_to_index[character]]=1
    y[i,char_to_index[next_char[i]]]=1
   
# model=Sequential()
# model.add(LSTM(128,input_shape=(SEQ_LENGTH,len(characters))))
# model.add(Dense(len(characters)))
# model.add(Activation('softmax'))

# model.compile(loss='categorical_crossentropy',optimizer=RMSprop(lr=0.01))    # lr->learning rate
# model.fit(x,y,batch_size=256,epochs=4)

# model.save('textgenerator.model')


model=tf.keras.models.load_model('textgenerator.model')


# Making Predictions
def sample(preds,temperature=1.0):
    preds=np.asarray(preds).astype('float64')
    preds=np.log(preds)/temperature
    exp_pred=np.exp(preds)
    preds=exp_pred/np.sum(exp_pred)
    probas=np.random.multinomial(1,preds,1)
    return np.argmax(probas)


def generate_text(length,temperature):
    start_index=random.randint(0,len(text)-SEQ_LENGTH-1)
    generated=''
    sentence=text[start_index:start_index+SEQ_LENGTH]
    generated+=sentence
    for i in range(length):
        x=np.zeros((1,SEQ_LENGTH,len(characters)))
        for t,character in enumerate(sentence):
            x[0,t,char_to_index[character]]=1
            
        predicitons=model.predict(x,verbose=0)[0]
        next_index=sample(predicitons,temperature)
        next_character=index_to_char[next_index]
        
        generated+=next_character
        sentence=sentence[1:]+next_character
    return generated


print('-----------0.2----------')
print(generate_text(300,0.2))
print('-----------0.4----------')
print(generate_text(300,0.4))
print('-----------0.6----------')
print(generate_text(300,0.6))
print('-----------0.8----------')
print(generate_text(300,0.8))
print('-----------1----------')
print(generate_text(300,1))






