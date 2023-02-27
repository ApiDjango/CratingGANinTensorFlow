from tensorflow.keras.preprocessing.text import Tokenizer
import pickle

# создание экземпляра Tokenizer
tokenizer = Tokenizer()

# обучение Tokenizer на текстовом корпусе
texts = ["some text here", "another text sample", "yet another text"]
tokenizer.fit_on_texts(texts)

# сохранение объекта Tokenizer в файле tokenizer.pickle
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
