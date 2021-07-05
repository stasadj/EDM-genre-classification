import numpy as np

from glob import glob
from keras.models import load_model
from sklearn.utils import shuffle
from PIL import Image


genres = ['bigroom', 'dnb', 'house', 'trance']


def load_data():
    X = []
    y = []

    for i in range(len(genres)):
        genre = genres[i]
        j = 0
        for file_name in glob('data\\spectrogram-mono-test\\{}\\*.png'.format(genre)):

            if j == 100:
                break
            j += 1
            image = np.array(Image.open(file_name))
            image = np.resize(image, (128, 128, 1))

            print(file_name)
            print(image.shape)
            X.append(image)
            y.append(i)

    X = np.array(X)
    y = np.array(y)
    return X, y


if __name__ == '__main__':
    print('EDM Genre Classification - Test')

    X, y = load_data()
    X, y = shuffle(X, y, random_state=42)
    model = load_model('model_5.h5')

    test_loss, test_acc = model.evaluate(X, y, verbose=2)
    print('\nTest loss:', test_loss)
    print('\nTest accuracy:', test_acc)

    print('EDM Genre Classification - Predictions')
    print('--------------------------------------')

    samples = [X[5], X[123], X[343], X[0], X[200]]
    p = model.predict(np.array(samples))

    print('TEST 1')
    print('Expected: {}, Result: {}'.format(genres[y[5]], genres[np.argmax(p, axis=1)[0]]))
    print('--------------------------------------')

    print('TEST 2')
    print('Expected: {}, Result: {}'.format(genres[y[123]], genres[np.argmax(p, axis=1)[1]]))
    print('--------------------------------------')

    print('TEST 3')
    print('Expected: {}, Result: {}'.format(genres[y[343]], genres[np.argmax(p, axis=1)[2]]))
    print('--------------------------------------')

    print('TEST 4')
    print('Expected: {}, Result: {}'.format(genres[y[0]], genres[np.argmax(p, axis=1)[3]]))
    print('--------------------------------------')

    print('TEST 5')
    print('Expected: {}, Result: {}'.format(genres[y[200]], genres[np.argmax(p, axis=1)[4]]))
    print('--------------------------------------')
