import json
import pandas as pd
import matplotlib.pyplot as plt

from glob import glob
from tempfile import mktemp
from scipy.io import wavfile
from pydub import AudioSegment


def original_metadata():
    genre_song_count = {}

    for f_name in glob('data\\meta\\*.json'):

        try:
            f = open(f_name)
            data = json.load(f)
            genre = data['genres'][0]['slug']
            if genre not in genre_song_count:
                genre_song_count[genre] = 1
            else:
                genre_song_count[genre] += 1
            f.close()
        except:
            continue

    print(genre_song_count)
    print('Total: {}'.format(sum(genre_song_count.values())))


def num_of_songs_for_genre():
    genre_song_count = {}

    data = pd.read_csv('data/Beatport-EDM-Key-Dataset.csv')
    data_frame = pd.DataFrame(data)
    for index, row in data_frame.iterrows():
        genre = row['genres']
        if genre not in genre_song_count:
            genre_song_count[genre] = 1
        else:
            genre_song_count[genre] += 1

    print(genre_song_count)
    print('Total genres: {}'.format(len(genre_song_count)))
    print('Total songs: {}'.format(sum(genre_song_count.values())))


def generate_spectrograms():
    data = pd.read_csv('data/Beatport-EDM-Key-Dataset.csv')
    data_frame = pd.DataFrame(data)
    num = 0

    for index, row in data_frame.iterrows():
        genre = row['genres']
        id = row['id']

        audio_file_name = glob('data\\audio\\{}*.mp3'.format(id))
        num += 1
        print('{}. {}'.format(num, id))

        if genre == 'Electronica / Downtempo':
            save_png(audio_file_name[0], 'electronica-downtempo', id)
            continue

        if genre == 'Drum & Bass':
            save_png(audio_file_name[0], 'drum-and-bass', id)
            continue

        if genre in ['Techno', 'Minimal']:
            save_png(audio_file_name[0], 'techno', id)
            continue

        if genre in ['House', 'Tech House', 'Deep House']:
            save_png(audio_file_name[0], 'house', id)
            continue

        if genre in ['Trance', 'Psy-Trance']:
            save_png(audio_file_name[0], 'trance', id)
            continue


def save_png(file_name, genre, id):
    mp3_audio = AudioSegment.from_file(file_name, format="mp3")  # read mp3
    wname = mktemp('.wav')  # use temporary file
    mp3_audio.export(wname, format="wav")  # convert to wav
    FS, data = wavfile.read(wname)  # read wav file

    fig, ax = plt.subplots(1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax.axis('off')
    pxx, freqs, bins, im = ax.specgram(x=data[:, 0], Fs=FS, noverlap=384, NFFT=512)
    ax.axis('off')
    fig.savefig('data\\spectrogram\\{0}\\{1}.png'.format(genre, id), dpi=100, frameon='false')


if __name__ == '__main__':
    print('EDM Genre Classification - Spectrogram Generator')

    generate_spectrograms()

