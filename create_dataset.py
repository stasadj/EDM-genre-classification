import json
from glob import glob


def num_of_songs_for_genre():
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


if __name__ == '__main__':
    print('EDM Genre Classification - Spectrogram Generator')

    num_of_songs_for_genre()
