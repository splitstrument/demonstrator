import web
import librosa
import soundfile
import uuid
import io
import os
import yaml
import json
import pafy

from datetime import datetime

from unmix.source.api import prediction
from unmix.source.configuration import Configuration

render = web.template.render('templates/')

urls = (
    '/', 'Index',
    '/splitter', 'Splitter',
    '/result', 'Result'
)

predictions = {}

with open('conf.yaml') as file:
    config = yaml.load(file, Loader=yaml.FullLoader)


class Index:
    def GET(self):
        trainings = self.load_trainings()
        return render.index(trainings)

    def load_trainings(self):
        training_folder = config['training_folder']
        trainings = []
        for folder in os.listdir(training_folder):
            configuration_file = os.path.join(training_folder, folder, 'configuration.jsonc')
            with open(configuration_file, 'r') as configuration_file:
                configuration = json.loads(configuration_file.read())
                collection = configuration['collection']
                name = collection.get('name', folder)
            trainings.append({
                'name': name,
                'folder': folder
            })
        return trainings


class Splitter:
    def POST(self):
        training = web.input().training
        engine = prediction.create_engine(os.path.join(config['training_folder'], training))

        stereo = Configuration.get('collection.stereo', optional=False)
        fft_length = Configuration.get('spectrogram_generation.fft_length', optional=False)
        sample_rate = Configuration.get('collection.sample_rate', optional=False)

        id = str(uuid.uuid4())

        song = self.load_song(id, sample_rate, stereo)
        instrument, rest = self.run_prediction(song, engine, fft_length, sample_rate)
        predictions[id] = {
            'instrument': instrument,
            'rest': rest,
            'timestamp': datetime.now()
        }

        return render.display(id)

    def run_prediction(self, song, engine, fft_length, sample_rate):
        stft = librosa.stft(song, fft_length)
        predicted_instrument, predicted_rest = prediction.run_prediction([stft], engine)
        return self.write_predictions(predicted_instrument, predicted_rest, sample_rate)

    def load_song(self, id, sample_rate, stereo):
        file_path = self.save_song(id)
        song, _ = librosa.load(file_path, sr=sample_rate, mono=(not stereo))
        os.remove(file_path)
        return song

    def save_song(self, id):
        upload = web.input(song={})
        youtube_link = upload.youtube
        uploaded_song = upload.song
        if len(youtube_link) > 0:
            video = pafy.new(youtube_link)
            bestaudio = video.getbestaudio()
            extension = bestaudio.extension
            file_path = self.build_temp_path(extension, id)
            bestaudio.download(file_path)
            return file_path
        elif len(uploaded_song.value) > 0:
            filepath = uploaded_song.filename.replace('\\', '/')
            filename = filepath.split('/')[-1]
            extension = filename.split('.')[-1]
            file_path = self.build_temp_path(extension, id)
            fout = open(file_path, 'wb')
            fout.write(uploaded_song.file.read())
            fout.close()
            return file_path
        else:
            raise web.seeother('/')

    def build_temp_path(self, extension, id):
        return os.path.join(config['tmp_directory'], id + '.' + extension)

    def write_predictions(self, predicted_instrument, predicted_rest, sample_rate):
        instrument = io.BytesIO()
        soundfile.write(instrument, predicted_instrument[0], sample_rate, format='wav')
        rest = io.BytesIO()
        soundfile.write(rest, predicted_rest[0], sample_rate, format='wav')
        return instrument, rest


class Result:
    def GET(self):
        i = web.input()
        if i.id in predictions:
            return predictions[i.id].pop(i.type).getvalue()
        else:
            raise web.notfound()


if __name__ == "__main__":
    app = web.application(urls, globals())
    app.run()
