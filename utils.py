import librosa
import soundfile as sf
import csv
import os
import shutil
import time


def resample(sample_rate=None, dir=None, csv_path=None):

    clips = []
    start_time = time.time()

    # List all clips that appear on the csv (train, eval or test)

    if csv_path != 'test':
        with open(csv_path, 'r') as csvFile:
            reader = csv.reader(csvFile)
            for row in reader:
                clips.append(row[0])

        csvFile.close()
        clips.remove('fname')
    else:
        clips = os.listdir(dir)

    if os.path.exists(dir+'/resampled/'):
        shutil.rmtree(dir+'/resampled', ignore_errors=True)  # ignore errors whit read only files

    os.mkdir(dir+'/resampled')

    for clip in clips:
        # Audio clip is read
        data, sr = sf.read(dir+'/'+clip)
        data = data.T
        # Audio data is resampled to desired sample_rate
        if sr != sample_rate:
            data_resampled = librosa.resample(data, sr, sample_rate)
        # Processed data is saved into a directory under train_clip_dir
        sf.write(dir+'/resampled/'+clip, data_resampled, sample_rate, subtype='PCM_16')

    print('Audio data has been resampled successfully')
    elapsed_time = time.time() - start_time
    print('Elapsed time ' + str(elapsed_time) + ' seconds')
