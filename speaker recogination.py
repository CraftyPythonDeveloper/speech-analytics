import os
import shutil
import datetime
import time
import numpy as np
import json
from pandas import DataFrame

from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import torch
from pyannote.audio import Audio
from pyannote.core import Segment
import traceback
import sys
from stt_utils import create_dir_if_not_exists, get_audio_duration, get_config, remove_file_ext, WORKING_DIR, get_logger

logger = get_logger()
no_traceback = True
if no_traceback:
    def custom_excepthook(exc_type, exc_value, exc_traceback):
        print("An error occurred: ", exc_value)
    sys.excepthook = custom_excepthook


def segment_embedding(segment, audio_file, duration):
    audio = Audio()
    start = segment["start"]
    # Whisper overshoots the end timestamp in the last segment
    end = min(duration, segment["end"])
    clip = Segment(start, end)
    waveform, sample_rate = audio.crop(audio_file, clip)
    return embedding_model(waveform[None])


def get_segments(transcription_file):
    with open(transcription_file, "r") as fp:
        data = json.load(fp)
    assert "segments" in data.keys(), f"segments not found in {transcription_file}"
    return data["segments"]


def convert_time(secs):
    return datetime.timedelta(seconds=round(secs))


if __name__ == "__main__":
    working_dir = WORKING_DIR
    logger.debug(f"Current working dir is {working_dir}")
    audio_dir = os.path.join(working_dir, "wav_audio_files")
    transcriptions_dir = os.path.join(working_dir, "transcriptions")
    output_dir = os.path.join(working_dir, "segmentations")
    speechbrain_model_files_src = os.path.join(working_dir, "bin", "speechbrain")
    speechbrain_model_files_dest = os.path.join(os.environ["USERPROFILE"], ".cache", "torch", "pyannote", "speechbrain")
    create_dir_if_not_exists(output_dir)
    if not os.path.exists(speechbrain_model_files_dest) or not os.listdir(speechbrain_model_files_dest):
        logger.debug("speechbrain model checkpoints does not exists in .cache. Adding")
        os.makedirs(speechbrain_model_files_dest, exist_ok=True)
        # copy_tree(speechbrain_model_files_src, os.path.join(speechbrain_model_files_dest, "speechbrain"))
        shutil.copytree(speechbrain_model_files_src, speechbrain_model_files_dest, dirs_exist_ok=True)
        logger.info("Copied speechbrain models..")

    assert os.path.exists(transcriptions_dir), "transcription folder does not exists in working_dir"
    assert "num_speaker" in get_config(all_column_values="name"), "num_speaker name not found in config.csv file.."

    num_speaker = int(get_config("num_speaker"))
    all_audios = [file for file in os.listdir(audio_dir) if file.endswith(".wav")]
    all_transcripts = [file for file in os.listdir(transcriptions_dir) if file.endswith(".json")]
    logger.info(f"Found {len(all_transcripts)} transcripts to process..")
    logger.info("Loading speechbrain model..")
    try:
        embedding_model = PretrainedSpeakerEmbedding("speechbrain/spkrec-ecapa-voxceleb",
                                                     device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    except Exception as e:
        logger.info("Unable to load the model.. Please try again.., ", e)
        input("Press any key to exit")
        sys.exit()
    logger.info("Loaded speechbrain model..")
    for transcription_file in all_transcripts:
        logger.info(f"Processing {transcription_file} for segmentation..")
        if remove_file_ext(transcription_file) not in remove_file_ext(all_audios):
            logger.info(f"Audio file not found for {transcription_file}")
            continue
        trans_name, trans_ext = transcription_file.rsplit(os.extsep, 1)  # will split name and ext
        segments = get_segments(os.path.join(transcriptions_dir, transcription_file))
        embeddings = np.zeros(shape=(len(segments), 192))
        audio_file = os.path.join(audio_dir, f"{trans_name}.wav")
        logger.debug(f"Generating embeddings for {len(segments)}..")
        for i, segment in enumerate(segments):
            try:
                embeddings[i] = segment_embedding(segment, audio_file, get_audio_duration(audio_file))
                logger.debug(f"Created embedding for segment no {i}")
            except Exception as e:
                logger.debug(f"Exception while creating embedding for segment: {i}, {e}")
                continue
        embeddings = np.nan_to_num(embeddings)

        num_speakers = num_speaker
        logger.debug("Finding the speakers..")
        if num_speakers == 0:
            # Find the best number of speakers
            score_num_speakers = {}

            for num_speakers in range(2, 10 + 1):
                clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
                score = silhouette_score(embeddings, clustering.labels_, metric='euclidean')
                score_num_speakers[num_speakers] = score
            best_num_speaker = max(score_num_speakers, key=lambda x: score_num_speakers[x])
            logger.info(f"The best number of speakers: {best_num_speaker} with {score_num_speakers[best_num_speaker]} score")
        else:
            best_num_speaker = num_speakers

        # Assign speaker label
        clustering = AgglomerativeClustering(best_num_speaker).fit(embeddings)
        logger.debug("Fitted embeddings to model..")
        labels = clustering.labels_
        for i in range(len(segments)):
            segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

        time_start = time.time()
        # Make output
        objects = {
            'Start': [],
            'End': [],
            'Speaker': [],
            'Text': []
        }
        text = ''
        for (i, segment) in enumerate(segments):
            logger.debug(f"Segments: {objects} ")
            if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
                objects['Start'].append(str(convert_time(segment["start"])))
                objects['Speaker'].append(segment["speaker"])
                if i != 0:
                    objects['End'].append(str(convert_time(segments[i - 1]["end"])))
                    objects['Text'].append(text)
                    text = ''
            text += segment["text"] + ' '
        objects['End'].append(str(convert_time(segments[i - 1]["end"])))
        objects['Text'].append(text)
        logger.debug("Finished adding speakers..")
        time_end = time.time()
        time_diff = time_end - time_start
        logger.debug("Saving the output to a csv file..")
        save_path = f"{trans_name}.csv"
        df_results = DataFrame(objects)
        df_results.to_csv(os.path.join(output_dir, save_path))
        logger.info(f"Done processing {transcription_file} and the output with a filename of {save_path}")
    input("press any key to exit..")
