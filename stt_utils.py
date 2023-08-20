import os
import subprocess
import sys
import contextlib
import wave
from pandas import read_csv
import logging

WORKING_DIR = os.path.join(os.getcwd(), "working_dir")
# WORKING_DIR = os.path.join(os.path.dirname(os.getcwd()), "working_dir")
CONFIG = read_csv(os.path.join(WORKING_DIR, "config.csv"))


def get_script_folder():
    script_path = os.path.dirname(os.path.abspath(sys.modules['__main__'].__file__))
    return script_path


def get_data_folder():
    data_folder_path = os.path.dirname(os.path.abspath(sys.modules['__main__'].__file__))
    return data_folder_path


def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)
        return True
    return False


def convert_mp3_to_wav(filepath, output_dir):
    filename = os.path.split(filepath)[1]
    output_file = os.path.join(output_dir, f'{filename.rsplit(".", 1)[0]}.wav')
    cmd = f"ffmpeg -i {filepath} -acodec pcm_s16le -ac 1 -ar 16000 {output_file}"
    try:
        subprocess.check_output(cmd, stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return True
    except subprocess.CalledProcessError:
        return False


def get_audio_duration(audio_file):
    with contextlib.closing(wave.open(audio_file, 'r')) as f:
        frames = f.getnframes()
        rate = f.getframerate()
        return frames / float(rate)


def remove_file_ext(filename):
    if isinstance(filename, str):
        return filename.rsplit(os.extsep, 1)[0]
    return [file.rsplit(os.extsep, 1)[0] for file in filename]


def get_config(name=None, all_column_values=None):
    if name:
        return CONFIG[CONFIG["name"] == name]["value"].values[0]
    elif all_column_values:
        return CONFIG[all_column_values].tolist()
    else:
        return None


def get_logger():
    log = logging.INFO
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    try:
        if sys.argv[1] == "--debug":
            log = logging.DEBUG
            logger.addHandler(logging.FileHandler("debug.log"))
    except IndexError:
        pass

    logger.setLevel(log)
    return logger
