import os
import sys
import json
import whisper
# import whisperx
from stt_utils import create_dir_if_not_exists, convert_mp3_to_wav, remove_file_ext, get_config, WORKING_DIR, get_logger
import traceback

no_traceback = True
if no_traceback:
    def custom_excepthook(exc_type, exc_value, exc_traceback):
        logger.info("An error occurred: ", exc_value)
    sys.excepthook = custom_excepthook

logger = get_logger()

if __name__ == "__main__":
    input_statement = "Press any key to continue.."
    # temp_dir = get_data_folder()
    cwd_wrk_dir = WORKING_DIR

    # ffmpeg path
    bin_path = os.path.join(cwd_wrk_dir, "bin")
    os.environ["PATH"] = f"%PATH%;{bin_path}"

    assert os.path.exists(cwd_wrk_dir), "working_dir not found.."

    mp3_audio_files = os.path.join(cwd_wrk_dir, "audio_files")
    audio_folder_path = os.path.join(cwd_wrk_dir, "wav_audio_files")
    text_folder = os.path.join(cwd_wrk_dir, "transcriptions")
    create_dir_if_not_exists(mp3_audio_files)
    create_dir_if_not_exists(audio_folder_path)
    create_dir_if_not_exists(text_folder)

    list_of_audio_files = [audio for audio in os.listdir(audio_folder_path) if audio.endswith(".wav")]
    list_of_mp3_audio_files = [audio for audio in os.listdir(mp3_audio_files) if audio.endswith(".mp3")]
    if not list_of_mp3_audio_files:
        logger.info("No mp3 audio files found in audio_files folder. Please make sure you add audio file with extension .mp3 "
              "in audio_files folder and try again..")
        input(input_statement)
        sys.exit()
    mp3_needs_to_be_converted = [file for file in (set(remove_file_ext(list_of_mp3_audio_files)) -
                                                   set(remove_file_ext(list_of_audio_files)))
                                 if f"{file}.mp3" in list_of_mp3_audio_files]
    logger.info(f"Found {len(mp3_needs_to_be_converted)} mp3 files which needs to be converted to wav..")
    logger.info("Started audio conversion process..")
    for file in mp3_needs_to_be_converted:
        status = convert_mp3_to_wav(os.path.join(mp3_audio_files, f"{file}.mp3"), audio_folder_path)
        if not status:
            logger.info(f"Unable to convert {file} to wav. Skipping this file..")
            continue
        logger.info(f"converted {file} to wav")
    logger.info("Completed audio conversion process..")
    list_of_audio_files = [audio for audio in os.listdir(audio_folder_path) if audio.endswith(".wav")]
    logger.info(f"Found {len(list_of_audio_files)} wav audio files to process")
    try:
        model = whisper.load_model(get_config("model"), download_root=bin_path)
        # model = whisperx.load_model(get_config("model"), device=get_config("device"),
        #                             compute_type=get_config("compute_type"))
    except:
        logger.info("Error while loading language model..")
        input(input_statement)
        sys.exit()
    for n, audio_name in enumerate(list_of_audio_files, start=1):
        try:
            audio_path = os.path.join(cwd_wrk_dir, "wav_audio_files", audio_name)
            audio_file_name = audio_name.rsplit(os.extsep, 1)[0]
            logger.info(f"Processing {audio_file_name}.. File no {n} of {len(list_of_audio_files)}")
            # result = model.transcribe(audio_path, batch_size=int(get_config("batch_size")))
            result = model.transcribe(audio_path, fp16=False, language=get_config("language", None), verbose=False)
            text = result.get("text", "").strip()
            # realignment of whisper output
            # logger.info("Realigning whisper output")
            # model_a, metadata = whisperx.load_align_model(language_code=result["language"],
            #                                               device=get_config("device"))
            # result_align = whisperx.align(result["segments"], model_a, metadata, audio_path, get_config("device"),
            #                               return_char_alignments=False)
            # result["segments"] = result_align["segments"]
            segments = []
            i = 0
            for segment_chunk in result["segments"]:
                chunk = {"start": segment_chunk["start"], "end": segment_chunk["end"], "text": segment_chunk["text"]}
                segments.append(chunk)
                i += 1
            result["segments"] = segments
            text_output_path = os.path.join(text_folder, audio_file_name + ".json")
            with open(text_output_path, "w") as fp:
                json.dump(result, fp)
            logger.info(f"Processed audio file and transcribed the audio files..")
        except Exception as e:
            logger.info(e)

    logger.info("Done processing all the audio files...")
    input(input_statement)
    sys.exit()
