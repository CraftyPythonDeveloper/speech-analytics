import os
from pandas import read_excel, DataFrame
import re
import sys
import json

input_text = "Press any key to continue"


def create_dir_if_not_exists(path, not_exist_msg=None, exit=False):
    if not os.path.exists(path):
        if not_exist_msg:
            print(not_exist_msg)
        os.mkdir(path)
        if exit:
            input(input_text)
            sys.exit()
        return True
    return False


def generate_scorecard(quality_df, transcript, filename):
    scorecard = []
    scored_examples = []
    transcript = transcript.lower()
    transcript = re.sub('[^A-Za-z0-9 ]+', '', transcript)
    for row in quality_df.itertuples(index=False):
        temp_dict = {"id": row.id, "parameter": row.parameter, "sub_parameter": row.sub_parameter, "score": 0}
        for i in row.examples.split("|"):
            if (i.strip().lower() in transcript) and (i.lower() not in scored_examples):
                temp_dict["score"] = row.score
                break
            scored_examples.append(i)
        scorecard.append(temp_dict)

    scorecard_df = DataFrame(scorecard)
    scorecard_df.loc[len(scorecard_df.index)] = ["", "Total Score", "", scorecard_df.score.sum()]
    try:
        scorecard_df.to_csv(filename, index=False)
    except:
        print("error while saving scorecard... Please make sure to close the existing csv file..")
    return scorecard_df


if __name__ == "__main__":
    WORKING_DIR = os.path.join(os.getcwd(), "working_dir")
    # WORKING_DIR = os.path.join(os.path.dirname(os.getcwd()), "working_dir")
    cwd = os.path.join(os.path.dirname(os.getcwd()), "working_dir")
    transcription_dir = os.path.join(cwd, "transcriptions")
    quality_df_path = os.path.join(cwd, "quality_parameters.xlsx")
    scorecard_path = os.path.join(cwd, "scorecards")

    cwd_err_msg = "working_dir folder does not exists. Creating.."
    quality_err_msg = "quality_parameters.xlsx file does not exists in current directory.. Please add it first.."
    trans_err_msg = "transcriptions folder does not exists, Creating...\nPlease add your txt files in transcriptions " \
                    "folder"
    create_dir_if_not_exists(cwd, cwd_err_msg, True)
    create_dir_if_not_exists(quality_df_path, quality_err_msg, True)
    create_dir_if_not_exists(transcription_dir, trans_err_msg, True)
    create_dir_if_not_exists(scorecard_path)

    quality_df = read_excel(quality_df_path)
    transcriptions = [file for file in os.listdir(transcription_dir) if file.endswith(".json")]
    if not transcriptions:
        print("Transcriptions not found, Please add call transcriptions in .txt format")
        input(input_text)
        sys.exit()
    print(f"found {len(transcriptions)} files to process..")
    for file in transcriptions:
        file_path = os.path.join(transcription_dir, file)
        filename = file.rsplit(os.extsep, 1)[0]
        with open(file_path, "r") as fp:
            data = json.load(fp)
            text = data["text"]
        generate_scorecard(quality_df, text, os.path.join(scorecard_path, filename+".csv"))
        print(f"Updated quality scores for {filename} and saved into scorecard folder...")

    print("Done processing all the files..")
    input(input_text)
    sys.exit()
