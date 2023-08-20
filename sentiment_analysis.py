import os
from pandas import read_csv
from transformers import pipeline
from stt_utils import WORKING_DIR, create_dir_if_not_exists, get_logger

logger = get_logger()

if __name__ == "__main__":
    logger.debug(f"current working dir is {WORKING_DIR}..")
    SENTIMENT_OUTPUT_DIR = os.path.join(WORKING_DIR, "sentiments")
    create_dir_if_not_exists(SENTIMENT_OUTPUT_DIR)
    SEGMENTATION_DIR = os.path.join(WORKING_DIR, "segmentations")
    MODEL_PATH = os.path.join(WORKING_DIR, "bin", "sentiment-analysis")

    #loading model
    logger.info("Loading sentiment analysis model..")
    sentiment_pipeline = pipeline("sentiment-analysis", model=MODEL_PATH, tokenizer=MODEL_PATH)
    logger.info("Done loading sentiment analysis model..")

    all_segmentation_files = [file for file in os.listdir(SEGMENTATION_DIR) if file.endswith(".csv")]
    if not all_segmentation_files:
        logger.info("No csv file found in segmentation folder..")
    logger.info(f"Found {len(all_segmentation_files)} csv files in segmentations folder..")
    for csv_file in all_segmentation_files:
        logger.info(f"Processing {csv_file} for sentiment analysis..")
        csv_file_path = os.path.join(SEGMENTATION_DIR, csv_file)
        df = read_csv(csv_file_path, header=0, index_col=0)
        sentiments = sentiment_pipeline(df.Text.tolist())
        sentiments_list = [row["label"] for row in sentiments]
        df["sentiment"] = sentiments_list
        output_csv_dir = os.path.join(SENTIMENT_OUTPUT_DIR, csv_file)
        df.to_csv(output_csv_dir)
        logger.info(f"Done processing {csv_file} for sentiment analysis and saved in sentiments folder..")
