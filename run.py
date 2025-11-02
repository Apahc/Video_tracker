# run.py
import argparse
from src.pipeline import run_pipeline

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default = "data/input/my_video_2.mp4", help="Путь к видео")
    parser.add_argument("--output", default="output/trajectory.json", help="Путь к JSON")
    parser.add_argument("--config", default="config/bodycam.yaml", help="Конфиг")
    args = parser.parse_args()

    # УДАЛИ vocab_path — он не нужен!
    run_pipeline(
        video_path=args.input,
        config_path=args.config,
        output_path=args.output
    )

if __name__ == "__main__":
    main()