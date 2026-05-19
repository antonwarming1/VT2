"""
run_pipeline.py — Run the full preprocessing pipeline in order:

  1. data_cleaning.py        — Fix NaN, negative values, unit encoding
  2. data_preprocessing.py   — Remove idle phase, resample, smooth
  3. exclude_features.py     — Drop unwanted columns

Each step reads from the previous step's output folder.
Configure which subfolders and features to process in each script's Config section.
"""

from Preprocessing.data_cleaning import main as clean
from Preprocessing.data_preprocessing import main as preprocess
from Preprocessing.exclude_features import main as exclude_features
from Feature_engineering.code import main as feature_engineering

def main():
    print("=" * 60)
    print("  Step 1/3: Cleaning")
    print("=" * 60)
    clean()

    print("\n")
    print("=" * 60)
    print("  Step 2/3: Preprocessing (idle removal + resampling)")
    print("=" * 60)
    preprocess()

    print("\n")
    print("=" * 60)
    print("  Step 3/3: Exclude features")
    print("=" * 60)
    exclude_features()

    print("\n")
    print("=" * 60)
    print("  Step 4/4: Feature engineering")
    print("=" * 60)
    feature_engineering()

    print("\n" + "=" * 60)
    print("  Pipeline complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
