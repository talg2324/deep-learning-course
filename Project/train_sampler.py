import pandas as pd
import os
import sys


def reduce_and_save_csv(input_path, output_path, sample_frac):
    # Load the CSV into a DataFrame
    df = pd.read_csv(input_path)

    # Randomly sample the DataFrame
    sampled_df = df.sample(frac=sample_frac, random_state=42)  # You can change the random_state for different results

    # Save the sampled DataFrame to a new CSV file
    sampled_df.to_csv(output_path, index=False)

    print(f"Reduced DataFrame saved to {output_path}.")


if __name__ == '__main__':
    curr_dir = os.getcwd()
    assert curr_dir.split('/')[-1] == 'Project', "this script must be ran from Project directory!"

    if len(sys.argv) != 2:
        print("Usage: python train_sampler.py <sampling frac>")
        sys.exit(1)
    sample_frac = float(sys.argv[1])
    assert 0 < sample_frac < 1, f"sampling frac must be between 0 to 1"

    reduced_train = f"data/ct-rsna/train/reduced_{int(100 * sample_frac)}pcg_train_set.csv"
    reduced_val = f"data/ct-rsna/validation/reduced_{int(100 * sample_frac)}pcg_validation_set.csv"
    assert not os.path.isfile(reduced_train), f"already performed the requested reduction. delete the files: {reduced_train, reduced_val}, and rerun."

    # sample train
    train_file_path = "data/ct-rsna/train/train_set.csv"
    reduce_and_save_csv(train_file_path, reduced_train, sample_frac)

    # sample val
    val_file_path = "data/ct-rsna/validation/validation_set.csv"
    reduced_val_file_path = reduced_val
    reduce_and_save_csv(val_file_path, reduced_val, sample_frac)
