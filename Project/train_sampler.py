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
    assert curr_dir.split()[-1] == 'latent-diffusion', "this script must be ran from latent-diffusion directory!"

    if len(sys.argv) != 2:
        print("Usage: python train_sampler.py <sampling frac>")
        sys.exit(1)
    frac = sys.argv[1]

    reduced_path = f"../data/ct-rsna/reduced_{int(100 * frac)}_pcg"
    assert (not os.path.exists(reduced_path),
            f"already performed the requested reduction. "
            f"delete the directory: {reduced_path}, and rerun.")

    os.mkdir(reduced_path)

    # sample train
    train_file_path = "../data/ct-rsna/train/train_set.csv"
    reduced_train_file_path = os.path.join(reduced_path, "train_set.csv")
    sample_frac = frac  # Percentage of data to sample (adjust as needed)

    reduce_and_save_csv(train_file_path, reduced_train_file_path, sample_frac)

    # sample val
    val_file_path = "../data/ct-rsna/validation/validation_set.csv"
    reduced_val_file_path = os.path.join(reduced_path,"validation_set.csv")
    reduce_and_save_csv(val_file_path, reduced_val_file_path, sample_frac)
