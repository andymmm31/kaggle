import pandas as pd

# Define paths
TRAIN_CSV_PATH = '/kaggle/input/csiro-biomass/train.csv'

# Load the long-format data
print(f"Loading data from {TRAIN_CSV_PATH}...")
df_train_long = pd.read_csv(TRAIN_CSV_PATH)

# Pivot to wide format to get targets in columns
print("Pivoting data to wide format...")
df_train_wide = df_train_long.pivot(
    index='image_path',
    columns='target_name',
    values='target'
).reset_index()

# Drop rows with any missing target values, as the training script does
df_train_wide = df_train_wide.dropna()

print("Calculating the sum of component parts...")
# Define the component columns
component_columns = ['Dry_Green_g', 'Dry_Dead_g', 'Dry_Clover_g']
df_train_wide['sum_of_parts'] = df_train_wide[component_columns].sum(axis=1)

# Compare the calculated sum with the 'Dry_Total_g' column
print("Comparing sum_of_parts with Dry_Total_g...")
# Using np.isclose for robust floating-point comparison
are_equal = pd.Series(
    [
        abs(a - b) < 1e-9  # Using a small tolerance for float comparison
        for a, b in zip(df_train_wide['sum_of_parts'], df_train_wide['Dry_Total_g'])
    ]
)

if are_equal.all():
    print("\n✅ HYPOTHESIS CONFIRMED!")
    print("For all rows, Dry_Total_g is the sum of Dry_Green_g, Dry_Dead_g, and Dry_Clover_g.")
else:
    print("\n❌ HYPOTHESIS REJECTED.")
    num_mismatches = len(are_equal) - are_equal.sum()
    print(f"Found {num_mismatches} mismatches out of {len(df_train_wide)} rows.")
    # Show some examples of mismatches
    mismatch_df = df_train_wide[~are_equal]
    print("\nExamples of mismatches:")
    print(mismatch_df[['sum_of_parts', 'Dry_Total_g'] + component_columns].head())

# Clean up the dataframe to show a few examples
print("\n--- Example Data ---")
print(df_train_wide[['sum_of_parts', 'Dry_Total_g'] + component_columns].head())
