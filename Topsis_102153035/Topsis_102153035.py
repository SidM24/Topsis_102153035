import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def helper(df):
    # Selecting the non-numeric columns
    non_numeric_col = df.iloc[:, 1:].select_dtypes(exclude=['number']).columns

    # Using label encoding to convert all non numeric columns to numeric values
    label_encoder = LabelEncoder()
    for col in non_numeric_col:
        df[col] = label_encoder.fit_transform(df[col])

    return df


def topsis(input_csv, weights, impacts, result_csv):
    try:
        # Read the input CSV file into a pandas DataFrame
        df = pd.read_csv(input_csv)

        # Checking if the input CSV file has 3 or more columns or not
        if len(df.columns) < 3:
            raise ValueError("Input CSV must contain at least 3 columns")

        # Converting any non-numeric columns to numeric values using encoding
        df = helper(df)

        # Convert weights and impacts to lists
        weights = [float(w) for w in weights.split(',')]
        impacts = [1 if i == '+' else -1 if i == '-' else None for i in impacts.split(',')]

        # Checking if the entered weights are numeric
        if not all(isinstance(w, (int, float)) for w in weights):
            raise ValueError("Weights must be numeric values separated by ','.")

        # Checking if the impacts are entered in correct format
        if not all(impact in [1, -1] for impact in impacts):
            raise ValueError("Impacts must be entered as '+' or '-'.")

        # Checking if the length of weights and impact match the number of columns in the dataframe
        if len(weights) != len(df.columns) - 1 or len(impacts) != len(df.columns) - 1:
            raise ValueError("Number of weights and impacts must match the number of columns in the CSV file")

        # Normalize the data
        normalized_df = df.iloc[:, 1:] / np.sqrt((df.iloc[:, 1:] ** 2).sum())

        # Multiply each column by its corresponding weight
        weighted_df = normalized_df * weights

        # Calculate the positive and negative ideal solutions
        index = 0
        ideal_positive = []
        ideal_negative = []
        for column in weighted_df.columns:
            if impacts[index] == -1:
                ideal_positive.append(min(weighted_df[column]))
                ideal_negative.append(max(weighted_df[column]))
            else:
                ideal_positive.append(max(weighted_df[column]))
                ideal_negative.append(min(weighted_df[column]))

            index = index + 1

        # Calculate the separation measures
        separation_positive = []
        separation_negative = []

        for row in range(0, len(weighted_df)):
            sum1 = 0
            sum2 = 0
            for col in range(0, len(weighted_df.columns)):
                sum1 += (weighted_df.iloc[row, col] - ideal_positive[col]) ** 2
                sum2 += (weighted_df.iloc[row, col] - ideal_negative[col]) ** 2
            separation_negative.append(sum2 ** 0.5)
            separation_positive.append(sum1 ** 0.5)

        # Calculate the TOPSIS score
        topsis_scores = []
        for i in range(0, len(weighted_df)):
            sum3 = 0
            sum3 = separation_negative[i] / (separation_positive[i] + separation_negative[i])
            topsis_scores.append(sum3)

        # Add columns for TOPSIS score and rank to the original DataFrame
        df['TOPSIS Score'] = topsis_scores
        df['Rank'] = df['TOPSIS Score'].rank(ascending=False)
        # Write the updated DataFrame to a new CSV file
        df.to_csv(result_csv, index=False)
        print(f"TOPSIS analysis completed. Results saved at {result_csv}")

    except FileNotFoundError as fnf:
        print(f"Error : {fnf}")
        print(f"Error: {input_csv} not found. Please provide a valid path")

    except ValueError as ve:
        print(f"Error: {ve}")
        print("Please provide the correct values of weights and impacts")

    except Exception as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(description='TOPSIS application')
    parser.add_argument('input_csv', type=str, help='Path to the input CSV file')
    parser.add_argument('weights', type=str, help='Comma-separated weights for each column')
    parser.add_argument('impacts', type=str, help='Impacts for each column as "+" or "-"')
    parser.add_argument('result_csv', type=str, help='Path to the output CSV file for TOPSIS results')

    args = parser.parse_args()

    # Function to calculate and compile the final result into a csv result file
    topsis(args.input_csv, args.weights, args.impacts, args.result_csv)


if __name__ == "__main__":
    main()
