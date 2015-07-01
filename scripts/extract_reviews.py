import sys
import pandas as pd

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: %s <data_file> [target_label]")
        sys.exit(1)

    data_file    = sys.argv[1]
    target_label = sys.argv[2] if len(sys.argv) >= 3 else None

    df = pd.read_csv(data_file, escapechar='\\', delimiter='\t')
    for index, row in df.iterrows():
        if target_label == 'positive' and row['sentiment'] == 1 \
                or target_label == 'negative' and row['sentiment'] == 0 \
                or target_label is None:
            print(row['review'])
