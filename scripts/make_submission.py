import numpy as np
import pandas as pd

if __name__ == "__main__":
    sample_submission_file = '../data/sampleSubmission.csv'
    assembled_file         = 'ensemble.csv'
    submission_file        = '../ensemble-submission.csv'

    sample_submission_df = pd.read_csv(sample_submission_file)
    assembled_df         = pd.read_csv(assembled_file)

    ids   = sample_submission_df['id'].values
    probs = assembled_df['pos'].values

    result_df = pd.DataFrame(np.asarray([ids, probs]).T)
    result_df.to_csv(submission_file, index=False, header=['id', 'sentiment'])

    print("Wrote submission to file %s." % submission_file)

