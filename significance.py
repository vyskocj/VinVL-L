import re
import json
import tqdm
import argparse

import numpy as np
import pandas as pd


def accuracy_score(df, model_name):
    acc = 0
    for i in range(len(df)):
        if df["true_answer"][i][0] in df[model_name][i]:
            acc += 1

    return acc / len(df)


def get_dataframe(m_1, m_2):
    m1_preds = []
    m2_preds = []
    true_ans = []
    for i in range(len(m_1)):
        # just check that question ids are matching (it should match)
        if model_1[i]["question"] != m_2[i]["question"]:
            # this should never occur
            print(f'Questions are different: {m_1[i]["question"]} vs. {m_1[i]["question"]}')
            continue

        m1_preds.append(m_1[i]["predicted"][0:1])
        m2_preds.append(m_2[i]["predicted"][0:1])
        true_ans.append(m_2[i]["true_answer"])

    return pd.DataFrame({
        args.model_1: m1_preds,
        args.model_2: m2_preds,
        "true_answer": true_ans
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--model_1", default="", type=str, required=True, help="Top 5 predictions of the 1st model.")
    parser.add_argument("--model_2", default="", type=str, required=True, help="Top 5 predictions of the 2nd model.")

    # Optional parameters
    parser.add_argument("--num_iters", default=1_000, type=int, help="Number of iterations (typically at least 1,000).")

    # Parse arguments
    args = parser.parse_args()
    print(f"Script called with: {args}\n")

    # load the inputs
    model_1 = json.load(open(args.model_1, "r"))
    model_2 = json.load(open(args.model_2, "r"))
    df = get_dataframe(model_1, model_2)

    # Statistical Significance
    np.random.seed(423)  # lock seeds

    # compute the difference
    acc_1 = accuracy_score(df, args.model_1)
    acc_2 = accuracy_score(df, args.model_2)
    vanilla_diff = abs(acc_1 - acc_2)

    re_pattern = r'(\w+)\.?\w*$'  # to get model name, e.g., from file: "top5-finalOrBest.json", "top5-2nd_model.json"
    print(f"Accuracy of model 1 ({re.search(re_pattern, args.model_1).group(1)}): {acc_1 * 100:.2f}")
    print(f"Accuracy of model 2 ({re.search(re_pattern, args.model_2).group(1)}): {acc_1 * 100:.2f}")
    print(f"Accuracy difference: {vanilla_diff * 100:.2f}")

    # significance
    r = 0
    for _ in tqdm.tqdm(range(args.num_iters), total=args.num_iters):
        idx = np.random.rand(len(df)) < .5

        # passing numpy array to bypass column alignment
        df.loc[idx, [args.model_1, args.model_2]] = df.loc[idx, [args.model_2, args.model_1]].to_numpy()

        # compute the accuracy of both models and make the difference
        acc_1 = accuracy_score(df, args.model_1)
        acc_2 = accuracy_score(df, args.model_2)
        accuracy_diff = abs(acc_1 - acc_2)

        # check the changes for: all questions, and binary & open ones
        if accuracy_diff >= vanilla_diff:
            r += 1

        # return the changes
        df.loc[idx, [args.model_1, args.model_2]] = df.loc[idx, [args.model_2, args.model_1]].to_numpy()

    print(f"The result is (need bellow 0.05 to reject H0 - no diff. between system A and B):")
    print(f"p-value = {(r + 1) / (args.num_iters + 1):.3f}")
