# /// script
# dependencies = [
#   "pandas",
#   "scipy",
#   "statsmodels",
# ]
# ///

import argparse
import random
from ast import literal_eval
from pathlib import Path

import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
from statsmodels.stats.inter_rater import fleiss_kappa


def main():
    # fmt: off
    parser = argparse.ArgumentParser(description="Compute human annotation IAA and gold-reference IAA")
    parser.add_argument("--input_path", type=Path, help="Path to the annotations file.")
    args = parser.parse_args()
    # fmt: on

    # Load annotations and fix some typings for consistency
    df = pd.read_csv(args.input_path).dropna()
    df["Gold Answer"] = df["Gold Answer"].apply(lambda x: literal_eval(x)[0])
    number_to_letter = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G", 7: "H"}
    df["Gold Answer"] = df["Gold Answer"].map(number_to_letter)
    df = df.rename(
        columns={
            "Gold Answer": "gold_answer",
            "Category": "category",
            "Text": "text",
            "LJ": "annotator_1",
            "Ely": "annotator_2",
            "Conner": "annotator_3",
        }
    )

    # Separate MCF and Generation tasks
    gen_df = df[
        df.category.isin(
            ["filbench|tico19_tgl", "filbench|tatoeba_tgl", "filbench|ntrex128_fil"]
        )
    ].reset_index(drop=True)
    mcf_df = df[
        ~df.category.isin(
            ["filbench|tico19_tgl", "filbench|tatoeba_tgl", "filbench|ntrex128_fil"]
        )
    ].reset_index(drop=True)

    # Compute IAA for MCF (Inter-annotator)
    annotator_columns = ["annotator_1", "annotator_2", "annotator_3"]
    unique_categories = sorted(set(mcf_df[annotator_columns].values.flatten()))
    kappa_matrix = []
    for _, row in mcf_df.iterrows():
        # Count the number of times each category is assigned by the annotators
        category_counts = [0] * len(unique_categories)
        for annotator in annotator_columns:
            category_index = unique_categories.index(row[annotator])
            category_counts[category_index] += 1
        kappa_matrix.append(category_counts)
    kappa_score = fleiss_kappa(kappa_matrix)
    print(f"Intra-annotator Fleiss' Kappa for MCF tasks: {kappa_score:.4f}")

    # Compute IAA for MCF (Intra-annotator)
    def majority_vote_or_random(row):
        votes = [row[annotator] for annotator in annotator_columns]
        vote_count = {vote: votes.count(vote) for vote in set(votes)}
        max_count = max(vote_count.values())
        majority_votes = [
            vote for vote, count in vote_count.items() if count == max_count
        ]
        return (
            random.choice(majority_votes)
            if len(majority_votes) > 1
            else majority_votes[0]
        )

    # Apply the function to determine the majority response or random choice
    mcf_df["majority_response"] = mcf_df.apply(majority_vote_or_random, axis=1)

    # Prepare data for Fleiss' Kappa computation
    kappa_matrix_gold = []
    for _, row in mcf_df.iterrows():
        # Count the number of times each category is assigned by the majority response and gold answer
        category_counts = [0] * len(unique_categories)
        majority_index = unique_categories.index(row["majority_response"])
        gold_index = unique_categories.index(row["gold_answer"])
        category_counts[majority_index] += 1
        category_counts[gold_index] += 1
        kappa_matrix_gold.append(category_counts)

    # Compute Fleiss' Kappa between majority response and gold_answer
    kappa_score_gold = fleiss_kappa(kappa_matrix_gold)
    print(f"Inter-annotator Fleiss' Kappa for MCF tasks: {kappa_score_gold:.4f}")
    print("\n")

    # Compute ROUGE-L
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    annotator_pairs = [
        ("annotator_1", "annotator_2"),
        ("annotator_1", "annotator_3"),
        ("annotator_2", "annotator_3"),
    ]
    rouge_l_scores = {pair: [] for pair in annotator_pairs}
    for _, row in gen_df.iterrows():
        for annotator1, annotator2 in annotator_pairs:
            score = scorer.score(row[annotator1], row[annotator2])
            rouge_l_scores[(annotator1, annotator2)].append(score["rougeL"].fmeasure)
    all_average_rouge_l_scores = [np.mean(scores) for scores in rouge_l_scores.values()]
    overall_average_rouge_l = np.mean(all_average_rouge_l_scores)
    print(
        f"Intra-annotator ROUGE-L (avg.) for all generation tasks: {overall_average_rouge_l:.4f}"
    )

    # Compute ROUGE-L between gold_answer and each annotator, then average
    inter_rouge_l_scores = []
    for _, row in mcf_df.iterrows():
        scores = []
        for annotator in annotator_columns:
            score = scorer.score(row["gold_answer"], row[annotator])
            scores.append(score["rougeL"].fmeasure)
        average_score = np.mean(scores)
        inter_rouge_l_scores.append(average_score)

    # Average the ROUGE-L scores across all instances
    overall_inter_rouge_l = np.mean(inter_rouge_l_scores)
    print(f"Inter-annotator ROUGE-L (avg.) for MCF tasks: {overall_inter_rouge_l:.4f}")


if __name__ == "__main__":
    main()
