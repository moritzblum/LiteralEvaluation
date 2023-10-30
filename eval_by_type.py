import argparse
import json
import numpy as np

if __name__ == "__main__":
    # Parse arguments (txt file and json file)
    parser = argparse.ArgumentParser(description='Evaluate the scores of the literals')
    parser.add_argument('--tsv', type=str, help='The txt file containing the literals')
    parser.add_argument('--json', type=str, help='The json file containing the scores')
    args = parser.parse_args()

    # Load relation types from json file (1-1, 1-N, N-1, N-N) as a dictionary
    with open(args.json) as f:
        relation_types = json.load(f)

    print("Relation types: " + str(relation_types.keys()))

    # Order the literals by relation type
    ordered_samples = {
        "1-1": [],
        "1-n": [],
        "n-1": [],
        "n-n": []
    }
    with open(args.tsv) as f:
        lines = f.readlines()
        for line in lines:
            sample = line.split("\t")
            # Get the relation type
            relation_type = sample[1]
            # Check in which list of the dictionary the relation type is contained
            if relation_type in relation_types["1-1"]:
                ordered_samples["1-1"].append(sample)
            elif relation_type in relation_types["1-n"]:
                ordered_samples["1-n"].append(sample)
            elif relation_type in relation_types["n-1"]:
                ordered_samples["n-1"].append(sample)
            elif relation_type in relation_types["n-n"]:
                ordered_samples["n-n"].append(sample)
            else:
                print("Relation type not found for sample: " + str(sample))

    # Calculate the mean rank and mean reciprocal rank for each list and overall (for head and tail)
    # Mean rank = sum of the ranks of the samples in the list / number of samples in the list
    # Mean reciprocal rank = sum of the reciprocal ranks of the samples in the list / number of samples in the list
    ranks_head_all = []
    ranks_tail_all = []
    for key in ["1-1", "1-n", "n-1", "n-n"]:
        ranks_head = []
        ranks_tail = []
        list = ordered_samples[key]
        # Sum the ranks and reciprocal ranks of the samples in the list

        for sample in list:
            ranks_head.append(int(sample[3]))
            ranks_tail.append(int(sample[4]))

        ranks_head_all.extend(ranks_head)
        ranks_tail_all.extend(ranks_tail)

        # Calculate the mean ranks and mean reciprocal ranks for the list and print them
        mean_rank_head = np.mean(ranks_head)
        mean_reciprocal_rank_head = np.mean(1 / np.array(ranks_head))
        mean_rank_tail = np.mean(ranks_tail)
        mean_reciprocal_rank_tail = np.mean(1 / np.array(ranks_tail))
        mean_rank = np.mean(ranks_head + ranks_tail)
        mean_reciprocal_rank = np.mean(1/np.array(ranks_head + ranks_tail))

        print(f'---{key}---')
        print("Mean rank for head:", mean_rank_head)
        print("Mean reciprocal rank for head:", mean_reciprocal_rank_head)
        print("Mean rank for tail:", mean_rank_tail)
        print("Mean reciprocal rank for tail:", mean_reciprocal_rank_tail)
        print("Mean rank:", mean_rank)
        print("Mean reciprocal rank:", mean_reciprocal_rank)
        print()

    # Overall
    len_overall = len(ordered_samples["1-1"]) + len(ordered_samples["1-n"]) + len(ordered_samples["n-1"]) + len(ordered_samples["n-n"])
    overall_mean_rank_head = np.mean(ranks_head_all)
    overall_mean_reciprocal_rank_head = np.mean(1 / np.array(ranks_head_all))
    overall_mean_rank_tail = np.mean(ranks_tail_all)
    overall_mean_reciprocal_rank_tail = np.mean(1 / np.array(ranks_tail_all))
    overall_mean_rank = np.mean(ranks_head_all + ranks_tail_all)
    overall_mean_reciprocal_rank = np.mean(1 / np.array(ranks_head_all + ranks_tail_all))
    print("--- Overall ---")
    print("Mean rank for head:", overall_mean_rank_head)
    print("Mean reciprocal rank for head:", overall_mean_reciprocal_rank_head)
    print("Mean rank for tail:", overall_mean_rank_tail)
    print("Mean reciprocal rank for tail:", overall_mean_reciprocal_rank_tail)
    print("Mean rank:" , overall_mean_rank)
    print("Mean reciprocal rank:", overall_mean_reciprocal_rank)







