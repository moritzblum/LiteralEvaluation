import argparse
import json

if __name__ == "__main__":
    # Parse arguments (txt file and json file)
    parser = argparse.ArgumentParser(description='Evaluate the scores of the literals')
    parser.add_argument('--txt', type=str, help='The txt file containing the literals')
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
    with open(args.txt) as f:
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
    overall_mean_rank_head = 0
    overall_mean_reciprocal_rank_head = 0
    overall_mean_rank_tail = 0
    overall_mean_reciprocal_rank_tail = 0
    for key in ["1-1", "1-n", "n-1", "n-n"]:
        list = ordered_samples[key]
        # Sum the ranks and reciprocal ranks of the samples in the list
        mean_rank_head = 0
        mean_reciprocal_rank_head = 0
        mean_rank_tail = 0
        mean_reciprocal_rank_tail = 0
        for sample in list:
            mean_rank_head += int(sample[3])
            mean_reciprocal_rank_head += 1/int(sample[3])
            mean_rank_tail += int(sample[4])
            mean_reciprocal_rank_tail += 1/int(sample[4])

        # Add the mean ranks and mean reciprocal ranks to the overall ones
        overall_mean_rank_head += mean_rank_head
        overall_mean_reciprocal_rank_head += mean_reciprocal_rank_head
        overall_mean_rank_tail += mean_rank_tail
        overall_mean_reciprocal_rank_tail += mean_reciprocal_rank_tail

        # Calculate the mean ranks and mean reciprocal ranks for the list and print them
        mean_rank_head /= len(list)
        mean_reciprocal_rank_head /= len(list)
        mean_rank_tail /= len(list)
        mean_reciprocal_rank_tail /= len(list)
        print(key)
        print("Mean rank for head: " + str(mean_rank_head))
        print("Mean reciprocal rank for head: " + str(mean_reciprocal_rank_head))
        print("Mean rank for tail: " + str(mean_rank_tail))
        print("Mean reciprocal rank for tail: " + str(mean_reciprocal_rank_tail))
        print()

    # Overall
    len_overall = len(ordered_samples["1-1"]) + len(ordered_samples["1-n"]) + len(ordered_samples["n-1"]) + len(ordered_samples["n-n"])
    overall_mean_rank_head /= len_overall
    overall_mean_reciprocal_rank_head /= len_overall
    overall_mean_rank_tail /= len_overall
    overall_mean_reciprocal_rank_tail /= len_overall
    print("Overall")
    print("Mean rank for head: " + str(overall_mean_rank_head))
    print("Mean reciprocal rank for head: " + str(overall_mean_reciprocal_rank_head))
    print("Mean rank for tail: " + str(overall_mean_rank_tail))
    print("Mean reciprocal rank for tail: " + str(overall_mean_reciprocal_rank_tail))







