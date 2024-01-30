import argparse
import pandas as pd
import random

if __name__ == "__main__":
    # Get argument for dataset to use
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FB15k-237", choices=["FB15k-237", "YAGO3-10"],
                        help="Dataset to use. Default: FB15k-237. Other option: YAGO3-10")
    args = parser.parse_args()
    DATASET_NAME = args.dataset

    # Load numerical literal txt file as a dataframe
    num_literals_df = pd.read_csv(f"data/{DATASET_NAME}/numerical_literals.txt", sep="\t", header=None)

    # Filter out every literal that has an entitiy which does not have the class label "human"
    # First, load the class mappings of the entities
    entity_class_df = pd.read_csv(f"data/{DATASET_NAME}/final_mapping.csv", sep=";")
    # Then, get all entities that have the class label "human"
    human_entities = entity_class_df[entity_class_df["class_label"] == "human"]["dataset_entity"].unique()
    # Lastly, filter the literal dataframe accordingly
    human_literals_df = num_literals_df[num_literals_df[0].isin(human_entities)]

    # Enrich the human entities with a new literal type "net_worth" in a separate literal file
    net_worth_literal_df = pd.DataFrame(human_entities, columns=["entity"])
    net_worth_literal_df["attr_rel"] = "net_worth"

    # Give 50% of the entities a high random net worth and the other 50% a low random net worth
    # First, split the entities randomly into two groups
    split_index = int(len(human_entities) / 2)
    high_net_worth_entities = human_entities[:split_index]
    low_net_worth_entities = human_entities[split_index:]
    # Then, assign a random net worth to the entities
    for entity in high_net_worth_entities:
        net_worth_literal_df.loc[net_worth_literal_df["entity"] == entity, "literal_value"] = random.randint(7, 10)
    for entity in low_net_worth_entities:
        net_worth_literal_df.loc[net_worth_literal_df["entity"] == entity, "literal_value"] = random.randint(1, 4)

    # Randomly shuffle the new dataset
    net_worth_literal_df = net_worth_literal_df.sample(frac=1).reset_index(drop=True)

    # Save it as a txt file without the header and index
    net_worth_literal_df.to_csv(f"data/{DATASET_NAME}/net_worth_literals.txt", sep="\t", header=False, index=False)
