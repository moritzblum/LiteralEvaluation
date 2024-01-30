import argparse

import numpy as np
import pandas as pd

if __name__ == "__main__":
    # Get argument for dataset to use
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="FB15k-237", choices=["FB15k-237", "YAGO3-10"],
                        help="Dataset to use. Default: FB15k-237. Other option: YAGO3-10")
    args = parser.parse_args()
    DATASET_NAME = args.dataset

    # Load FB15k-237 dataset with the useful literals
    train_df = pd.read_csv(f"data/{DATASET_NAME}/train.txt", sep="\t", header=None)
    valid_df = pd.read_csv(f"data/{DATASET_NAME}/valid.txt", sep="\t", header=None)
    test_df = pd.read_csv(f"data/{DATASET_NAME}/test.txt", sep="\t", header=None)

    # Load the class mappings of the entities
    entity_class_df = pd.read_csv(f"data/{DATASET_NAME}/final_mapping.csv", sep=";")

    # Filter out every entity in train, valid and test that does not have the class label "human"
    # First, get all entities that have the class label "human"
    human_entities = entity_class_df[entity_class_df["class_label"] == "human"]["dataset_entity"].unique()
    # Then, filter the datasets accordingly
    train_humans_df = train_df[train_df[0].isin(human_entities)]
    valid_humans_df = valid_df[valid_df[0].isin(human_entities)]
    test_humans_df = test_df[test_df[0].isin(human_entities)]

    # Log the number of triples in train, valid and test
    print("Number of triples in train, valid and test:")
    print(f"Train: {len(train_df)}")
    print(f"Valid: {len(valid_df)}")
    print(f"Test: {len(test_df)}")

    # Enrich the datasets with the relation (rich_person, has_net_worth_category, rich)
    # if the net worth is over 1 million

    # First, get all entities that have a net worth over 1 million in train, valid and test
    # Load the net_worth literals of the human entities
    literals_df = pd.read_csv(f"data/{DATASET_NAME}/net_worth_literals.txt", sep="\t", header=None)
    # Get every entity that has a net worth over 7 (rich humans have net worth of either 7, 8 or 9)
    human_entities_rich = literals_df[literals_df[2] >= 7][0].unique()

    # Filter the datasets accordingly
    train_humans_rich_df = train_df[train_df[0].isin(human_entities_rich)]
    valid_humans_rich_df = valid_df[valid_df[0].isin(human_entities_rich)]
    test_humans_rich_df = test_df[test_df[0].isin(human_entities_rich)]

    # Log the number of unique rich humans in train, valid, and test
    print(f"Number of unique rich humans in total: {len(human_entities_rich)}")
    print("Number of unique rich humans in train, valid, and test:")
    print(f"Train: {len(train_humans_rich_df[0].unique())}")
    print(f"Valid: {len(valid_humans_rich_df[0].unique())}")
    print(f"Test: {len(test_humans_rich_df[0].unique())}")

    # Then, create a new dataframe with the rich human entities and their new relation
    new_relation_train_df = pd.DataFrame(columns=[0, 1, 2])
    new_relation_df_valid = pd.DataFrame(columns=[0, 1, 2])
    new_relation_df_test = pd.DataFrame(columns=[0, 1, 2])

    # Get the 70/15/15 split of the unique rich human entities
    valid_and_test_split = int(len(human_entities_rich) * 0.15)
    # Test gets 15% of the rich human entities (that exist in test already)
    test_humans_rich_split = np.random.choice(test_humans_rich_df[0].unique(), size=valid_and_test_split, replace=False)
    # Valid gets 15% of the rich human entities (that exist in valid already and were not added to test before)
    valid_humans_rich_split = np.random.choice(valid_humans_rich_df[~valid_humans_rich_df[0].isin(test_humans_rich_split)][0].unique(), size=valid_and_test_split, replace=False)
    # Train gets the rest
    train_humans_rich_split = np.setdiff1d(human_entities_rich, np.concatenate((test_humans_rich_split, valid_humans_rich_split)))

    # Add the new relation to the new dataframes (70% to train, 15% to valid, 15% to test)
    new_relation_train_df[0] = train_humans_rich_split
    new_relation_train_df[1] = "has_net_worth_category"
    new_relation_train_df[2] = "rich"
    new_relation_df_valid[0] = valid_humans_rich_split
    new_relation_df_valid[1] = "has_net_worth_category"
    new_relation_df_valid[2] = "rich"
    new_relation_df_test[0] = test_humans_rich_split
    new_relation_df_test[1] = "has_net_worth_category"
    new_relation_df_test[2] = "rich"

    # Concatenate the new datasets with the original ones (both for all entities and only for human entities)
    train_df = pd.concat([train_df, new_relation_train_df], ignore_index=True)
    valid_df = pd.concat([valid_df, new_relation_df_valid], ignore_index=True)
    test_df = pd.concat([test_df, new_relation_df_test], ignore_index=True)

    train_humans_df = pd.concat([train_humans_df, new_relation_train_df], ignore_index=True)
    valid_humans_df = pd.concat([valid_humans_df, new_relation_df_valid], ignore_index=True)
    test_humans_df = pd.concat([test_humans_df, new_relation_df_test], ignore_index=True)

    # Log the number of triples in train, valid and test after adding the new relation
    print("Number of triples in train, valid and test after adding the new relation:")
    print(f"Train: {train_df.shape[0]} ({len(new_relation_train_df)} new triples)")
    print(f"Valid: {valid_df.shape[0]} ({len(new_relation_df_valid)} new triples)")
    print(f"Test: {test_df.shape[0]} ({len(new_relation_df_test)} new triples)")

    print("Number of triples in train, valid and test after adding the new relation (only humans):")
    print(f"Train: {train_humans_df.shape[0]} ({len(new_relation_train_df)} new triples)")
    print(f"Valid: {valid_humans_df.shape[0]} ({len(new_relation_df_valid)} new triples)")
    print(f"Test: {test_humans_df.shape[0]} ({len(new_relation_df_test)} new triples)")

    # Save the datasets as txt files with all entities
    train_df.to_csv(f"data/{DATASET_NAME}/train_all.txt", sep="\t", header=False, index=False)
    valid_df.to_csv(f"data/{DATASET_NAME}/valid_all.txt", sep="\t", header=False, index=False)
    test_df.to_csv(f"data/{DATASET_NAME}/test_all.txt", sep="\t", header=False, index=False)

    # Save the datasets as txt files with only human entities
    train_humans_df.to_csv(f"data/{DATASET_NAME}/train_only_humans.txt", sep="\t", header=False, index=False)
    valid_humans_df.to_csv(f"data/{DATASET_NAME}/valid_only_humans.txt", sep="\t", header=False, index=False)
    test_humans_df.to_csv(f"data/{DATASET_NAME}/test_only_humans.txt", sep="\t", header=False, index=False)