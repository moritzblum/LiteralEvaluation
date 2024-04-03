

if __name__ == "__main__":
    # Load numpy array from file
    import numpy as np

    # Load the numerical literal array from file
    literals_num = np.load("data/FB15k-237/numerical_literals_rep.npy", allow_pickle=True)
    print(literals_num.shape)

    # Load the text literal array from file
    literals_text = np.load("data/FB15k-237/text_literals_rep.npy", allow_pickle=True)
    print(literals_text.shape)

    # Load original numerical literals
    orig_lits_num = np.load("numerical_literals_org.npy", allow_pickle=True)
    print(orig_lits_num.shape)

    # Load variation 1 numerical literal array from file
    literals_num_v1 = np.load("data/FB15k-237/numerical_literals_rep_attr.npy", allow_pickle=True)
    print(literals_num_v1.shape)

    # Load variation 1 numerical literal array from file
    literals_num_v1 = np.load("data/FB15k-237/numerical_literals_rep_attr_mean.npy", allow_pickle=True)
    print(literals_num_v1.shape)

    # Load variation 2 numerical literal array from file
    literals_num_v2 = np.load("data/FB15k-237/numerical_literals_rep_filtered_100.npy", allow_pickle=True)
    print(literals_num_v2.shape)

    # Load variation 3 text literal array from file
    literals_text_v3 = np.load("data/FB15k-237/text_literals_rep_clustered_100_mean.npy", allow_pickle=True)
    print(literals_text_v3.shape)
    print(literals_text_v3[0:5])
