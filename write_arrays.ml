
let write_arrays data fmt = 
  match fmt with 
  | COMPACT -> 
    Npy.write data.train_ctx "Data/train_ctxs.npz";
    Npy.write data.train_ctx_n "Data/train_ctxs_n.npz";
    Npy.write data.train_goals "Data/train_goals.npz";
    Npy.write data.train_goals_n "Data/train_goals_n.npz";
    Npy.write data.train_meta "Data/train_meta.npz";
    Npy.write2 data.train_labels "Data/train_labels.npz";

    Npy.write data.test_ctx "Data/test_ctxs.npz";
    Npy.write data.test_ctx_n "Data/test_ctxs_n.npz";
    Npy.write data.test_goals "Data/test_goals.npz";
    Npy.write data.test_goals_n "Data/test_goals_n.npz";
    Npy.write data.test_meta "Data/test_meta.npz";
    Npy.write2 data.test_labels "Data/test_labels.npz";

  | BON -> 
    Npy.write data.train_ctx "Data/bon1_train_ctxs.npz";
    Npy.write data.train_ctx_n "Data/bon1_train_ctxs_n.npz";
    Npy.write data.train_goals "Data/bon1_train_goals.npz";
    Npy.write data.train_goals_n "Data/bon1_train_goals_n.npz";
    Npy.write data.train_meta "Data/bon1_train_meta.npz";
    Npy.write2 data.train_labels "Data/bon1_train_labels.npz";

    Npy.write data.train_premise_ctx "Data/bon1_train_premise_ctxs.npz";
    Npy.write data.train_premise_ctx_n "Data/bon1_train_premise_ctxs_n.npz";
    Npy.write data.train_premise_goals "Data/bon1_train_premise_goals.npz";
    Npy.write data.train_premise_goals_n "Data/bon1_train_premise_goals_n.npz";
    Npy.write data.train_premise_meta "Data/bon1_train_premise_meta.npz";

    Npy.write data.test_ctx "Data/bon1_test_ctxs.npz";
    Npy.write data.test_ctx_n "Data/bon1_test_ctxs_n.npz";
    Npy.write data.test_goals "Data/bon1_test_goals.npz";
    Npy.write data.test_goals_n "Data/bon1_test_goals_n.npz";
    Npy.write data.test_meta "Data/bon1_test_meta.npz";
    Npy.write2 data.test_labels "Data/bon1_test_labels.npz";

    Npy.write data.test_premise_ctx "Data/bon1_test_premise_ctxs.npz";
    Npy.write data.test_premise_ctx_n "Data/bon1_test_premise_ctxs_n.npz";
    Npy.write data.test_premise_goals "Data/bon1_test_premise_goals.npz";
    Npy.write data.test_premise_goals_n "Data/bon1_test_premise_goals_n.npz";
    Npy.write data.test_premise_meta "Data/bon1_test_premise_meta.npz";
 | OH -> 
    Npy.write data.train_ctx "Data/oh2_train_ctxs.npz";
    Npy.write data.train_ctx_n "Data/oh2_train_ctxs_n.npz";
    Npy.write data.train_goals "Data/oh2_train_goals.npz";
    Npy.write data.train_goals_n "Data/oh2_train_goals_n.npz";
    Npy.write data.train_meta "Data/oh2_train_meta.npz";
    Npy.write2 data.train_labels "Data/oh2_train_labels.npz";

    Npy.write data.train_premise_ctx "Data/oh2_train_premise_ctxs.npz";
    Npy.write data.train_premise_ctx_n "Data/oh2_train_premise_ctxs_n.npz";
    Npy.write data.train_premise_goals "Data/oh2_train_premise_goals.npz";
    Npy.write data.train_premise_goals_n "Data/oh2_train_premise_goals_n.npz";
    Npy.write data.train_premise_meta "Data/oh2_train_premise_meta.npz";

    Npy.write data.test_ctx "Data/oh2_test_ctxs.npz";
    Npy.write data.test_ctx_n "Data/oh2_test_ctxs_n.npz";
    Npy.write data.test_goals "Data/oh2_test_goals.npz";
    Npy.write data.test_goals_n "Data/oh2_test_goals_n.npz";
    Npy.write data.test_meta "Data/oh2_test_meta.npz";
    Npy.write2 data.test_labels "Data/oh2_test_labels.npz";

    Npy.write data.test_premise_ctx "Data/oh2_test_premise_ctxs.npz";
    Npy.write data.test_premise_ctx_n "Data/oh2_test_premise_ctxs_n.npz";
    Npy.write data.test_premise_goals "Data/oh2_test_premise_goals.npz";
    Npy.write data.test_premise_goals_n "Data/oh2_test_premise_goals_n.npz";
    Npy.write data.test_premise_meta "Data/oh2_test_premise_meta.npz";
