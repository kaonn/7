
let write_arrays data fmt = 
  match fmt with 
  | COMPACT -> 
    Npy.write data.train_ctx "Data/train_ctxs.npz";
    Npy.write data.train_ctx_n "Data/train_ctxs_n.npz";
    Npy.write3 data.train_goals "Data/train_goals.npz";
    Npy.write data.train_goals_n "Data/train_goals_n.npz";
    Npy.write2 data.train_meta "Data/train_meta.npz";
    Npy.write2 data.train_labels "Data/train_labels.npz";

    Npy.write data.test_ctx "Data/test_ctxs.npz";
    Npy.write data.test_ctx_n "Data/test_ctxs_n.npz";
    Npy.write3 data.test_goals "Data/test_goals.npz";
    Npy.write data.test_goals_n "Data/test_goals_n.npz";
    Npy.write2 data.test_meta "Data/test_meta.npz";
    Npy.write2 data.test_labels "Data/test_labels.npz";

  | BON -> 
    Npy.write data.train_ctx "Data/bon_train_ctxs.npz";
    Npy.write data.train_ctx_n "Data/bon_train_ctxs_n.npz";
    Npy.write3 data.train_goals "Data/bon_train_goals.npz";
    Npy.write data.train_goals_n "Data/bon_train_goals_n.npz";
    Npy.write2 data.train_meta "Data/bon_train_meta.npz";
    Npy.write2 data.train_labels "Data/bon_train_labels.npz";

    Npy.write data.test_ctx "Data/bon_test_ctxs.npz";
    Npy.write data.test_ctx_n "Data/bon_test_ctxs_n.npz";
    Npy.write3 data.test_goals "Data/bon_test_goals.npz";
    Npy.write data.test_goals_n "Data/bon_test_goals_n.npz";
    Npy.write2 data.test_meta "Data/bon_test_meta.npz";
    Npy.write2 data.test_labels "Data/bon_test_labels.npz";
 | OH -> 
    Npy.write data.train_ctx "Data/oh_train_ctxs.npz";
    Npy.write data.train_ctx_n "Data/oh_train_ctxs_n.npz";
    Npy.write3 data.train_goals "Data/oh_train_goals.npz";
    Npy.write data.train_goals_n "Data/oh_train_goals_n.npz";
    Npy.write2 data.train_meta "Data/oh_train_meta.npz";
    Npy.write2 data.train_labels "Data/oh_train_labels.npz";

    Npy.write data.test_ctx "Data/oh_test_ctxs.npz";
    Npy.write data.test_ctx_n "Data/oh_test_ctxs_n.npz";
    Npy.write3 data.test_goals "Data/oh_test_goals.npz";
    Npy.write data.test_goals_n "Data/oh_test_goals_n.npz";
    Npy.write2 data.test_meta "Data/oh_test_meta.npz";
    Npy.write2 data.test_labels "Data/oh_test_labels.npz";
