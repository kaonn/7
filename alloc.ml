
#load "unix.cma";;
#load "str.cma";;
#load "bigarray.cma";;

let mAX_NUM_NODES = 100
let mAX_CONTEXT_LEN = 5
let nUM_TRAINING = 20000
let nUM_TEST = 2000
let training_contexts =
  Bigarray.Genarray.create Bigarray.Int64 Bigarray.C_layout [|nUM_TRAINING; mAX_CONTEXT_LEN; mAX_NUM_NODES; 2|]
let training_contexts_nodes =
  Bigarray.Genarray.create Bigarray.Int64 Bigarray.C_layout [|nUM_TRAINING; mAX_CONTEXT_LEN; mAX_NUM_NODES; 6|]
let training_goals =
  Bigarray.Array3.create Bigarray.Int64 Bigarray.C_layout nUM_TRAINING mAX_NUM_NODES 2
let training_goals_nodes =
  Bigarray.Array3.create Bigarray.Int64 Bigarray.C_layout nUM_TRAINING mAX_NUM_NODES 6
let training_meta =
  Bigarray.Array2.create Bigarray.Int64 Bigarray.C_layout nUM_TRAINING 3
let training_labels =
  Bigarray.Array2.create Bigarray.Int64 Bigarray.C_layout nUM_TRAINING 13

;;
    Bigarray.Genarray.fill training_contexts (Int64.of_int (-1));
    Bigarray.Genarray.fill training_contexts_nodes (Int64.of_int (0));
    Bigarray.Array3.fill training_goals (Int64.of_int (-1));
    Bigarray.Array3.fill training_goals_nodes (Int64.of_int (0));
    Bigarray.Array2.fill training_labels (Int64.of_int (0));

;;

let test_contexts =
  Bigarray.Genarray.create Bigarray.Int64 Bigarray.C_layout [|nUM_TEST; mAX_CONTEXT_LEN; mAX_NUM_NODES; 2|]
let test_contexts_nodes =
  Bigarray.Genarray.create Bigarray.Int64 Bigarray.C_layout [|nUM_TEST; mAX_CONTEXT_LEN; mAX_NUM_NODES; 6|]
let test_goals =
  Bigarray.Array3.create Bigarray.Int64 Bigarray.C_layout nUM_TEST mAX_NUM_NODES 2
let test_goals_nodes =
  Bigarray.Array3.create Bigarray.Int64 Bigarray.C_layout nUM_TEST mAX_NUM_NODES 6
let test_meta =
  Bigarray.Array2.create Bigarray.Int64 Bigarray.C_layout nUM_TEST 3
let test_labels =
  Bigarray.Array2.create Bigarray.Int64 Bigarray.C_layout nUM_TEST 13
;;
    Bigarray.Genarray.fill test_contexts (Int64.of_int (-1));
    Bigarray.Genarray.fill test_contexts_nodes (Int64.of_int (0));
    Bigarray.Array3.fill test_goals (Int64.of_int (-1));
    Bigarray.Array3.fill test_goals_nodes (Int64.of_int (0));
    Bigarray.Array2.fill test_labels (Int64.of_int (0));
