
#load "unix.cma";;
#load "str.cma";;
#load "bigarray.cma";;

type data_format = COMPACT | OH | BON

type ('a, 'b, 'c) data = 
  {
    mAX_NUM_NODES : int;
    mAX_CONTEXT_LEN : int;
    nUM_TRAINING : int;
    nUM_TEST : int;
    train_ctx : ('a, 'b, 'c) Bigarray.Genarray.t;
    train_ctx_n : ('a, 'b, 'c) Bigarray.Genarray.t;
    train_goals : ('a, 'b, 'c) Bigarray.Array3.t;
    train_goals_n : ('a, 'b, 'c) Bigarray.Genarray.t;
    train_meta : ('a, 'b, 'c) Bigarray.Array2.t;
    train_labels : ('a, 'b, 'c) Bigarray.Array2.t; 

    test_ctx : ('a, 'b, 'c) Bigarray.Genarray.t;
    test_ctx_n : ('a, 'b, 'c) Bigarray.Genarray.t;
    test_goals : ('a, 'b, 'c) Bigarray.Array3.t;
    test_goals_n : ('a, 'b, 'c) Bigarray.Genarray.t;
    test_meta : ('a, 'b, 'c) Bigarray.Array2.t;
    test_labels : ('a, 'b, 'c) Bigarray.Array2.t; 

  }

let alloc dformat = 
  match dformat with 
  | OH -> 
    let mAX_NUM_NODES = 100 in 
    let mAX_CONTEXT_LEN = 5 in 
    let nUM_TRAINING = 20000 in 
    let nUM_TEST = 2000 in
    let training_contexts =
      Bigarray.Genarray.create Bigarray.Int8_unsigned Bigarray.C_layout [|nUM_TRAINING; mAX_CONTEXT_LEN; mAX_NUM_NODES; mAX_NUM_NODES|] in 
    let training_contexts_nodes =
      Bigarray.Genarray.create Bigarray.Int8_unsigned Bigarray.C_layout [|nUM_TRAINING; mAX_CONTEXT_LEN; mAX_NUM_NODES; 329|] in
    let training_goals =
      Bigarray.Array3.create Bigarray.Int8_unsigned Bigarray.C_layout nUM_TRAINING mAX_NUM_NODES mAX_NUM_NODES in
    let training_goals_nodes =
      Bigarray.Genarray.create Bigarray.Int8_unsigned Bigarray.C_layout [|nUM_TRAINING; mAX_NUM_NODES; 329|] in
    let training_meta =
      Bigarray.Array2.create Bigarray.Int8_unsigned Bigarray.C_layout nUM_TRAINING 7 in 
    let training_labels =
      Bigarray.Array2.create Bigarray.Int8_unsigned Bigarray.C_layout nUM_TRAINING 13 in

    let _ = 
      Bigarray.Genarray.fill training_contexts (0);
      Bigarray.Genarray.fill training_contexts_nodes (0);
      Bigarray.Array3.fill training_goals (0);
      Bigarray.Genarray.fill training_goals_nodes (0);
      Bigarray.Array2.fill training_labels (0);
    in 

    let test_contexts =
      Bigarray.Genarray.create Bigarray.Int8_unsigned Bigarray.C_layout [|nUM_TEST; mAX_CONTEXT_LEN; mAX_NUM_NODES; mAX_NUM_NODES|] in 
    let test_contexts_nodes =
      Bigarray.Genarray.create Bigarray.Int8_unsigned Bigarray.C_layout [|nUM_TEST; mAX_CONTEXT_LEN; mAX_NUM_NODES; 329|] in
    let test_goals =
      Bigarray.Array3.create Bigarray.Int8_unsigned Bigarray.C_layout nUM_TEST mAX_NUM_NODES mAX_NUM_NODES in
    let test_goals_nodes =
      Bigarray.Genarray.create Bigarray.Int8_unsigned Bigarray.C_layout [|nUM_TEST; mAX_NUM_NODES; 329|] in
    let test_meta =
      Bigarray.Array2.create Bigarray.Int8_unsigned Bigarray.C_layout nUM_TEST 7 in
    let test_labels =
      Bigarray.Array2.create Bigarray.Int8_unsigned Bigarray.C_layout nUM_TEST 13 in

    let _ = 
      Bigarray.Genarray.fill test_contexts (0);
      Bigarray.Genarray.fill test_contexts_nodes (0);
      Bigarray.Array3.fill test_goals (0);
      Bigarray.Genarray.fill test_goals_nodes (0);
      Bigarray.Array2.fill test_labels (0);
    in 
    
    {
      mAX_NUM_NODES = mAX_NUM_NODES;
      mAX_CONTEXT_LEN = mAX_CONTEXT_LEN;
      nUM_TRAINING = nUM_TRAINING;
      nUM_TEST = nUM_TEST;
      train_ctx = training_contexts;
      train_ctx_n = training_contexts_nodes;
      train_goals = training_goals;
      train_goals_n = training_goals_nodes;
      train_meta = training_meta;
      train_labels = training_labels;

      test_ctx = test_contexts;
      test_ctx_n = test_contexts_nodes;
      test_goals = test_goals;
      test_goals_n = test_goals_nodes;
      test_meta = test_meta;
      test_labels =test_labels;
    }
  | BON ->
    let mAX_NUM_NODES = 100 in 
    let mAX_CONTEXT_LEN = 5 in 
    let nUM_TRAINING = 20000 in 
    let nUM_TEST = 2000 in
    let training_contexts =
      Bigarray.Genarray.create Bigarray.Int8_unsigned Bigarray.C_layout [|nUM_TRAINING; mAX_NUM_NODES; mAX_NUM_NODES|] in 
    let training_contexts_nodes =
      Bigarray.Genarray.create Bigarray.Int8_unsigned Bigarray.C_layout [|nUM_TRAINING; 329|] in
    let training_goals =
      Bigarray.Array3.create Bigarray.Int8_unsigned Bigarray.C_layout nUM_TRAINING mAX_NUM_NODES mAX_NUM_NODES in
    let training_goals_nodes =
      Bigarray.Genarray.create Bigarray.Int8_unsigned Bigarray.C_layout [|nUM_TRAINING; 329|] in
    let training_meta =
      Bigarray.Array2.create Bigarray.Int8_unsigned Bigarray.C_layout nUM_TRAINING 7 in 
    let training_labels =
      Bigarray.Array2.create Bigarray.Int8_unsigned Bigarray.C_layout nUM_TRAINING 13 in

    let _ = 
      Bigarray.Genarray.fill training_contexts (0);
      Bigarray.Genarray.fill training_contexts_nodes (0);
      Bigarray.Array3.fill training_goals (0);
      Bigarray.Genarray.fill training_goals_nodes (0);
      Bigarray.Array2.fill training_labels (0);
    in 

    let test_contexts =
      Bigarray.Genarray.create Bigarray.Int8_unsigned Bigarray.C_layout [|nUM_TEST; mAX_NUM_NODES; mAX_NUM_NODES|] in 
    let test_contexts_nodes =
      Bigarray.Genarray.create Bigarray.Int8_unsigned Bigarray.C_layout [|nUM_TEST;  329|] in
    let test_goals =
      Bigarray.Array3.create Bigarray.Int8_unsigned Bigarray.C_layout nUM_TEST mAX_NUM_NODES mAX_NUM_NODES in
    let test_goals_nodes =
      Bigarray.Genarray.create Bigarray.Int8_unsigned Bigarray.C_layout [|nUM_TEST; 329|] in
    let test_meta =
      Bigarray.Array2.create Bigarray.Int8_unsigned Bigarray.C_layout nUM_TEST 7 in
    let test_labels =
      Bigarray.Array2.create Bigarray.Int8_unsigned Bigarray.C_layout nUM_TEST 13 in

    let _ = 
      Bigarray.Genarray.fill test_contexts (0);
      Bigarray.Genarray.fill test_contexts_nodes (0);
      Bigarray.Array3.fill test_goals (0);
      Bigarray.Genarray.fill test_goals_nodes (0);
      Bigarray.Array2.fill test_labels (0);
    in 
    
    {
      mAX_NUM_NODES = mAX_NUM_NODES;
      mAX_CONTEXT_LEN = mAX_CONTEXT_LEN;
      nUM_TRAINING = nUM_TRAINING;
      nUM_TEST = nUM_TEST;
      train_ctx = training_contexts;
      train_ctx_n = training_contexts_nodes;
      train_goals = training_goals;
      train_goals_n = training_goals_nodes;
      train_meta = training_meta;
      train_labels = training_labels;

      test_ctx = test_contexts;
      test_ctx_n = test_contexts_nodes;
      test_goals = test_goals;
      test_goals_n = test_goals_nodes;
      test_meta = test_meta;
      test_labels =test_labels;
    }
  | COMPACT -> 
    let mAX_NUM_NODES = 100 in 
    let mAX_CONTEXT_LEN = 5 in 
    let nUM_TRAINING = 20000 in 
    let nUM_TEST = 2000 in
    let training_contexts =
      Bigarray.Genarray.create Bigarray.Int8_unsigned Bigarray.C_layout [|nUM_TRAINING; mAX_CONTEXT_LEN; mAX_NUM_NODES; 2|] in 
    let training_contexts_nodes =
      Bigarray.Genarray.create Bigarray.Int8_unsigned Bigarray.C_layout [|nUM_TRAINING; mAX_CONTEXT_LEN; mAX_NUM_NODES; 6|] in
    let training_goals =
      Bigarray.Array3.create Bigarray.Int8_unsigned Bigarray.C_layout nUM_TRAINING mAX_NUM_NODES mAX_NUM_NODES in
    let training_goals_nodes =
      Bigarray.Genarray.create Bigarray.Int8_unsigned Bigarray.C_layout [|nUM_TRAINING; mAX_NUM_NODES; 6|] in
    let training_meta =
      Bigarray.Array2.create Bigarray.Int8_unsigned Bigarray.C_layout nUM_TRAINING 7 in 
    let training_labels =
      Bigarray.Array2.create Bigarray.Int8_unsigned Bigarray.C_layout nUM_TRAINING 13 in

    let _ = 
      Bigarray.Genarray.fill training_contexts (0);
      Bigarray.Genarray.fill training_contexts_nodes (0);
      Bigarray.Array3.fill training_goals (0);
      Bigarray.Genarray.fill training_goals_nodes (0);
      Bigarray.Array2.fill training_labels (0);
    in 

    let test_contexts =
      Bigarray.Genarray.create Bigarray.Int8_unsigned Bigarray.C_layout [|nUM_TEST; mAX_CONTEXT_LEN; mAX_NUM_NODES; 2|] in 
    let test_contexts_nodes =
      Bigarray.Genarray.create Bigarray.Int8_unsigned Bigarray.C_layout [|nUM_TEST; mAX_CONTEXT_LEN; mAX_NUM_NODES; 6|] in
    let test_goals =
      Bigarray.Array3.create Bigarray.Int8_unsigned Bigarray.C_layout nUM_TEST mAX_NUM_NODES mAX_NUM_NODES in
    let test_goals_nodes =
      Bigarray.Genarray.create Bigarray.Int8_unsigned Bigarray.C_layout [|nUM_TEST; mAX_NUM_NODES; 6|] in
    let test_meta =
      Bigarray.Array2.create Bigarray.Int8_unsigned Bigarray.C_layout nUM_TEST 7 in
    let test_labels =
      Bigarray.Array2.create Bigarray.Int8_unsigned Bigarray.C_layout nUM_TEST 13 in

    let _ = 
      Bigarray.Genarray.fill test_contexts (0);
      Bigarray.Genarray.fill test_contexts_nodes (0);
      Bigarray.Array3.fill test_goals (0);
      Bigarray.Genarray.fill test_goals_nodes (0);
      Bigarray.Array2.fill test_labels (0);
    in 
    
    {
      mAX_NUM_NODES = mAX_NUM_NODES;
      mAX_CONTEXT_LEN = mAX_CONTEXT_LEN;
      nUM_TRAINING = nUM_TRAINING;
      nUM_TEST = nUM_TEST;
      train_ctx = training_contexts;
      train_ctx_n = training_contexts_nodes;
      train_goals = training_goals;
      train_goals_n = training_goals_nodes;
      train_meta = training_meta;
      train_labels = training_labels;

      test_ctx = test_contexts;
      test_ctx_n = test_contexts_nodes;
      test_goals = test_goals;
      test_goals_n = test_goals_nodes;
      test_meta = test_meta;
      test_labels =test_labels;
    }

