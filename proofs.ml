(* ========================================================================= *)
(* Tooling for the generation of the ProofTrace dataset.                     *)
(* ========================================================================= *)

#load "unix.cma";;
#load "str.cma";;
#load "bigarray.cma";;

(* ------------------------------------------------------------------------- *)
(* Marshalling of term to AST-like.                                          *)
(* ------------------------------------------------------------------------- *)


type node_type = N_VAR | N_TYVAR | N_TYAPP | N_CONST | N_COMB | N_ABS
type node = 
    Nodes of node_type * int * int * (node list) | 
    Node of node_type * int * node * node | Single of node_type * int * int * node  | Leaf of node_type * int * int

let type_string ty =
  let rec args_str args =
    match args with
      [] -> ""
    | ty::tail -> Printf.sprintf "[%s]%s"
                                 (type_str ty) (args_str tail)
  and type_str ty =
    match ty with
      Tyvar(v) -> 
      Printf.sprintf "v[%s]"
                                 (String.escaped v)
    | Tyapp(c,args) -> 
      Printf.sprintf "c[%s][%s]"
                                      (String.escaped c) (args_str args)
  in (type_str ty)

let rec term_string tm =
  match tm with
    Var(v,ty) -> 
    Printf.sprintf "v(%s)(%s)"
                                (String.escaped v) (type_string ty)
  | Const(c,ty) -> 
    Printf.sprintf "c(%s)(%s)"
                                  (String.escaped c) (type_string ty)
  | Comb(t1,t2) -> Printf.sprintf "C(%s)(%s)"
                                  (term_string t1) (term_string t2)
  | Abs(t1,t2) -> Printf.sprintf "A(%s)(%s)"
                    (term_string t1) (term_string t2)

;;
exception SIZE
exception CONST
exception TYPEVAR
exception Fail of string


module M = Map.Make(String)
module IntS = Set.Make( 
  struct
    let compare = Pervasives.compare
    type t = int
  end)
    
let constant_index = 
  let m = ref M.empty in
  let _ = List.iteri (fun i (s,_) -> m := M.add s (i+2) !m) (constants ()) in 
  m

let find' c m =
  match c with 
  | "bool" -> 0
  | "fun" -> 1
  | s ->
    begin
    match M.find_opt c m with
    | Some v -> v
    | _ -> if List.length (explode c) != 1 then raise (Fail ("variable with more than 1 char: " ^ c)) else 
        let code = String.get c 0 |> Char.code in 
          if code > 25 then raise (Fail "not alphabetic")
          else code
    end

(* (node index, category, char occurence) *)
let mk_node id nt str fmt : int * int * int = 
  match fmt with 
  | COMPACT -> 
    begin
    match nt with 
    | N_VAR -> if str = "" then raise (Fail "empty string") else (id,0,String.get str 0 |> Char.code) 
    | N_TYVAR -> if str = "" then raise (Fail "empty string") else (id,4,String.get str 0 |> Char.code) 
    | N_TYAPP -> if str = "" then raise (Fail "empty string") else (id,5,find' str (!constant_index))
    | N_CONST -> (id,1,find' str (!constant_index))
    | N_COMB -> (id,2,1)
    | N_ABS -> (id,3,1)
    end
  | OH -> 
    begin
    match nt with 
    | N_VAR -> if str = "" then raise (Fail "empty string") else (id,0,String.get str 0 |> Char.code) 
    | N_TYVAR -> if str = "" then raise (Fail "empty string") else (id,4,String.get str 0 |> Char.code) 
    | N_TYAPP -> if str = "" then raise (Fail "empty string") else (id,5,find' str (!constant_index))
    | N_CONST -> (id,1,find' str (!constant_index))
    | N_COMB -> (id,2,2)
    | N_ABS -> (id,3,3)
    end
  | BON -> 
    begin
    match nt with 
    | N_VAR -> if str = "" then raise (Fail "empty string") else (id,0,(String.get str 0 |> Char.code) + 6) 
    | N_TYVAR -> if str = "" then raise (Fail "empty string") else (id,4,(String.get str 0 |> Char.code) + 6) 
    | N_TYAPP -> if str = "" then raise (Fail "empty string") else (id,5,(find' str (!constant_index)) + 32)
    | N_CONST -> (id,1,(find' str (!constant_index)) + 32)
    | N_COMB -> (id,2,-1)
    | N_ABS -> (id,3,-1)
  end

let mk_write : (int * int * int) -> data_format -> (int * int * int) list = 
  fun (id',c,id) ->
    function
   COMPACT -> [(id',c,id)]
    | OH -> [(id',id,1);(id,id',1)]
    | BON -> [(id',id,1);(id,id',1)]

;;
let term_node (tm : term) (fmt : data_format) mnn : (((int * int * int) list) ref * (int * int * int) list ref -> unit) -> (int * node) = fun cc ->
  let idx = ref 0 in
  let dlevel = ref 0 in
  let dbind = ref M.empty in
  let writes = ref [] in
  let nodes = ref [] in
  let m = !constant_index in

  let get_var tm = 
    match tm with 
      Var(v,ty) -> v
    | _ -> raise (Fail "abstractor didn't bind variable")
  in

  let rec f tm : int * node = 
    if !idx >= mnn then raise (Fail ("max node size reached: " ^ (string_of_int !idx)))
    else
      begin
    match tm with
      Var(v,ty) -> 
      let id,child = g ty in 
      let id' = !idx in 
      let r = (id',Single (N_VAR, id', find' v m, child)) in
      let _ = writes := List.append(mk_write (id',0,id) fmt) (!writes) in
      let _ = nodes := (mk_node id' N_VAR v fmt)::(!nodes) in 
      let _ = idx := !idx + 1 in 
      r
    | Const(c,ty) -> 
      let (id,child) = g ty in 
      let id' = !idx in 
      let r = (id',Single (N_CONST, id', find' c m, child)) in
      let _ = writes := List.append (mk_write (id',0,id) fmt) (!writes) in
      let _ = nodes := (mk_node id' N_CONST c fmt)::(!nodes) in 
      let _ = idx := !idx + 1 in
      r
    | Comb(t1,t2) -> 
      let id1,c1 = f t1 in 
      let id2,c2 = f t2 in 
      let id' = !idx in 
      let r = id',Node(N_COMB, id', c1, c2) in
      let _ = writes := List.concat [mk_write (id',0,id1) fmt; mk_write (id',1,id2) fmt; (!writes)] in
      let _ = nodes := (mk_node id' N_COMB "" fmt)::(!nodes) in 
      let _ = idx := !idx + 1 in
      r
   | Abs(t1,t2) -> 
      let _ = dlevel := !dlevel + 1 in
      let var = get_var t1 in
      let _ = dbind := M.add var (!dlevel) (!dbind) in
      let (id1,c1) = f t1 in 
      let (id2,c2) = f t2 in 
      let id' = !idx in 
      let r = (id',Node(N_ABS, id',c1, c2)) in
      let _ = writes := List.concat [mk_write (id',0,id1) fmt; mk_write (id',1,id2) fmt; (!writes)] in
      let _ = nodes := (mk_node id' N_ABS "" fmt)::(!nodes) in 
      let _ = idx := !idx + 1 in
      let _ = dlevel := !dlevel - 1 in
      r
    end

  and g ty = 
    if !idx >= mnn then raise (Fail ("max node size reached: " ^ (string_of_int !idx)) )
    else
      begin
    match ty with
      Tyvar(v) -> 
      let em = if List.length (explode v) != 1 then raise (Fail ("type var with more than 1 char: " ^ v)) else String.get v 0 |> Char.code in
      let id' = !idx in 
      let r = ((id',Leaf(N_TYVAR, id', em))) in 
      let _ = nodes := (mk_node id' N_TYVAR v fmt)::(!nodes) in 
      let _ = idx := !idx + 1 in
      r
    | Tyapp(c,args) ->
      let _ = if List.length args > 2 then raise (Fail ("more than 2 apps: " ^ (string_of_int !idx))) else () in
      let res = List.map g args in 
      let idxs = List.map (fun (i,_) -> i) res in
      let childs = List.map (fun (_,c) -> c) res in 
      let id' = !idx in 
      let r = (id',Nodes(N_TYAPP, id', find' c m, childs)) in 
      let _ = 
        match idxs with 
        | [] -> () 
        | [id] -> writes := List.append (mk_write (id',0,id) fmt) (!writes) 
        | [id1;id2] -> writes := List.concat [mk_write (id',0,id1) fmt; mk_write (id',1,id2) fmt; (!writes)]
      in
      let _ = nodes := (mk_node id' N_TYAPP c fmt)::(!nodes) in 
      let _ = idx := !idx + 1 in
      r
    end
     in 

     let size,tmn = f tm in 
     if size >= mnn then raise (Fail ("max node size reached: " ^ (string_of_int size)) ) else
     let _ = cc (writes,nodes) in
     (size,tmn)
;;

let check_ctx ctx = 
  if List.length ctx > 5 then raise (Fail "context too big") 
  else ()

let label content =
  match content with
    Prefl(tm) -> 0
  | Ptrans(p1,p2) -> 1
  | Pmkcomb(p1,p2) -> 2 
  | Pabs(p1,tm) -> 3
  | Pbeta(tm) -> 4
  | Passume(tm) -> 5
  | Peqmp(p1,p2) -> 6 
  | Pdeduct(p1,p2) -> 7
  | Pinst(p1,insts) -> 8
  | Pinstt(p1,insts) -> 9
  | Paxiom(tm) -> 10
  | Pdef(tm,name,ty) -> 11
  | Pdeft(p1,tm,name,ty) -> 12

let check_content content =
  match content with
    Prefl(tm) -> []
  | Ptrans(p1,p2) -> 
    let Proof(_,thm1,_) = p1 in 
    let Proof(_,thm2,_) = p2 in 
    let _ = check_ctx (hyp thm1); check_ctx (hyp thm2) in 
    [p1;p2]
  | Pmkcomb(p1,p2) -> 
    let Proof(_,thm1,_) = p1 in 
    let Proof(_,thm2,_) = p2 in 
    let _ = check_ctx (hyp thm1); check_ctx (hyp thm2) in 
    [p1;p2]

  | Pabs(p1,tm) -> 
    let Proof(_,thm1,_) = p1 in 
    let _ = check_ctx (hyp thm1) in [p1]

  | Pbeta(tm) -> []
  | Passume(tm) -> []
  | Peqmp(p1,p2) -> 
    let Proof(_,thm1,_) = p1 in 
    let Proof(_,thm2,_) = p2 in 
    let _ = check_ctx (hyp thm1); check_ctx (hyp thm2) in 
    [p1;p2]
  | Pdeduct(p1,p2) -> 
    let Proof(_,thm1,_) = p1 in 
    let Proof(_,thm2,_) = p2 in 
    let _ = check_ctx (hyp thm1); check_ctx (hyp thm2) in 
    [p1;p2]
  | Pinst(p1,insts) -> raise (Fail "cannot generate inst rules")
  | Pinstt(p1,insts) -> raise (Fail "cannot generate inst rules") 
  | Paxiom(tm) -> []
  | Pdef(tm,name,ty) -> raise (Fail "cannot generate def rules")
  | Pdeft(p1,tm,name,ty) -> raise (Fail "cannot generate def rules")

type data_type = Train | Test | TrainPremise | TestPremise
type prob = CL | GEN 

let flip which = 
  match which with 
  | Train -> TrainPremise
  | Test -> TestPremise

let rec matrify index proof which fmt prob side (dataref : (('a, 'b, 'c) data) ref) =
  let data = !dataref in
  let mnn = data.mAX_NUM_NODES in
  let goals, goals_nodes, meta, contexts, contexts_nodes = 
    match which with
    | Train -> data.train_goals, data.train_goals_n, data.train_meta, data.train_ctx, data.train_ctx_n
    | Test -> data.test_goals, data.test_goals_n, data.test_meta, data.test_ctx, data.test_ctx_n
    | TrainPremise -> data.train_premise_goals, data.train_premise_goals_n, data.train_premise_meta, data.train_premise_ctx, data.train_premise_ctx_n
    | TestPremise-> data.test_premise_goals, data.test_premise_goals_n, data.test_premise_meta, data.test_premise_ctx, data.test_premise_ctx_n
  in
  try
  let Proof(idx,thm,content) = proof in
  let asl,tm = dest_thm thm in
  let _ = check_ctx asl in 
  let _ =  
    match prob with 
    | GEN ->
      let premises = check_content content in 
      List.iteri (
        fun i p -> let _ = matrify index p (flip which) OH CL i dataref in ()
      ) premises
    | _ -> () in 
  let len = List.length asl in
  let goal_cc = 
    match fmt with 
    | COMPACT -> 
      fun (writes,nodes) ->
        List.iter (fun (i,j,k) -> Bigarray.Genarray.set goals [|index; i; j|] k) !writes;
        List.iter (fun (i,j,k) -> Bigarray.Genarray.set goals_nodes [|index;i;j|] k) !nodes
    | OH -> 
        fun (writes,nodes) ->
          List.iter (fun (i,j,k) -> Bigarray.Genarray.set goals [|index; side; i; j|] k) !writes;
          List.iter (fun (i,j,k) -> 
              Bigarray.Genarray.set goals_nodes [|index; side; i; j|] 1;
              Bigarray.Genarray.set goals_nodes [|index; side; i; k|] 1) !nodes
    | BON ->
        fun (writes,nodes) ->
          List.iter (fun (i,j,_) -> 
              let n = Bigarray.Genarray.get goals [|index; i; j|] in 
              Bigarray.Genarray.set goals [|index; i; j|] (n+1)
              ) !writes;
          List.iter (fun (_,j,k) -> 
              let n = Bigarray.Genarray.get goals_nodes [|index; j|] in 
              let _ = Bigarray.Genarray.set goals_nodes [|index; j|] (n+1) in
              if k = -1 then ()
              else 
                let m = Bigarray.Genarray.get goals_nodes [|index; k|] in 
                Bigarray.Genarray.set goals_nodes [|index;k|] (m+1) ) (!nodes)

  in
    let size,_ = term_node tm fmt mnn goal_cc  in
    let _ = Printf.printf "goal succeeded" in 
    let _ = 
        match which with 
          | Train -> 
            Bigarray.Array2.set data.train_labels index (label content) 1;
            Bigarray.Genarray.set meta [|index; 0|] len;
            Bigarray.Genarray.set meta [|index; 1|] size
          | Test -> 
            Bigarray.Array2.set data.test_labels index (label content) 1;
            Bigarray.Genarray.set meta [|index; 0|] len;
            Bigarray.Genarray.set meta [|index; 1|] size
          | _ -> 
            Bigarray.Genarray.set meta [|index; side; 0|] len;
            Bigarray.Genarray.set meta [|index; side; 1|] size
      in
    let ctx_cc l = 
          match fmt with
          | COMPACT -> 
            fun (writes,nodes) -> 
              List.iter (fun (i,j,k) -> Bigarray.Genarray.set contexts [|index; l; i; j|] (k)) !writes ;
              List.iter (fun (i,j,k) -> Bigarray.Genarray.set contexts_nodes [|index; l; i; j|] ( k)) !nodes 
          | OH -> 
            fun (writes,nodes) -> 
              List.iter (fun (i,j,k) -> Bigarray.Genarray.set contexts [|index; side; l; i; j|] ( k)) !writes ;
              List.iter (fun (i,j,k) -> 
                  Bigarray.Genarray.set contexts_nodes [|index; side; l; i; j|] 1;
                  Bigarray.Genarray.set contexts_nodes [|index; side; l; i; k|] 1) !nodes 
          | BON -> 
            fun (writes,nodes) -> 
              List.iter (fun (i,j,_) -> 
                  let n = Bigarray.Genarray.get contexts [|index; i; j|] in 
                  Bigarray.Genarray.set contexts [|index; i; j|] (n+1) 
                ) !writes ;
              List.iter (fun (_,j,k) -> 
                  let n = Bigarray.Genarray.get goals_nodes [|index; j|] in 
                  let _ = Bigarray.Genarray.set goals_nodes [|index;j|] (n+1) in
                  if k = -1 then ()
                  else 
                    let m = Bigarray.Genarray.get goals_nodes [|index; k|] in 
                    Bigarray.Genarray.set goals_nodes [|index;k|] (m+1) 
                        ) !nodes 
        in
      Some(
        List.iteri (fun l tm -> 
        let _ =  Printf.printf "ctx\n" in
        let sizei,_ = term_node tm fmt mnn (ctx_cc l) in
        match which with 
          | Train | Test -> Bigarray.Genarray.set meta [|index;(l + 1)|] sizei
          | _ -> Bigarray.Genarray.set meta [|index;side;(l + 1)|] sizei
          ) asl
        )
  with (Fail msg) -> let _ = Printf.printf "%s\n" msg in None
     | Not_found ->  let _ = Printf.printf "fdaa" in None


let gen_data (fmt : data_format) = 
  let _ = Random.init 0 in
  let num_succ = ref 0 in
  let dataref = ref (alloc fmt) in
  let n = (!dataref).nUM_TRAINING in 
  let m = (!dataref).nUM_TEST in 
  let seen = ref IntS.empty in
    while !num_succ <  n do 
      let i = Random.int 12576083 in
      if IntS.mem i !seen then () 
      else 
        let _ = seen := IntS.add i (!seen) in
        let _ =  Printf.printf "i: %d\n" i in 
        let _ =  Printf.printf "#succ: %d\n" (!num_succ) in 
        try
        let p = proof_at i in 
        match matrify !num_succ p Train fmt CL (-1) (dataref) with
        | Some () -> num_succ := !num_succ + 1
        | _ -> ()
        with Not_found -> ()
    done
  ;
    num_succ := 0;
    while !num_succ <  m do 
      let i = Random.int 12576083 in
      if IntS.mem i !seen then () 
      else 
        let _ = seen := IntS.add i (!seen) in
        let _ =  Printf.printf "i: %d\n" i in 
        let _ =  Printf.printf "#succ: %d\n" (!num_succ) in 
        try
        let p = proof_at i in 
        match matrify !num_succ p Test fmt CL (-1) dataref with
        | Some () -> num_succ := !num_succ + 1
        | _ -> ()
        with Not_found -> ()
    done
  ;
  write_arrays (!dataref) fmt

(* ------------------------------------------------------------------------- *)
(* Marshalling of proof to JSON parts.                                       *)
(* ------------------------------------------------------------------------- *)

let rec inst_string insts =
  match insts with
    [] -> ""
  | (t1,t2)::[] -> Printf.sprintf "[\"%s\", \"%s\"]"
                                  (term_string t2)
                                  (term_string t1)
  | (t1,t2)::tail -> Printf.sprintf "[\"%s\", \"%s\"], %s"
                                    (term_string t2)
                                    (term_string t1)
                                    (inst_string tail)

let rec instt_string insts =
  match insts with
    [] -> ""
  | (t1,t2)::[] -> Printf.sprintf "[\"%s\", \"%s\"]"
                                  (type_string t2)
                                  (type_string t1)
  | (t1,t2)::tail -> Printf.sprintf "[\"%s\", \"%s\"], %s"
                                    (type_string t2)
                                    (type_string t1)
                                    (instt_string tail)
(* ------------------------------------------------------------------------- *)
(* Marshalling of thm to JSON.                                               *)
(* ------------------------------------------------------------------------- *)

;;
exception Fail of string

let thm_string th =
  let asl,tm = dest_thm th in
  let rec asl_string asl =
    match asl with
      [] -> ""
    | tm::[] -> Printf.sprintf "\"%s\"" (term_string tm)
    | tm::tail -> Printf.sprintf "\"%s\", %s"
                                 (term_string tm)
                                 (asl_string tail)
  in Printf.sprintf "{\"hy\": [%s], \"cc\": \"%s\"}"
                    (asl_string asl)
                    (term_string tm)

let theorem_string proof =
  let Proof(idx,thm,content) = proof in
  Printf.sprintf "{\"id\": %d, \"th\": %s}"
                 idx
                 (thm_string thm);;

let proof_index proof =
  let Proof(idx,_,_) = proof in idx

let proof_content_string content =
  match content with
    Prefl(tm) -> Printf.sprintf "[\"REFL\", \"%s\"]"
                                (term_string tm)
  | Ptrans(p1,p2) -> Printf.sprintf "[\"TRANS\", %d, %d]"
                                    (proof_index p1)
                                    (proof_index p2)
  | Pmkcomb(p1,p2) -> Printf.sprintf "[\"MK_COMB\", %d, %d]"
                                     (proof_index p1)
                                     (proof_index p2)
  | Pabs(p1,tm) -> Printf.sprintf "[\"ABS\", %d, \"%s\"]"
                                  (proof_index p1)
                                  (term_string tm)
  | Pbeta(tm) -> Printf.sprintf "[\"BETA\", \"%s\"]"
                                (term_string tm)
  | Passume(tm) -> Printf.sprintf "[\"ASSUME\", \"%s\"]"
                                  (term_string tm)
  | Peqmp(p1,p2) -> Printf.sprintf "[\"EQ_MP\", %d, %d]"
                                   (proof_index p1)
                                   (proof_index p2)
  | Pdeduct(p1,p2) -> Printf.sprintf "[\"DEDUCT_ANTISYM_RULE\", %d, %d]"
                                     (proof_index p1)
                                     (proof_index p2)
  | Pinst(p1,insts) -> Printf.sprintf "[\"INST\", %d, [%s]]"
                                      (proof_index p1)
                                      (inst_string insts)
  | Pinstt(p1,insts) -> Printf.sprintf "[\"INST_TYPE\", %d, [%s]]"
                                       (proof_index p1)
                                       (instt_string insts)
  | Paxiom(tm) -> Printf.sprintf "[\"AXIOM\", \"%s\"]"
                                 (term_string tm)
  | Pdef(tm,name,ty) -> Printf.sprintf "[\"DEFINITION\", \"%s\", \"%s\"]"
                                       (term_string tm)
                                       (String.escaped name)
  | Pdeft(p1,tm,name,ty) -> Printf.sprintf
                              "[\"TYPE_DEFINITION\", %d, \"%s\", \"%s\"]"
                              (proof_index p1)
                              (term_string tm)
                              (String.escaped name)

let proof_cut_string content =
  match content with
    Prefl(tm) -> Printf.sprintf "[\"REFL\", \"%s\"]"
                                (term_string tm)
  | Ptrans(p1,p2) -> Printf.sprintf "[\"TRANS\", %s, %s]"
                                    (theorem_string p1)
                                    (theorem_string p2)
  | Pmkcomb(p1,p2) -> Printf.sprintf "[\"MK_COMB\", %s, %s]"
                                     (theorem_string p1)
                                     (theorem_string p2)
  | Pabs(p1,tm) -> Printf.sprintf "[\"ABS\", %s, \"%s\"]"
                                  (theorem_string p1)
                                  (term_string tm)
  | Pbeta(tm) -> Printf.sprintf "[\"BETA\", \"%s\"]"
                                (term_string tm)
  | Passume(tm) -> Printf.sprintf "[\"ASSUME\", \"%s\"]"
                                  (term_string tm)
  | Peqmp(p1,p2) -> Printf.sprintf "[\"EQ_MP\", %s, %s]"
                                   (theorem_string p1)
                                   (theorem_string p2)
  | Pdeduct(p1,p2) -> Printf.sprintf "[\"DEDUCT_ANTISYM_RULE\", %s, %s]"
                                     (theorem_string p1)
                                     (theorem_string p2)
  | Pinst(p1,insts) -> Printf.sprintf "[\"INST\", %s, [%s]]"
                                      (theorem_string p1)
                                      (inst_string insts)
  | Pinstt(p1,insts) -> Printf.sprintf "[\"INST_TYPE\", %s, [%s]]"
                                       (theorem_string p1)
                                       (instt_string insts)
  | Paxiom(tm) -> Printf.sprintf "[\"AXIOM\", \"%s\"]"
                                 (term_string tm)
  | Pdef(tm,name,ty) -> Printf.sprintf "[\"DEFINITION\", \"%s\", \"%s\"]"
                                       (term_string tm)
                                       (String.escaped name)
  | Pdeft(p1,tm,name,ty) -> Printf.sprintf
                              "[\"TYPE_DEFINITION\", %s, \"%s\", \"%s\"]"
                              (theorem_string p1)
                              (term_string tm)
                              (String.escaped name)


let proof_string proof =
  let Proof(idx,thm,content) = proof in
  Printf.sprintf "{\"id\": %d, \"pr\": %s}"
                 idx
                 (proof_content_string content);;

let cut_string proof =
  let Proof(idx,thm,content) = proof in
  Printf.sprintf "{\"id\": %d, \"thm\": %s \"pr\": %s}"
                 idx
                 (thm_string thm)
                 (proof_cut_string content);;

let proof_range i j = 
  List.init (j -i + 1) (fun idx -> proof_at(idx + i)) ;; 

(* Search *)

(*
exception Search of string

let which_const c = 
  let d = definitions () in 
  List.reduce (fun thm b -> 
      match b with 
      | Some def => SOME def
      | None => 
        begin
        match concl thm with
        | Comb(Comb(Const ("=", `:bool->bool->bool`), Const(name,_)), def) -> 
          if c = name then Some def else None
        | _ -> None
        end) d

let prove conj = 
  match conj with
  | Var(v,ty) -> raise Search "encountered variable"
  | Const(c,ty) -> 
    let def = which_const c in 
    search def
  | Comb(t1,t2) -> 
    
  | Abs(t1,t2) -> 

*)



(* ------------------------------------------------------------------------- *)
(* Proofs and Theorems trace dumping.                                        *)
(* ------------------------------------------------------------------------- *)

let dump_proofs filename =
  let foutc = open_out filename in
  (do_list (fun p -> Printf.fprintf foutc
                                    "%s\n"
                                    (proof_string p)) (proofs());
   flush foutc;
   close_out foutc)
;;

let dump_theorems filename =
  let foutc = open_out filename in
  (do_list (fun p -> Printf.fprintf foutc
                     "%s\n"
                     (theorem_string p)) (proofs());
   flush foutc;
   close_out foutc)
;;

let dump_cuts filename =
  let foutc = open_out filename in
  (do_list (fun p -> Printf.fprintf foutc
                                    "%s\n"
                                    (cut_string p)) (proofs());
   flush foutc;
   close_out foutc)
;;

let dump_constants filename =
  let foutc = open_out filename in
  (do_list (fun (p,_) -> Printf.fprintf foutc
                                    "%s\n" p)
                                     (constants());
   flush foutc;
   close_out foutc)
;;
(* ------------------------------------------------------------------------- *)
(* Theorem names extraction (inspired by HolStep, but non-destructive).      *)
(* ------------------------------------------------------------------------- *)

let pROVE_1_RE = Str.regexp (String.concat "" (
  "\\(let\\|and\\)[ \n\t]*"::
  "\\([a-zA-Z0-9_-]+\\)[ \n\t]*"::
  "=[ \n\t]*"::
  "\\(prove\\|"::
  "prove_by_refinement\\|"::
  "new_definition\\|"::
  "new_basic_definition\\|"::
  "new_axiom\\|"::
  "new_infix_definition\\|"::
  "INT_OF_REAL_THM\\|"::
  "define_finite_type\\|"::
  "TAUT\\|"::
  "INT_ARITH\\|"::
  "new_recursive_definition\\)"::
  []
))

let pROVE_2_RE = Str.regexp (String.concat "" (
  "\\(let\\|and\\)[ \n\t]*"::
  "\\([a-zA-Z0-9_-]+\\)[ \n\t]*,[ \n\t]*"::
  "\\([a-zA-Z0-9_-]+\\)[ \n\t]*"::
  "=[ \n\t]*"::
  "\\(define_type\\|"::
  "(CONJ_PAIR o prove)\\)"::
  []
))

let pROVE_3_RE = Str.regexp (String.concat "" (
  "\\(let\\|and\\)[ \n\t]*"::
  "\\([a-zA-Z0-9_-]+\\)[ \n\t]*,[ \n\t]*"::
  "\\([a-zA-Z0-9_-]+\\)[ \n\t]*,[ \n\t]*"::
  "\\([a-zA-Z0-9_-]+\\)[ \n\t]*"::
  "=[ \n\t]*"::
  "\\(new_inductive_definition\\)"::
  []
))

let rec take n l = 
  match n with 
  | 0 -> []
  | n -> (List.hd l) :: take (n-1) (List.tl l) 

let source_files() =
  let select str = Str.string_match (Str.regexp ".*\\.[hm]l$") str 0 in
  let rec walk acc = function
  | [] -> (acc)
  | dir::tail ->
      let contents = Array.to_list (Sys.readdir dir) in
      let contents = List.rev_map (Filename.concat dir) contents in
      let dirs, files =
        List.fold_left (fun (dirs,files) f ->
                          match Sys.is_directory f with
                          | false -> (dirs, f::files)  (* Regular file *)
                          | true -> (f::dirs, files)  (* Directory *)
        ) ([],[]) contents in
      let matched = List.filter (select) files in
        walk (matched @ acc) (dirs @ tail)
  in 
  take 2 (walk [] [Sys.getcwd()])
;;

let load_file f =
  let ic = open_in f in
  let n = in_channel_length ic in
  let s = Bytes.create n in
  really_input ic s 0 n;
  close_in ic;
  (s)

let extract_prove_1 f =
  let content = Bytes.to_string(load_file f) in
  let rec search acc start =
    try
      let _ = Str.search_forward pROVE_1_RE content start in
      let matches = (Str.matched_group 2 content)::[] in
      search (matches @ acc) (Str.match_end())
    with e -> (acc)
  in search [] 0
;;

let extract_prove_2 f =
  let content = Bytes.to_string(load_file f) in
  let rec search acc start =
    try
      let _ = Str.search_forward pROVE_2_RE content start in
      let matches = (Str.matched_group 2 content)::
                    (Str.matched_group 3 content)::
                    [] in
      search (matches @ acc) (Str.match_end())
    with e -> (acc)
  in search [] 0
;;

let extract_prove_3 f =
  let content = Bytes.to_string(load_file f) in
  let rec search acc start =
    try
      let _ = Str.search_forward pROVE_3_RE content start in
      let matches = (Str.matched_group 2 content)::
                    (Str.matched_group 3 content)::
                    (Str.matched_group 4 content)::
                    [] in
      search (matches @ acc) (Str.match_end())
    with e -> (acc)
  in search [] 0
;;

(* ------------------------------------------------------------------------- *)
(* Names trace dumping (:see_no_evil)                                        *)
(* ------------------------------------------------------------------------- *)

let eval code =
  let as_buf = Lexing.from_string code in
  let parsed = !Toploop.parse_toplevel_phrase as_buf in
  ignore (Toploop.execute_phrase true Format.std_formatter parsed)

let _CODE_GEN name = Printf.sprintf
                       "_IDX := proof_index (proof_of %s);;"
                       name
let _IDX = ref (0)

let dump_names filename =
  let foutc = open_out filename in
  let acc = ref ([]) in
  (do_list (fun f -> acc := !acc @
                     (extract_prove_1 f) @
                     (extract_prove_2 f) @
                     (extract_prove_3 f)) (source_files());
   acc := List.sort_uniq compare !acc;
   do_list (fun name ->
               try
                 eval (_CODE_GEN name);
                 Printf.fprintf foutc
                                "{\"id\": %d, \"nm\": \"%s\"}\n"
                                !_IDX name;
               with _ -> ()
            ) (!acc);
   flush foutc;
   close_out foutc)
;;
