# 7
Data points look like X_i,y_i = (Gamma |- C, rule) #SAMPLES = (20k training, 2k test)
All trees have a maximum number of 100 nodes, represented as a 100 x 2 matrix

  *_ctxs.npz

The trees in Gamma. Dim = #SAMPLES x 5 x 100 x 2

  *_goals.npz  

The tree C. Dim = #SAMPLES x 100 x 2

  *_n.npz 

The labels of the nodes in Gamma (resp. C). Dim = #SAMPLES x 5 x 100 x 6 (resp. #SAMPLES x 100 x 6). 
Last dim layed out as (VAR, TYVAR, TYAPP, CONST, COMB, ABS) in [128] x [128] x [297] x [297] x [1] x [1],
since there are 128 ascii characters and 297 hol-light constants.

  *_meta.npz

Data about the actual sizes input. Dim = #SAMPLES x 2. Last dim layouted as (|Gamma|, |C|, proof_index).
proof_index is the index of the theorem. 

TODO: NEED TO INCLUDE sizes of each A in Gamma: |A|.

  *_labels.npz

One hot embedding of the rules. Dim = #SAMPLES x 13. Indices:
 
   Prefl(tm) -> 0
   Ptrans(p1,p2) -> 1
   Pmkcomb(p1,p2) -> 2 
   Pabs(p1,tm) -> 3
   Pbeta(tm) -> 4
   Passume(tm) -> 5
   Peqmp(p1,p2) -> 6 
   Pdeduct(p1,p2) -> 7
   Pinst(p1,insts) -> 8
   Pinstt(p1,insts) -> 9
   Paxiom(tm) -> 10
   Pdef(tm,name,ty) -> 11
   Pdeft(p1,tm,name,ty) -> 12


