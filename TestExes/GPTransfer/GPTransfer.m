(* ::Package:: *)

(* ::Input::Initialization:: *)
TestFunc[x_]=10x;
TestFunc2[x_]=3x;


(* ::Input::Initialization:: *)
KSE[x1_,x2_,l_]:=Exp[-((x1-x2)^2/(2l^2))];
KSEMat[x_,l_]:=Module[{nGps=Dimensions[x][[1]],nSmpls=Dimensions[x][[2]]},
Table[KSE[x[[gp,smpl1]],x[[gp,smpl2]],l[[gp]]],{gp,1,nGps},{smpl1,1,nSmpls},{smpl2,1,nSmpls}]]


(* ::Input:: *)
(**)


(* ::Input:: *)
(*trnSmpls={{1.0,2.0,3.3},{5.0,6.0,6.5}};*)
(*ls={2.0,4.0};*)
(*KSEMat[trnSmpls,ls][[2]]//MatrixForm*)
