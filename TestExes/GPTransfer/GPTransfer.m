(* ::Package:: *)

(* ::Input::Initialization:: *)
TestFunc[x_]=10x;
TestFunc2[x_]=3x;


(* ::Input::Initialization:: *)
KSE[x1_,x2_,l_]:=Exp[-((x1-x2)^2/(2l^2))];
KSEMat[x_,l_]:=Module[{nGps=Dimensions[x][[1]],nSmpls=Dimensions[x][[2]]},
Table[KSE[x[[gp,smpl1]],x[[gp,smpl2]],l[[gp]]],{gp,1,nGps},{smpl1,1,nSmpls},{smpl2,1,nSmpls}]]
KSEMat[x_,l_]:=Module[{nGps=Dimensions[x][[1]],nSmpls=Dimensions[x][[2]]},
Table[KSE[x[[gp,smpl1]],x[[gp,smpl2]],l[[gp]]],{gp,1,nGps},{smpl1,1,nSmpls},{smpl2,1,nSmpls}]]


(* ::Input::Initialization:: *)
dKSEdX1:=Derivative[1,0,0][KSE];
dKSEdX2:=Derivative[0,1,0][KSE];
dKSEdL:=Derivative[0,0,1][KSE];
dKSEMatdX[x_,l_,dLdKSE_]:=Module[{nGps=Dimensions[x][[1]],nSmpls=Dimensions[x][[2]],nOut=Dimensions[dLdKSE][[1]]},
Table[Sum[dLdKSE[[out,h,c,b]] dKSEdX1[x[[h,c]],x[[h,b]],l[[h]]],{b,1,nSmpls}]+
Sum[dLdKSE[[out,h,a,c]] dKSEdX2[x[[h,a]],x[[h,c]],l[[h]]],{a,1,nSmpls}],
{out,1,nOut},{h,1,nGps},{c,1,nSmpls}]];
dKSEMatdL[x_,l_,dLdKSE_]:=Module[{nGps=Dimensions[x][[1]],nSmpls=Dimensions[x][[2]],nOut=Dimensions[dLdKSE][[1]]},
Table[Sum[dLdKSE[[out,j,a,b]] dKSEdL[x[[j,a]],x[[j,b]],l[[j]]],{a,1,nSmpls},{b,1,nSmpls}],{out,1,nOut},{j,1,nGps}]];


(* ::Input:: *)
(**)


(* ::Input:: *)
(*dKSEdX1[1.,2.,3.]*)


(* ::Input:: *)
(*trnSmpls={{1.0,2.0,3.3},{5.0,6.0,6.5}};*)
(*ls={2.0,4.0};*)
(*KSEMat[trnSmpls,ls][[2]]//MatrixForm*)


(* ::Input:: *)
(*dlDKSEval={*)
(*{*)
(*{{1.,2.,3.},{4.,5.,6.}, {7.,8.,9}},*)
(*{{1.,2.,3.},{4.,5.,6.}, {7.,8.,9}}*)
(*}};*)
(*Dimensions[dlDKSEval]*)


(* ::Input:: *)
(*dKSEMatdX[trnSmpls,ls,dlDKSEval]*)


(* ::Input:: *)
(*D[KSE[x1,x2,l],x1]*)


(* ::Input:: *)
(*dKSEdX1[1.0,2.0,3.0]*)


(* ::Input:: *)
(*KSE*)


(* ::Input:: *)
(*Derivative[1,0,0][KSE][1,2,3]*)
