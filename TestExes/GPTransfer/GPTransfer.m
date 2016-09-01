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
(*lk[muy_,sigy_,x_,l_] := Sqrt[l^2/(l^2+sigy)]Exp[-((muy-x)^2/(2(l^2+sigy)))]*)
(*lkVec[mu_,sig_,x_,l_] := Module[{nGps=Dimensions[x][[1]],nSmpls=Dimensions[x][[2]]},*)
(*Table[EyKSE[mu[[gp]],sig[[gp,gp]],x[[gp,smpl]],l[[gp]]],{gp,1,nGps},{smpl,1,nSmpls}]]*)
(**)
(**)
(*dlkdMuy:=Derivative[1,0,0,0][lk]*)
(*dlkdSigy:=Derivative[0,1,0,0][lk]*)
(*dlkdX :=Derivative[0,0,1,0][lk]*)
(*dlkdL:=Derivative[0,0,0,1][lk]*)


(* ::Input:: *)
(*dlkVecdL[mu_,sig_,x_,l_,dldLkVec_] :=Module[{nGps=Dimensions[x][[1]],nSmpls=Dimensions[x][[2]],nOut=Dimensions[dldLkVec][[1]]},*)
(*Table[Sum[dldLkVec[[out,j,a]]dlkdL[mu[[j]],sig[[j,j]],x[[j,a]],l[[j]]],{a,1,nSmpls}],{out,1,nOut},{j,1,nGps}]];*)
(*dlkVecdMu[mu_,sig_,x_,l_,dldLkVec_] :=Module[{nGps=Dimensions[x][[1]],nSmpls=Dimensions[x][[2]],nOut=Dimensions[dldLkVec][[1]]},*)
(*Table[Sum[dldLkVec[[out,j,a]]dlkdMuy[mu[[j]],sig[[j,j]],x[[j,a]],l[[j]]],{a,1,nSmpls}],{out,1,nOut},{j,1,nGps}]];*)
(*dlkVecdSig[mu_,sig_,x_,l_,dldLkVec_] :=Module[{nGps=Dimensions[x][[1]],nSmpls=Dimensions[x][[2]],nOut=Dimensions[dldLkVec][[1]]},*)
(*Table[Sum[dldLkVec[[out,j,a]]dlkdSigy[mu[[j]],sig[[j,j]],x[[j,a]],l[[j]]],{a,1,nSmpls}],{out,1,nOut},{j,1,nGps}]];*)
(*dlkVecdX[mu_,sig_,x_,l_,dldLkVec_] :=Module[{nGps=Dimensions[x][[1]],nSmpls=Dimensions[x][[2]],nOut=Dimensions[dldLkVec][[1]]},*)
(*Table[dldLkVec[[out,j,c]]dlkdX[mu[[j]],sig[[j,j]],x[[j,c]],l[[j]]],{out,1,nOut},{j,1,nGps},{c,1,nSmpls}]];*)
(**)
(**)
(*dlkdX*)


(* ::Input:: *)
(**)


(* ::Input:: *)
(**)


(* ::Input:: *)
(**)


(* ::Input:: *)
(**)


(* ::Input:: *)
(**)


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


(* ::Input:: *)
(**)
(*KSE*)


(* ::Input:: *)
(*KSEMat[trnSmpls,ls][[1]]//MatrixForm*)


(* ::Input:: *)
(*mus = {2.,1.7}*)


(* ::Input:: *)
(*sigs = {{.5,.42},{.2,.4}}*)


(* ::Input:: *)
(**)
(*lk[2.,0.5,1.7,2.]*)


(* ::Input:: *)
(*lkVec[mus,sigs,trnSmpls,ls]*)


(* ::Input:: *)
(*dlkdMuy[2.,0.5,1.7,2.]*)


(* ::Input:: *)
(*dlkdX[2.,0.5,1.7,2.]*)


(* ::InheritFromParent:: *)
(**)
(*dlkdL[2.,0.5,1.7,2.]*)


(* ::Input:: *)
(*dlkVecdL[mus,sigs,trnSmpls,ls,dlDKSEval]*)


(* ::Input:: *)
(**)
(*dlkVecdMu[mus,sigs,trnSmpls,ls,dlDKSEval]*)


(* ::Input:: *)
(**)
(*dlkVecdSig[mus,sigs,trnSmpls,ls,dlDKSEval]*)


(* ::Input:: *)
(**)
(*dlkVecdX[mus,sigs,trnSmpls,ls,dlDKSEval]*)
