; RUN: llc < %s | FileCheck %s

; CHECK-LABEL: extractelement_index_i256_1:
define i8 @extractelement_index_i256_1(<32 x i8> %a) nounwind {
  %b = extractelement <32 x i8> %a, i256 1
  ret i8 %b
}

; CHECK-LABEL: extractelement_index_i256_2:
define i8 @extractelement_index_i256_2(<32 x i8> %a) nounwind {
  %b = extractelement <32 x i8> %a, i256 60
  ret i8 %b
}

; CHECK-LABEL: extractelement_index_i256_3:
define i8 @extractelement_index_i256_3(<32 x i8> %a, i256 %i) nounwind {
  %b = extractelement <32 x i8> %a, i256 %i
  ret i8 %b
}

