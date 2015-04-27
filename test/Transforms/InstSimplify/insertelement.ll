; RUN: opt < %s -instsimplify -S | FileCheck %s

; CHECK-LABEL: @insertelement_undef2
define <4 x i64> @insertelement_undef2(<4 x i64> %in) {
  %vec = insertelement <4 x i64> %in, i64 -5, i32 4
  ; CHECK: ret <4 x i64> undef
  ret <4 x i64> %vec
}