; RUN: llc < %s | FileCheck %s
; RUN: lli < %s

; CHECK-LABEL: mul256
define i256 @mul256(i256 %a, i256 %b) nounwind readnone {
  %p = mul i256 %a, %b
  ret i256 %p
}

; CHECK-LABEL: mul512
define i512 @mul512(i512 %a, i512 %b) nounwind readnone {
  %p = mul i512 %a, %b
  ret i512 %p
}

; CHECK-LABEL: mul1024
define i1024 @mul1024(i1024 %a, i1024 %b) nounwind readnone {
  %p = mul i1024 %a, %b
  ret i1024 %p
}

define i1 @main() nounwind {
  %p = call i256 @mul256(i256 1111111111111111111111111111111111111, i256 22222222222222222222222222222222222)
  %r = icmp ne i256 %p, 24691358024691358024691358024691357775308641975308641975308641975308642
  ret i1 %r
}

