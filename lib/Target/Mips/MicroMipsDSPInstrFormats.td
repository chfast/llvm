//===-- MicroMipsDSPInstrFormats.td - Instruction Formats --*- tablegen -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

class MMDSPInst<string opstr = "">
    : MipsInst<(outs), (ins), "", [], NoItinerary, FrmOther>, PredicateControl {
  let InsnPredicates = [HasDSP];
  let AdditionalPredicates = [InMicroMips];
  string BaseOpcode = opstr;
  string Arch = "mmdsp";
  let DecoderNamespace = "MicroMips";
}

class POOL32A_3R_FMT<string opstr, bits<11> op> : MMDSPInst<opstr> {
  bits<5> rd;
  bits<5> rs;
  bits<5> rt;

  let Inst{31-26} = 0b000000;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{15-11} = rd;
  let Inst{10-0}  = op;
}

class POOL32A_2R_FMT<string opstr, bits<10> op> : MMDSPInst<opstr> {
  bits<5> rt;
  bits<5> rs;

  let Inst{31-26} = 0b000000;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{15-6}  = op;
  let Inst{5-0}   = 0b111100;
}

class POOL32A_2RAC_FMT<string opstr, bits<8> op> : MMDSPInst<opstr> {
  bits<5> rt;
  bits<5> rs;
  bits<2> ac;

  let Inst{31-26} = 0b000000;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{15-14} = ac;
  let Inst{13-6}  = op;
  let Inst{5-0}   = 0b111100;
}

class POOL32A_3RB0_FMT<string opstr, bits<10> op> : MMDSPInst<opstr> {
  bits<5> rd;
  bits<5> rs;
  bits<5> rt;

  let Inst{31-26} = 0b000000;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{15-11} = rd;
  let Inst{10}    = 0b0;
  let Inst{9-0}   = op;
}

class POOL32A_2RSA4_FMT<string opstr, bits<12> op> : MMDSPInst<opstr> {
  bits<5> rt;
  bits<5> rs;
  bits<4> sa;

  let Inst{31-26} = 0b000000;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{15-12} = sa;
  let Inst{11-0}  = op;
}

class POOL32A_2RSA3_FMT<string opstr, bits<7> op> : MMDSPInst<opstr> {
  bits<5> rt;
  bits<5> rs;
  bits<3> sa;

  let Inst{31-26} = 0b000000;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{15-13} = sa;
  let Inst{12-6}  = op;
  let Inst{5-0}   = 0b111100;
}

class POOL32A_2RSA5B0_FMT<string opstr, bits<10> op> : MMDSPInst<opstr> {
  bits<5> rt;
  bits<5> rs;
  bits<5> sa;

  let Inst{31-26} = 0b000000;
  let Inst{25-21} = rt;
  let Inst{20-16} = rs;
  let Inst{15-11} = sa;
  let Inst{10}    = 0b0;
  let Inst{9-0}   = op;
}
