//===- AggressiveInstCombine.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the aggressive expression pattern combiner classes.
// Currently, it handles expression patterns for:
//  * Truncate instruction
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/AggressiveInstCombine/AggressiveInstCombine.h"
#include "AggressiveInstCombineInternal.h"
#include "llvm-c/Initialization.h"
#include "llvm-c/Transforms/AggressiveInstCombine.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Analysis/BasicAliasAnalysis.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Pass.h"
#include "llvm/Transforms/Utils/Local.h"
using namespace llvm;
using namespace PatternMatch;

#define DEBUG_TYPE "aggressive-instcombine"

namespace {
/// Contains expression pattern combiner logic.
/// This class provides both the logic to combine expression patterns and
/// combine them. It differs from InstCombiner class in that each pattern
/// combiner runs only once as opposed to InstCombine's multi-iteration,
/// which allows pattern combiner to have higher complexity than the O(1)
/// required by the instruction combiner.
class AggressiveInstCombinerLegacyPass : public FunctionPass {
public:
  static char ID; // Pass identification, replacement for typeid

  AggressiveInstCombinerLegacyPass() : FunctionPass(ID) {
    initializeAggressiveInstCombinerLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  /// Run all expression pattern optimizations on the given /p F function.
  ///
  /// \param F function to optimize.
  /// \returns true if the IR is changed.
  bool runOnFunction(Function &F) override;
};
} // namespace

/// Match a pattern for a bitwise rotate operation that partially guards
/// against undefined behavior by branching around the rotation when the shift
/// amount is 0.
static bool foldGuardedRotateToFunnelShift(Instruction &I) {
  if (I.getOpcode() != Instruction::PHI || I.getNumOperands() != 2)
    return false;

  // As with the one-use checks below, this is not strictly necessary, but we
  // are being cautious to avoid potential perf regressions on targets that
  // do not actually have a rotate instruction (where the funnel shift would be
  // expanded back into math/shift/logic ops).
  if (!isPowerOf2_32(I.getType()->getScalarSizeInBits()))
    return false;

  // Match V to funnel shift left/right and capture the source operand and
  // shift amount in X and Y.
  auto matchRotate = [](Value *V, Value *&X, Value *&Y) {
    Value *L0, *L1, *R0, *R1;
    unsigned Width = V->getType()->getScalarSizeInBits();
    auto Sub = m_Sub(m_SpecificInt(Width), m_Value(R1));

    // rotate_left(X, Y) == (X << Y) | (X >> (Width - Y))
    auto RotL = m_OneUse(
        m_c_Or(m_Shl(m_Value(L0), m_Value(L1)), m_LShr(m_Value(R0), Sub)));
    if (RotL.match(V) && L0 == R0 && L1 == R1) {
      X = L0;
      Y = L1;
      return Intrinsic::fshl;
    }

    // rotate_right(X, Y) == (X >> Y) | (X << (Width - Y))
    auto RotR = m_OneUse(
        m_c_Or(m_LShr(m_Value(L0), m_Value(L1)), m_Shl(m_Value(R0), Sub)));
    if (RotR.match(V) && L0 == R0 && L1 == R1) {
      X = L0;
      Y = L1;
      return Intrinsic::fshr;
    }

    return Intrinsic::not_intrinsic;
  };

  // One phi operand must be a rotate operation, and the other phi operand must
  // be the source value of that rotate operation:
  // phi [ rotate(RotSrc, RotAmt), RotBB ], [ RotSrc, GuardBB ]
  PHINode &Phi = cast<PHINode>(I);
  Value *P0 = Phi.getOperand(0), *P1 = Phi.getOperand(1);
  Value *RotSrc, *RotAmt;
  Intrinsic::ID IID = matchRotate(P0, RotSrc, RotAmt);
  if (IID == Intrinsic::not_intrinsic || RotSrc != P1) {
    IID = matchRotate(P1, RotSrc, RotAmt);
    if (IID == Intrinsic::not_intrinsic || RotSrc != P0)
      return false;
    assert((IID == Intrinsic::fshl || IID == Intrinsic::fshr) &&
           "Pattern must match funnel shift left or right");
  }

  // The incoming block with our source operand must be the "guard" block.
  // That must contain a cmp+branch to avoid the rotate when the shift amount
  // is equal to 0. The other incoming block is the block with the rotate.
  BasicBlock *GuardBB = Phi.getIncomingBlock(RotSrc == P1);
  BasicBlock *RotBB = Phi.getIncomingBlock(RotSrc != P1);
  Instruction *TermI = GuardBB->getTerminator();
  BasicBlock *TrueBB, *FalseBB;
  ICmpInst::Predicate Pred;
  if (!match(TermI, m_Br(m_ICmp(Pred, m_Specific(RotAmt), m_ZeroInt()), TrueBB,
                         FalseBB)))
    return false;

  BasicBlock *PhiBB = Phi.getParent();
  if (Pred != CmpInst::ICMP_EQ || TrueBB != PhiBB || FalseBB != RotBB)
    return false;

  // We matched a variation of this IR pattern:
  // GuardBB:
  //   %cmp = icmp eq i32 %RotAmt, 0
  //   br i1 %cmp, label %PhiBB, label %RotBB
  // RotBB:
  //   %sub = sub i32 32, %RotAmt
  //   %shr = lshr i32 %X, %sub
  //   %shl = shl i32 %X, %RotAmt
  //   %rot = or i32 %shr, %shl
  //   br label %PhiBB
  // PhiBB:
  //   %cond = phi i32 [ %rot, %RotBB ], [ %X, %GuardBB ]
  // -->
  // llvm.fshl.i32(i32 %X, i32 %RotAmt)
  IRBuilder<> Builder(PhiBB, PhiBB->getFirstInsertionPt());
  Function *F = Intrinsic::getDeclaration(Phi.getModule(), IID, Phi.getType());
  Phi.replaceAllUsesWith(Builder.CreateCall(F, {RotSrc, RotSrc, RotAmt}));
  return true;
}

/// This is used by foldAnyOrAllBitsSet() to capture a source value (Root) and
/// the bit indexes (Mask) needed by a masked compare. If we're matching a chain
/// of 'and' ops, then we also need to capture the fact that we saw an
/// "and X, 1", so that's an extra return value for that case.
struct MaskOps {
  Value *Root;
  APInt Mask;
  bool MatchAndChain;
  bool FoundAnd1;

  MaskOps(unsigned BitWidth, bool MatchAnds)
      : Root(nullptr), Mask(APInt::getNullValue(BitWidth)),
        MatchAndChain(MatchAnds), FoundAnd1(false) {}
};

/// This is a recursive helper for foldAnyOrAllBitsSet() that walks through a
/// chain of 'and' or 'or' instructions looking for shift ops of a common source
/// value. Examples:
///   or (or (or X, (X >> 3)), (X >> 5)), (X >> 8)
/// returns { X, 0x129 }
///   and (and (X >> 1), 1), (X >> 4)
/// returns { X, 0x12 }
static bool matchAndOrChain(Value *V, MaskOps &MOps) {
  Value *Op0, *Op1;
  if (MOps.MatchAndChain) {
    // Recurse through a chain of 'and' operands. This requires an extra check
    // vs. the 'or' matcher: we must find an "and X, 1" instruction somewhere
    // in the chain to know that all of the high bits are cleared.
    if (match(V, m_And(m_Value(Op0), m_One()))) {
      MOps.FoundAnd1 = true;
      return matchAndOrChain(Op0, MOps);
    }
    if (match(V, m_And(m_Value(Op0), m_Value(Op1))))
      return matchAndOrChain(Op0, MOps) && matchAndOrChain(Op1, MOps);
  } else {
    // Recurse through a chain of 'or' operands.
    if (match(V, m_Or(m_Value(Op0), m_Value(Op1))))
      return matchAndOrChain(Op0, MOps) && matchAndOrChain(Op1, MOps);
  }

  // We need a shift-right or a bare value representing a compare of bit 0 of
  // the original source operand.
  Value *Candidate;
  uint64_t BitIndex = 0;
  if (!match(V, m_LShr(m_Value(Candidate), m_ConstantInt(BitIndex))))
    Candidate = V;

  // Initialize result source operand.
  if (!MOps.Root)
    MOps.Root = Candidate;

  // The shift constant is out-of-range? This code hasn't been simplified.
  if (BitIndex >= MOps.Mask.getBitWidth())
    return false;

  // Fill in the mask bit derived from the shift constant.
  MOps.Mask.setBit(BitIndex);
  return MOps.Root == Candidate;
}

/// Match patterns that correspond to "any-bits-set" and "all-bits-set".
/// These will include a chain of 'or' or 'and'-shifted bits from a
/// common source value:
/// and (or  (lshr X, C), ...), 1 --> (X & CMask) != 0
/// and (and (lshr X, C), ...), 1 --> (X & CMask) == CMask
/// Note: "any-bits-clear" and "all-bits-clear" are variations of these patterns
/// that differ only with a final 'not' of the result. We expect that final
/// 'not' to be folded with the compare that we create here (invert predicate).
static bool foldAnyOrAllBitsSet(Instruction &I) {
  // The 'any-bits-set' ('or' chain) pattern is simpler to match because the
  // final "and X, 1" instruction must be the final op in the sequence.
  bool MatchAllBitsSet;
  if (match(&I, m_c_And(m_OneUse(m_And(m_Value(), m_Value())), m_Value())))
    MatchAllBitsSet = true;
  else if (match(&I, m_And(m_OneUse(m_Or(m_Value(), m_Value())), m_One())))
    MatchAllBitsSet = false;
  else
    return false;

  MaskOps MOps(I.getType()->getScalarSizeInBits(), MatchAllBitsSet);
  if (MatchAllBitsSet) {
    if (!matchAndOrChain(cast<BinaryOperator>(&I), MOps) || !MOps.FoundAnd1)
      return false;
  } else {
    if (!matchAndOrChain(cast<BinaryOperator>(&I)->getOperand(0), MOps))
      return false;
  }

  // The pattern was found. Create a masked compare that replaces all of the
  // shift and logic ops.
  IRBuilder<> Builder(&I);
  Constant *Mask = ConstantInt::get(I.getType(), MOps.Mask);
  Value *And = Builder.CreateAnd(MOps.Root, Mask);
  Value *Cmp = MatchAllBitsSet ? Builder.CreateICmpEQ(And, Mask)
                               : Builder.CreateIsNotNull(And);
  Value *Zext = Builder.CreateZExt(Cmp, I.getType());
  I.replaceAllUsesWith(Zext);
  return true;
}

/// Finds the first instruction after both A and B.
/// A and B are assumed to be either Instruction or Argument.
static Instruction *getInstructionAfter(Value *A, Value *B, DominatorTree &DT) {
  // TODO: Is there better way to achieve that?
  Instruction *I = nullptr;

  if (auto AI = dyn_cast<Instruction>(A))
    I = AI->getNextNode();
  else // If Argument use the first instruction in the entry block.
    I = &cast<Argument>(A)->getParent()->front().front();

  auto BI = dyn_cast<Instruction>(B);
  if (BI && DT.dominates(I, BI))
    I = BI->getNextNode(); // After B.

  return I;
}

/// Tries to find the full multiplication instructions pattern:
/// mul(zext(X), zext(Y)).
static Value *findFullMul(Value *X, Value *Y) {
  auto *FullTy = IntegerType::get(X->getContext(),
                                  X->getType()->getPrimitiveSizeInBits() * 2);
  for (const auto U : X->users()) {
    if (U->getType() == FullTy && match(U, m_ZExt(m_Specific(X)))) {
      for (const auto V : U->users()) {
        if (match(V, m_c_Mul(m_Specific(U), m_ZExt(m_Specific(Y)))))
          return V;
      }
    }
  }
  return nullptr;
}

/// Tries to find instruction mul(X, Y).
static Value *findLowMul(Value *X, Value *Y) {
  for (const auto U : X->users()) {
    if (match(U, m_c_Mul(m_Specific(X), m_Specific(Y))))
      return U;
  }
  return nullptr;
}

/// Tries to find a mul with X, Y as arguments. Creates a new one if not found.
static Value *findOrCreateLowMul(Instruction &I, Value *X, Value *Y,
                                 DominatorTree &DT) {
  if (auto *Mul = findLowMul(X, Y))
    return Mul;

  if (auto *FullMul = findFullMul(X, Y)) {
    IRBuilder<> Builder{&I};
    return Builder.CreateTrunc(FullMul, X->getType(), "fullmul.lo");
  }

  // Create the full multiplication instruction and place it just after its
  // operands. This position is the higher possible so will be safe to be used
  // as a replacement for all future matched patterns.
  IRBuilder<> Builder{getInstructionAfter(X, Y, DT)};
  return Builder.CreateMul(X, Y, "mul");
}

/// Tries to find the full mul with X, Y as arguments. If not found it creates
/// a new one. It also replaces low mul if found.
static Value *findOrCreateFullMul(Value *X, Value *Y, DominatorTree &DT) {

  if (auto *Mul = findFullMul(X, Y))
    return Mul;

  auto *MulTy = IntegerType::get(X->getContext(),
                                 X->getType()->getPrimitiveSizeInBits() * 2);
  IRBuilder<> Builder{getInstructionAfter(X, Y, DT)};
  auto *FullMul = Builder.CreateNUWMul(
      Builder.CreateZExt(X, MulTy, {"fullmul.", X->getName()}),
      Builder.CreateZExt(Y, MulTy, {"fullmul.", Y->getName()}), "fullmul");

  // If you find a low mul, replace it also with the full mul.
  if (auto *LowMul = findLowMul(X, Y)) {
    auto *FullMulLo =
        Builder.CreateTrunc(FullMul, LowMul->getType(), "fullmul.lo");
    LowMul->replaceAllUsesWith(FullMulLo);
  }

  return FullMul;
}

/// Matches the following pattern producing full multiplication:
///
/// %xl = and i64 %x, 4294967295
/// %xh = lshr i64 %x, 32
/// %yl = and i64 %y, 4294967295
/// %yh = lshr i64 %y, 32
///
/// %t0 = mul nuw i64 %yl, %xl
/// %t1 = mul nuw i64 %yl, %xh
/// %t2 = mul nuw i64 %yh, %xl
/// %t3 = mul nuw i64 %yh, %xh
///
/// %t0l = and i64 %t0, 4294967295
/// %t0h = lshr i64 %t0, 32
///
/// %u0 = add i64 %t0h, %t1
/// %u0l = and i64 %u0, 4294967295
/// %u0h = lshr i64 %u0, 32
///
/// %u1 = add i64 %u0l, %t2
/// %u1ls = shl i64 %u1, 32
/// %u1h = lshr i64 %u1, 32
///
/// %u2 = add i64 %u0h, %t3
///
/// %lo = or i64 %u1ls, %t0l
/// %hi = add i64 %u2, %u1h
///
static bool foldFullMul(Instruction &I, const DataLayout &DL,
                        DominatorTree &DT) {

  // We limit this up to 128 bits to have the low part mask be at most 64-bit
  // (m_SpecificInt() matcher limitation).
  static constexpr unsigned maxSizeInBits = 128;

  auto *ty = I.getType();
  if (!ty->isIntegerTy())
    return false;

  // Check the integer type size.
  // Also make sure the size in bits is even to make low-high split trivial.
  const auto sizeInBits = ty->getPrimitiveSizeInBits();
  if (sizeInBits > maxSizeInBits || sizeInBits % 2 != 0)
    return false;

  // Skip integers bigger than native.
  if (sizeInBits > DL.getLargestLegalIntTypeSizeInBits())
    return false;

  const auto halfSizeInBits = sizeInBits / 2; // Max 64.
  const auto Half = m_SpecificInt(halfSizeInBits);
  const auto lowMask =
      m_SpecificInt(~uint64_t{0} >> ((maxSizeInBits / 2) - halfSizeInBits));

  Value *x = nullptr;
  Value *y = nullptr;
  Value *t0 = nullptr;
  Value *t1 = nullptr;
  Value *t2 = nullptr;
  Value *t3 = nullptr;
  Value *u0 = nullptr;

  // Match low part of the full multiplication.
  //
  // First we match up to the multiplications t0, t1, t2.
  // The t0 is reachable by two edges and we _assume_ it's the same node
  // in general it does not have to be.
  //
  // The long pattern is: ((t2 + lo(t1 + hi(t0))) << 32) | lo(t0).
  bool LowLongPattern =
      match(&I, m_c_Or(m_And(m_Value(t0), lowMask),
                       m_Shl(m_c_Add(m_And(m_c_Add(m_LShr(m_Deferred(t0), Half),
                                                   m_Value(t1)),
                                           lowMask),
                                     m_Value(t2)),
                             Half)));

  // The short pattern is: ((t2 + t1) << Half) + t0.
  bool LowShortPattern = match(
      &I, m_c_Add(m_Value(t0), m_Shl(m_c_Add(m_Value(t1), m_Value(t2)), Half)));

  if (LowLongPattern || LowShortPattern) {
    // 1. Match t1 and remember its arguments. We start with t1 is asymmetric.
    // 2. Require t2 to be a swapped version of t1.
    // 3. For t0 require to have the same arguments as t1.
    if (match(t1,
              m_c_Mul(m_LShr(m_Value(x), Half), m_And(m_Value(y), lowMask))) &&
        match(t2, m_c_Mul(m_And(m_Specific(x), lowMask),
                          m_LShr(m_Specific(y), Half))) &&
        match(t0, m_c_Mul(m_And(m_Specific(x), lowMask),
                          m_And(m_Specific(y), lowMask)))) {
      // Replace with single multiplication.
      auto Mul = findOrCreateLowMul(I, x, y, DT);
      IRBuilder<> Builder{&I};
      auto Low = Builder.CreateTrunc(Mul, ty, "fullmul.lo");
      I.replaceAllUsesWith(Low);
      return true;
    }
  }

  // Match hi part of the full multiplication.
  //
  // First we match up to multiplications t2 and t3 and u0 node.
  // Then check the u0 node.
  // In the end check all 4 multiplications starting from asymmetric ones
  // the same as in matching the low part.
  if (match(&I,
            m_c_Add(
                m_LShr(m_c_Add(m_And(m_Value(u0), lowMask), m_Value(t2)), Half),
                m_c_Add(m_LShr(m_Deferred(u0), Half), m_Value(t3)))) &&
      match(u0, m_c_Add(m_LShr(m_Value(t0), Half), m_Value(t1)))) {
    if (match(t1,
              m_c_Mul(m_LShr(m_Value(x), Half), m_And(m_Value(y), lowMask))) &&
        match(t2, m_c_Mul(m_And(m_Specific(x), lowMask),
                          m_LShr(m_Specific(y), Half))) &&
        match(t0, m_c_Mul(m_And(m_Specific(x), lowMask),
                          m_And(m_Specific(y), lowMask))) &&
        match(t3, m_c_Mul(m_LShr(m_Specific(x), Half),
                          m_LShr(m_Specific(y), Half)))) {
      auto mul = findOrCreateFullMul(x, y, DT);
      IRBuilder<> Builder{&I};
      auto hi = Builder.CreateTrunc(Builder.CreateLShr(mul, halfSizeInBits), ty,
                                    "fullmul.hi");
      I.replaceAllUsesWith(hi);
      return true;
    }
  }

  return false;
}

/// This is the entry point for folds that could be implemented in regular
/// InstCombine, but they are separated because they are not expected to
/// occur frequently and/or have more than a constant-length pattern match.
static bool foldUnusualPatterns(Function &F, const DataLayout &DL,
                                DominatorTree &DT) {
  bool MadeChange = false;
  for (BasicBlock &BB : F) {
    // Ignore unreachable basic blocks.
    if (!DT.isReachableFromEntry(&BB))
      continue;
    // Do not delete instructions under here and invalidate the iterator.
    // Walk the block backwards for efficiency. We're matching a chain of
    // use->defs, so we're more likely to succeed by starting from the bottom.
    // Also, we want to avoid matching partial patterns.
    // TODO: It would be more efficient if we removed dead instructions
    // iteratively in this loop rather than waiting until the end.
    for (Instruction &I : make_range(BB.rbegin(), BB.rend())) {
      MadeChange |= foldAnyOrAllBitsSet(I);
      MadeChange |= foldGuardedRotateToFunnelShift(I);
      MadeChange |= foldFullMul(I, DL, DT);
    }
  }

  // We're done with transforms, so remove dead instructions.
  if (MadeChange)
    for (BasicBlock &BB : F)
      SimplifyInstructionsInBlock(&BB);

  return MadeChange;
}

/// This is the entry point for all transforms. Pass manager differences are
/// handled in the callers of this function.
static bool runImpl(Function &F, TargetLibraryInfo &TLI, DominatorTree &DT) {
  bool MadeChange = false;
  const DataLayout &DL = F.getParent()->getDataLayout();
  TruncInstCombine TIC(TLI, DL, DT);
  MadeChange |= TIC.run(F);
  MadeChange |= foldUnusualPatterns(F, DL, DT);
  return MadeChange;
}

void AggressiveInstCombinerLegacyPass::getAnalysisUsage(
    AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<DominatorTreeWrapperPass>();
  AU.addRequired<TargetLibraryInfoWrapperPass>();
  AU.addPreserved<AAResultsWrapperPass>();
  AU.addPreserved<BasicAAWrapperPass>();
  AU.addPreserved<DominatorTreeWrapperPass>();
  AU.addPreserved<GlobalsAAWrapperPass>();
}

bool AggressiveInstCombinerLegacyPass::runOnFunction(Function &F) {
  auto &TLI = getAnalysis<TargetLibraryInfoWrapperPass>().getTLI();
  auto &DT = getAnalysis<DominatorTreeWrapperPass>().getDomTree();
  return runImpl(F, TLI, DT);
}

PreservedAnalyses AggressiveInstCombinePass::run(Function &F,
                                                 FunctionAnalysisManager &AM) {
  auto &TLI = AM.getResult<TargetLibraryAnalysis>(F);
  auto &DT = AM.getResult<DominatorTreeAnalysis>(F);
  if (!runImpl(F, TLI, DT)) {
    // No changes, all analyses are preserved.
    return PreservedAnalyses::all();
  }
  // Mark all the analyses that instcombine updates as preserved.
  PreservedAnalyses PA;
  PA.preserveSet<CFGAnalyses>();
  PA.preserve<AAManager>();
  PA.preserve<GlobalsAA>();
  return PA;
}

char AggressiveInstCombinerLegacyPass::ID = 0;
INITIALIZE_PASS_BEGIN(AggressiveInstCombinerLegacyPass,
                      "aggressive-instcombine",
                      "Combine pattern based expressions", false, false)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(AggressiveInstCombinerLegacyPass, "aggressive-instcombine",
                    "Combine pattern based expressions", false, false)

// Initialization Routines
void llvm::initializeAggressiveInstCombine(PassRegistry &Registry) {
  initializeAggressiveInstCombinerLegacyPassPass(Registry);
}

void LLVMInitializeAggressiveInstCombiner(LLVMPassRegistryRef R) {
  initializeAggressiveInstCombinerLegacyPassPass(*unwrap(R));
}

FunctionPass *llvm::createAggressiveInstCombinerPass() {
  return new AggressiveInstCombinerLegacyPass();
}

void LLVMAddAggressiveInstCombinerPass(LLVMPassManagerRef PM) {
  unwrap(PM)->add(createAggressiveInstCombinerPass());
}
