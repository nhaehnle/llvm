//===-- GCNExperimentSubvectorAnalysis.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
///
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "AMDGPUSubtarget.h"
#include "GCNRegPressure.h"
#include "SIInstrInfo.h"
#include "SIMachineFunctionInfo.h"
#include "SIRegisterInfo.h"
#include "MCTargetDesc/AMDGPUMCTargetDesc.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/CodeGen/LiveIntervals.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineOptimizationRemarkEmitter.h"
#include "llvm/CodeGen/MachineRegionInfo.h"

using namespace llvm;

#define DEBUG_TYPE "gcn-experiment-subvector-analysis"

namespace {

class GCNExperimentSubvectorAnalysis : public MachineFunctionPass {
public:
  static char ID;

public:
  GCNExperimentSubvectorAnalysis() : MachineFunctionPass(ID) {
    initializeGCNExperimentSubvectorAnalysisPass(*PassRegistry::getPassRegistry());
  }

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override {
    return DEBUG_TYPE;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<MachineOptimizationRemarkEmitterPass>();
    AU.addRequired<LiveIntervals>();
    AU.addRequired<MachineDominatorTree>();
    AU.addRequired<MachineRegionInfoPass>();
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

private:
  const GCNSubtarget *ST;
  const SIInstrInfo *TII;
  const SIRegisterInfo *TRI;
  const MachineDominatorTree *MDT;
  const MachineRegisterInfo *MRI;
  const MachineRegionInfo *Regions;
  SIMachineFunctionInfo *MFI;

  unsigned MaxVGPRs; // maximum total number of live VGPRs encountered

  // The "blocking" point that uniquely defines the blocking polyhedra
  // containing all the (unshared, shared) VGPR counts with which the
  // encountered register pressure can be satisfied.
  unsigned BlockingUnsharedVGPRs;
  unsigned BlockingSharedVGPRs;

  enum RegisterKind {
    RegNotVGPR,
    RegShareable,
    RegNotShared,
  };

  // For every virtual register that has been encountered, indicate whether
  // it is a VGPR and if so, whether it is a candidate for a shared register.
  DenseMap<unsigned, RegisterKind> RegisterKinds;

  struct BBInfo {
    bool breaksSubvLoops = false; // contains an instruction that breaks subvector loops
    bool exitCanBeSubv = false; // a subvector loop can be active when leaving the BB
    bool entryCanBeSubv = false; // a subvector loop can be active when entering the BB
  };

  DenseMap<MachineBasicBlock *, BBInfo> BBInfos;

  struct RegionInfo {
    // Whether the interior (i.e., the region excluding the entry / exit blocks)
    // contains subvector loop breaking instructions.
    bool interiorBreaksSubvLoops = false;

    // Whether it is possible to have a subvector loop that starts in the entry
    // block and ends in the exit block.
    bool allowsSubvLoop = false;
  };

  // Set of regions whose interior (i.e., excluding the entry / exit blocks)
  // contains subvector loop breaking instructions.
  DenseMap<MachineRegion *, RegionInfo> RegionInfos;

  void markRegion(MachineRegion *region, bool subvLoopCover = false);
  void markNotShared(GCNRPTracker &RPT);
  void updateRegisterPressure(GCNRPTracker &RPT);
  RegisterKind getRegisterKind(unsigned Reg);
  bool breaksSubvectorLoops(const MachineInstr &MI) const;
};

} // anonymous namespace

INITIALIZE_PASS_BEGIN(GCNExperimentSubvectorAnalysis, DEBUG_TYPE, DEBUG_TYPE,
                      false, false)
INITIALIZE_PASS_DEPENDENCY(LiveIntervals)
INITIALIZE_PASS_DEPENDENCY(MachineDominatorTree)
INITIALIZE_PASS_DEPENDENCY(MachineRegionInfoPass)
INITIALIZE_PASS_DEPENDENCY(MachineOptimizationRemarkEmitterPass)
INITIALIZE_PASS_END(GCNExperimentSubvectorAnalysis, DEBUG_TYPE, DEBUG_TYPE,
                    false, false)


char GCNExperimentSubvectorAnalysis::ID = 0;

char &llvm::GCNExperimentSubvectorAnalysisID = GCNExperimentSubvectorAnalysis::ID;

FunctionPass *llvm::createGCNExperimentSubvectorAnalysisPass() {
  return new GCNExperimentSubvectorAnalysis;
}

bool GCNExperimentSubvectorAnalysis::runOnMachineFunction(MachineFunction &MF) {
  ST = &MF.getSubtarget<GCNSubtarget>();
  TII = ST->getInstrInfo();
  TRI = ST->getRegisterInfo();
  MRI = &MF.getRegInfo();
  MDT = &getAnalysis<MachineDominatorTree>();
  Regions = &getAnalysis<MachineRegionInfoPass>().getRegionInfo();
  MFI = MF.getInfo<SIMachineFunctionInfo>();
  LiveIntervals *LIS = &getAnalysis<LiveIntervals>();
//  SlotIndexes *Ind = LIS->getSlotIndexes();

  // First, determine which basic blocks contain instructions that we cannot
  // or don't want to have in subvector loops.
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (breaksSubvectorLoops(MI)) {
        BBInfos[&MBB].breaksSubvLoops = true;

        MachineRegion *region = Regions->getRegionFor(&MBB);
        while (region && region->getEntry() != &MBB) {
          RegionInfo &regionInfo = RegionInfos[region];
          if (regionInfo.interiorBreaksSubvLoops)
            break;
          regionInfo.interiorBreaksSubvLoops = true;
          region = region->getParent();
        }

        break;
      }
    }
  }

  // Recursively determine which regions' interiors can be covered by
  // subvector loops.
  for (const auto &region : *Regions->getTopLevelRegion())
    markRegion(&*region);

  // Use the collected information to determine which basic blocks can be
  // entered/exited while in a subvector loop.
  for (MachineBasicBlock &MBB : MF) {
    BBInfo &info = BBInfos[&MBB];
    MachineRegion *region = Regions->getRegionFor(&MBB);

    if (region == Regions->getTopLevelRegion())
      region = nullptr;

    // First, handle blocks which are not the entry of a region.
    if (!region || &MBB != region->getEntry()) {
      if (region && RegionInfos[region].allowsSubvLoop) {
        info.exitCanBeSubv = true;
      } else {
        // Handle individual blocks that are not merged with their successor
        if (MBB.succ_size() == 1) {
          MachineBasicBlock *succ = *MBB.succ_begin();
          if (succ->pred_size() == 1)
            info.exitCanBeSubv = true;
        }
      }
    } else {
      if (RegionInfos[region].allowsSubvLoop)
        info.exitCanBeSubv = true;
    }

    if (info.exitCanBeSubv) {
      for (MachineBasicBlock *succ : MBB.successors())
        BBInfos[succ].entryCanBeSubv = true;
    }
  }

  // Sanity checks of basic block subvector entry/exit status.
  for (MachineBasicBlock &MBB : MF) {
    BBInfo &info = BBInfos[&MBB];

    for (MachineBasicBlock *pred : MBB.predecessors()) {
      assert(info.entryCanBeSubv == BBInfos[pred].exitCanBeSubv);
    }
    for (MachineBasicBlock *succ : MBB.successors()) {
      assert(info.exitCanBeSubv == BBInfos[succ].entryCanBeSubv);
    }

    if (info.entryCanBeSubv)
      LLVM_DEBUG(dbgs() << MBB.getName() << ": can enter in subvector loop\n");
    if (info.exitCanBeSubv)
      LLVM_DEBUG(dbgs() << MBB.getName() << ": can exit in subvector loop\n");
  }

  LLVM_DEBUG(dbgs() << '\n');

  // Determine which registers are eligible for shared VGPRs.
  GCNDownwardRPTracker RPT(*LIS);

  for (MachineBasicBlock &MBB : MF) {
    if (MBB.empty()) {
      LLVM_DEBUG(dbgs() << "Skip empty basic block " << MBB.getName() << '\n');
      continue;
    }

    LLVM_DEBUG(dbgs() << "Scan basic block " << MBB.getName() << '\n');

    RPT.reset(*MBB.begin());

    if (!BBInfos[&MBB].entryCanBeSubv)
      markNotShared(RPT);

    for (MachineInstr &MI : MBB) {
      LLVM_DEBUG(dbgs() << "  Before: " << MI);

      bool breaksSubv = breaksSubvectorLoops(MI);
      if (breaksSubv)
        markNotShared(RPT);

      RPT.advance();

      if (breaksSubv)
        markNotShared(RPT);
    }

    if (!BBInfos[&MBB].exitCanBeSubv)
      markNotShared(RPT);

    LLVM_DEBUG(dbgs() << '\n');
  }

  // Now determine the register pressure across all basic blocks.
  MaxVGPRs = 0;
  BlockingUnsharedVGPRs = 0;
  BlockingSharedVGPRs = 0;

  for (MachineBasicBlock &MBB : MF) {
    if (MBB.empty()) {
      LLVM_DEBUG(dbgs() << "Skip empty basic block " << MBB.getName() << '\n');
      continue;
    }

    LLVM_DEBUG(dbgs() << "Rescan basic block " << MBB.getName() << '\n');

    RPT.reset(*MBB.begin());

    updateRegisterPressure(RPT);

    for (MachineInstr &MI : MBB) {
      LLVM_DEBUG(dbgs() << "  " << MI);
      RPT.advanceToNext();
      RPT.advanceBeforeNext();
      updateRegisterPressure(RPT);
    }

    LLVM_DEBUG(dbgs() << '\n');
  }

  MachineOptimizationRemarkEmitter &ORE = getAnalysis<MachineOptimizationRemarkEmitterPass>().getORE();
  MachineOptimizationRemarkAnalysis R(DEBUG_TYPE, "RegisterPressure",
                                      MF.getFunction().getSubprogram(),
                                      &MF.front());
  R << "MaxVGPRs: " << ore::NV("MaxVGPRs", MaxVGPRs)
    << " Blocking(Unshared, Shared): "
    << ore::NV("BlockingUnsharedVGPRs", BlockingUnsharedVGPRs)
    << ", " << ore::NV("BlockingSharedVGPRs", BlockingSharedVGPRs);
  ORE.emit(R);

  RegisterKinds.clear();
  BBInfos.clear();
  RegionInfos.clear();

  return false; // no changes
}

void GCNExperimentSubvectorAnalysis::markRegion(MachineRegion *region,
                                                bool subvLoopCover) {
  RegionInfo &info = RegionInfos[region];
  if (!subvLoopCover && !info.interiorBreaksSubvLoops) {
    // The exit block is not itself part of the region, and it could be the
    // exit block of multiple region. Check that it uniquely ends the given
    // region as a precondition for wrapping it in a subvector loop.
    MachineBasicBlock *entry = region->getEntry();
    MachineBasicBlock *exit = region->getExit();
    if (MDT->dominates(entry, exit)) {
      subvLoopCover = true;

      // Except if the region header is also a loop header with back edges
      // from inside the region...
      for (MachineBasicBlock *pred : entry->predecessors()) {
        if (region->contains(pred)) {
          subvLoopCover = false;
          break;
        }
      }
    }
  }
  if (subvLoopCover)
    info.allowsSubvLoop = true;

  for (const auto &child : *region)
    markRegion(&*child, subvLoopCover);
}

void GCNExperimentSubvectorAnalysis::markNotShared(GCNRPTracker &RPT) {
  for (const auto &liveRegEntry : RPT.getLiveRegs()) {
    unsigned reg = liveRegEntry.first;
    assert(TargetRegisterInfo::isVirtualRegister(reg));

    auto it = RegisterKinds.find(reg);
    if (it != RegisterKinds.end()) {
      if (it->second == RegShareable) {
        LLVM_DEBUG(dbgs() << "    not shared: " << printReg(reg, TRI, 0, MRI) << '\n');
        it->second = RegNotShared;
      }
      continue;
    }

    if (!TRI->isVGPR(*MRI, reg)) {
      RegisterKinds[reg] = RegNotVGPR;
    } else {
      RegisterKinds[reg] = RegNotShared;
      LLVM_DEBUG(dbgs() << "    not shared: " << printReg(reg, TRI, 0, MRI) << '\n');
    }
  }
}

void GCNExperimentSubvectorAnalysis::updateRegisterPressure(GCNRPTracker &RPT) {
  unsigned curVGPRs = 0;
  unsigned curUnsharedVGPRs = 0;
  unsigned curSharedVGPRs = 0;

  for (const auto &liveRegEntry : RPT.getLiveRegs()) {
    unsigned reg = liveRegEntry.first;
    RegisterKind kind = getRegisterKind(reg);
    if (kind == RegNotVGPR)
      continue;

    unsigned dwords = liveRegEntry.second.getNumLanes();
    curVGPRs += dwords;

    if (kind == RegNotShared)
      curUnsharedVGPRs += dwords;
    else
      curSharedVGPRs += dwords;
  }

  MaxVGPRs = std::max(MaxVGPRs, curVGPRs);
  BlockingUnsharedVGPRs = std::max(BlockingUnsharedVGPRs, curUnsharedVGPRs);
  BlockingSharedVGPRs = MaxVGPRs - BlockingUnsharedVGPRs;

  LLVM_DEBUG(dbgs() << "    VGPRs: " << curVGPRs << " (unshared: "
                    << curUnsharedVGPRs << ", shared: " << curSharedVGPRs
                    << "); overall: " << MaxVGPRs << " ("
                    << BlockingUnsharedVGPRs << ", " << BlockingSharedVGPRs
                    << ")\n");
}

/// Determine whether the given virtual register is a candidate for putting
/// into a shared VGPR.
GCNExperimentSubvectorAnalysis::RegisterKind
GCNExperimentSubvectorAnalysis::getRegisterKind(unsigned Reg) {
  auto it = RegisterKinds.find(Reg);
  if (it != RegisterKinds.end())
    return it->second;

  assert(TargetRegisterInfo::isVirtualRegister(Reg));
  if (!TRI->isVGPR(*MRI, Reg))
    return (RegisterKinds[Reg] = RegNotVGPR);

  // Must be shareable, since otherwise we would have marked it earlier.
  return (RegisterKinds[Reg] = RegShareable);
}

/// Returns \c true if the given instructions cannot go into a subvector loop
/// or we don't want it in a subvector loop.
bool GCNExperimentSubvectorAnalysis::breaksSubvectorLoops(
    const MachineInstr &MI) const {
  if (TII->hasUnwantedEffectsWhenEXECEmpty(MI))
    return true;

  // We don't want high-latency instructions in a subvector loop.
  //
  // Note: assume that scalar loads will be cached and are therefore okay in
  //       a subvector loop -- this is a stretch, but we want to estimate the
  //       maximum gain of subvector loops.
  if (TII->isVMEM(MI) || TII->isFLAT(MI) || TII->isDS(MI))
    return true;

  // We don't want interpolation instructions in a subvector loop for
  // efficiency.
  if (TII->isVINTRP(MI))
    return true;

  return false;
}
