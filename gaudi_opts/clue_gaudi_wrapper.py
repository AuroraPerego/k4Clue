#
# Copyright (c) 2020-2024 Key4hep-Project.
#
# This file is part of Key4hep.
# See https://key4hep.github.io/key4hep-doc/ for further info.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from Gaudi.Configuration import WARNING, DEBUG

from Configurables import k4DataSvc, MarlinProcessorWrapper

from Configurables import PodioInput
from Configurables import ClueGaudiAlgorithmWrapper__unsignedschar_3_ as ClueGaudiAlgorithmWrapper3
from Configurables import ClueGaudiAlgorithmWrapper__unsignedschar_2_ as ClueGaudiAlgorithmWrapper2
from Configurables import CLUENtuplizer
from Configurables import THistSvc
from Configurables import PodioOutput
from Configurables import ApplicationMgr

algList = []


evtsvc = k4DataSvc('EventDataSvc')
evtsvc.input = "/eos/user/a/aperego/fcc/k4Clue/build2/test/input_files/20240905_gammaFromVertex_10GeV_uniform_10events_reco_edm4hep.root"

inp = PodioInput('InputReader')
inp.collections = [
  'EventHeader',
  'MCParticles',
  'ECALBarrel',
  'ECALEndcap',
]
inp.OutputLevel = WARNING

MyAIDAProcessor = MarlinProcessorWrapper("MyAIDAProcessor")
MyAIDAProcessor.OutputLevel = WARNING
MyAIDAProcessor.ProcessorType = "AIDAProcessor"
MyAIDAProcessor.Parameters = {"FileName": ["histograms_clue_standalone"],
                    "FileType": ["root"],
                    "Compress": ["1"],
                    }

rho = 0.1 # MinLocalDensity
dc = 25 # CriticalDistance
do = 4 # OutlierDeltaFactor
MyClueGaudiAlgorithmWrapper = ClueGaudiAlgorithmWrapper3("ClueGaudiAlgorithmWrapperName")
MyClueGaudiAlgorithmWrapper.BarrelCaloHitsCollection = "ECALBarrel"
MyClueGaudiAlgorithmWrapper.EndcapCaloHitsCollection = "ECALEndcap"
MyClueGaudiAlgorithmWrapper.CriticalDistance = dc
MyClueGaudiAlgorithmWrapper.MinLocalDensity = rho
MyClueGaudiAlgorithmWrapper.OutlierDeltaFactor = do
MyClueGaudiAlgorithmWrapper.OutputLevel = DEBUG

MyCLUENtuplizer = CLUENtuplizer("CLUEAnalysis")
MyCLUENtuplizer.ClusterCollection = "CLUEClusters"
MyCLUENtuplizer.BarrelCaloHitsCollection = "ECALBarrel"
MyCLUENtuplizer.EndcapCaloHitsCollection = "ECALEndcap"
MyCLUENtuplizer.SingleMCParticle = True
MyCLUENtuplizer.OutputLevel = WARNING

str_params = str(rho).replace(".","p") + "_" + str(dc).replace(".","p") + "_" + str(do).replace(".","p")
filename = "rec DATAFILE='k4clue_analysis_output_3D_"+str_params+".root' TYP='ROOT' OPT='RECREATE'"
THistSvc().Output = [filename]
THistSvc().OutputLevel = WARNING
THistSvc().PrintAll = False
THistSvc().AutoSave = True
THistSvc().AutoFlush = True

out = PodioOutput("out")
MyClueGaudiAlgorithmWrapper.BarrelCaloHitsCollection = "ECALBarrel"
MyClueGaudiAlgorithmWrapper.EndcapCaloHitsCollection = "ECALEndcap"
out.filename = "my_output_clue_standalone.root"
out.outputCommands = ["keep *"]

algList.append(inp)
algList.append(MyAIDAProcessor)
algList.append(MyClueGaudiAlgorithmWrapper)
algList.append(MyCLUENtuplizer)
algList.append(out)

ApplicationMgr( TopAlg = algList,
                EvtSel = 'NONE',
                EvtMax   = 3,
                ExtSvc = [evtsvc],
                OutputLevel=WARNING
              )
