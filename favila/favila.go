// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//color diff runs a color learning task, looking at differentiation of color.
//based on ra25:

// ra25 runs a simple random-associator four-layer leabra network
// that uses the standard supervised learning paradigm to learn
// mappings between 25 random input / output patterns
// defined over 5x5 input / output layers (i.e., 25 units)
package main

import (
	// "reflect"
	"flag"
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
	"time"
	"math"
	"encoding/json"
	// "io"
	"io/ioutil"
	// "bufio"
	"strings"

	"github.com/PrincetonCompMemLab/private-emergent/emer"
	"github.com/PrincetonCompMemLab/private-emergent/env"
	"github.com/PrincetonCompMemLab/private-emergent/netview"
	"github.com/PrincetonCompMemLab/private-emergent/params"
	"github.com/PrincetonCompMemLab/private-emergent/patgen"
	"github.com/PrincetonCompMemLab/private-emergent/prjn"
	"github.com/PrincetonCompMemLab/private-emergent/relpos"
	"github.com/PrincetonCompMemLab/private-emergent/weights"
	"github.com/PrincetonCompMemLab/private-emergent/erand"
	"github.com/emer/etable/agg"
	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"
	_ "github.com/emer/etable/etview" // include to get gui views
	"github.com/emer/etable/split"
	"github.com/PrincetonCompMemLab/private-leabra/leabra"
	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/gi/giv"
	"github.com/goki/ki/ki"
	"github.com/goki/ki/kit"
	"github.com/goki/mat32"
)


func main() {
	TheSim.New()
	TheSim.Config()
	if len(os.Args) > 1 {
		TheSim.CmdArgs() // simple assumption is that any args = no gui -- could add explicit arg if you want
	} else {
		TheSim.ConfigSaveWts("experiment", "")
		gimain.Main(func() { // this starts gui -- requires valid OpenGL display connection (e.g., X11)
			guirun()
		})
	}
}

func guirun() {
	TheSim.NewRndSeed()
	TheSim.Init()
	win := TheSim.ConfigGui()
	win.StartEventLoop()
}

// LogPrec is precision for saving float values in logs
const LogPrec = 4

// ParamSets is the default set of parameters -- Base is always applied, and others can be optionally
// selected to apply on top of that
var ParamSets = params.Sets{
	{Name: "Base", Desc: "these are the best params", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Prjn", Desc: "norm and momentum on works better, but wt bal is not better for smaller nets",
				Params: params.Params{
					"Prjn.Learn.Norm.On":     "false",
					"Prjn.Learn.Momentum.On": "false",
					"Prjn.Learn.WtBal.On":    "false",
					"Prjn.WtScale.Rel": "1",
					"Prjn.Learn.Lrate": ".2",
					"Prjn.Learn.XCal.SetLLrn" : "0", //fix LLrn
					"Prjn.Learn.XCal.MLrn" : "0", //disable error correction learning
					"Prjn.WtInit.InitStrategy" : "Rand",
					"Prjn.WtInit.NumOverlapUnits" : "2",
					"Prjn.WtInit.NumTotalUnits" : "6",
					"Prjn.WtInit.Var" : ".05",

				}},
			{Sel: ".Back", Desc: "top-down back-projections MUST have lower relative weight scale, otherwise network hallucinates",
				Params: params.Params{
					"Prjn.WtScale.Rel": "1",
					"Prjn.WtScale.Abs": "1",
				}},

			{Sel: "#OutputToOutput", Desc: "default connection (will change for non-shape learning task).",
				Params: params.Params{
					"Prjn.Learn.Lrate": ".2",
					"Prjn.WtScale.Rel": "1",
					"Prjn.WtInit.InitStrategy" : "TesselSpec",
				}},

			{Sel: "#HiddenToOutput", Desc: "default connection (will change for non-shape learning task).",
				Params: params.Params{
					"Prjn.WtScale.Abs": "1.2", //was 1.4
					"Prjn.WtInit.InitStrategy" : "HiddenToOutput",
					"Prjn.WtInit.Var" : ".01",
					"Prjn.WtInit.Mean" : ".02",
					"Prjn.WtInit.SameDiffCondition" : "Same", // Same or Different

				}},
			{Sel: "#OutputToHidden", Desc: "default connection (will change for non-shape learning task).",
				Params: params.Params{
					"Prjn.WtScale.Abs": "1.7", //1.1
					"Prjn.WtInit.Var" : ".01",
					"Prjn.WtInit.Mean" : ".02",

					}},

			{Sel: "#HiddenToHidden", Desc: "default connection (will change for non-shape learning task).",
			Params: params.Params{
				"Prjn.WtInit.InitStrategy" : "RandomHiddenToHidden",
				"Prjn.WtInit.Var" : ".1", //was .1
				"Prjn.WtInit.Mean" : "0.5",
				// "Prjn.WtInit.Dist" : "Gaussian",
				"Prjn.WtInit.SparseMix" : "0.5", //.3
				"Prjn.WtInit.SecondModeMean" : "0.05",
				"Prjn.WtInit.SecondModeVar" : "0.01",
			}},
			{Sel: "#CategoryToHidden", Desc: "default connection (will change for non-shape learning task).",
				Params: params.Params{
					"Prjn.WtInit.InitStrategy" : "CategoryToHidden",
					"Prjn.WtInit.Var" : ".01", //was .1
					"Prjn.WtInit.Mean" : ".02",

				}},

				{Sel: "#HiddenToCategory", Desc: "default connection (will change for non-shape learning task).",
					Params: params.Params{
						"Prjn.WtInit.Var" : ".01", //was .1
						"Prjn.WtInit.Mean" : ".02",

					}},

			{Sel: "Layer", Desc: "using default 1.8 inhib for all of network + soft clamping + no oscillation",
				Params: params.Params{
					"Layer.Inhib.Layer.Gi": "1.8",
					"Layer.Off": "false",
					"Layer.Inhib.ActAvg.Fixed" : "true",
					"Layer.Inhib.InhibType" : "FFFB",
					"Layer.Inhib.ActAvg.Init" : ".15",
					"Layer.Act.Clamp.Hard": "false",
					"Layer.Act.Clamp.Gain": ".2", // originally was .5
					"Layer.OscAmnt" : "0", // make this true or false depending on whether you want oscillations
					"Layer.Learn.AvgL.SetAveL" : "true",
					"Layer.Learn.AvgL.AveLFix" : ".4",
				  	"Layer.Learn.ActAvg.LrnM" : ".9", //default is .1. How much of medium term to mix with short term for comparison in XCAL caluclation.
					"Layer.Inhib.K_max" : "-1",

				}},

			// {Sel: "#Scene", Desc: "color output layer turned on for color learning task",
			// 	Params: params.Params{
			// 		"Layer.Inhib.Layer.Gi": "2.0", //1.6
			// 		"Layer.Inhib.Pool.On": "false",
			// 		"Layer.Inhib.Pool.Gi": "1.8", // .66
			// 		"Layer.Act.Clamp.Gain": "1", // originally was .5
			// }},

			// {Sel: "#Hidden", Desc: "color output layer turned on for color learning task",
			// 	Params: params.Params{
			// 		"Layer.Inhib.Layer.Gi": "3", //1.6
			// 		"Layer.Inhib.Layer.FF": "3.6",
			// 		"Layer.Inhib.Layer.FB": "1", //1.6
			// 		"Layer.Learn.AvgL.SetAveL" : "true",
			// 		"Layer.Learn.AvgL.AveLFix" : ".6",
			// 	}},


			{Sel: "#SceneToHidden", Desc: "default connection (will change for non-shape learning task).",
			Params: params.Params{
				"Prjn.WtInit.InitStrategy" : "MedShared",
				}},

			// {Sel: "#Output", Desc: "output definitely needs lower inhib -- true for smaller layers in general",
			// 	Params: params.Params{
			// 		"Layer.Inhib.Layer.Gi": "1.4",
			// 	}},
		},

		"Sim": &params.Sheet{ // sim params apply to sim object
			{Sel: "Sim", Desc: "Number of epochs to loop through two tasks to interleave. The number is for each task. So if set to be 10, it'll go to epoch 20 overall",
				Params: params.Params{
					"Sim.LoopEpcs": "8",
					"Sim.Time.CycPerQtr": "50",
					"Sim.DoRunColorTests": "true",
					"Sim.DoRunSceneTests": "false",
				}},
		},
	}},
	// {Name: "DefaultInhib", Desc: "output uses default inhib instead of lower", Sheets: params.Sheets{
	// 	"Network": &params.Sheet{
	// 		{Sel: "#Output", Desc: "go back to default",
	// 			Params: params.Params{
	// 				"Layer.Inhib.Layer.Gi": "1.8",
	// 			}},
	// 	},
	// 	"Sim": &params.Sheet{ // sim params apply to sim object
	// 		{Sel: "Sim", Desc: "takes longer -- generally doesn't finish..",
	// 			Params: params.Params{
	// 				"Sim.MaxEpcs": "100",
	// 			}},
	// 	},
	// }},
	{Name: "NoMomentum", Desc: "no momentum or normalization", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Prjn", Desc: "no norm or momentum",
				Params: params.Params{
					"Prjn.Learn.Norm.On":     "false",
					"Prjn.Learn.Momentum.On": "false",
				}},
		},
	}},
	{Name: "WtBalOn", Desc: "try with weight bal on", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Prjn", Desc: "weight bal on",
				Params: params.Params{
					"Prjn.Learn.WtBal.On": "true",
				}},
		},
	}},
	{Name: "TaskColorWOOsc", Desc: "params for the second, color learning task", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Layer", Desc: "using default 1.8 inhib for all of network + no oscillation",
				Params: params.Params{
					"Layer.OscAmnt" : "0",
					"Layer.Learn.AvgL.SetAveL" : "true",
					"Layer.Learn.AvgL.AveLFix" : ".4",
					"Layer.Learn.LearningMP" : "1",
				}},
			{Sel: "Prjn", Desc: "norm and momentum on works better, but wt bal is not better for smaller nets",
				Params: params.Params{
					"Prjn.Learn.XCal.SetLLrn" : "true", // was 0
					"Prjn.Learn.XCal.MLrn" : ".9",
					"Prjn.Learn.XCal.LLrn" : ".1",
					"Prjn.Learn.XCal.LTD_mult" : "1",
					"Prjn.Learn.XCal.DThr_NMPH" : ".0001",
					"Prjn.Learn.XCal.DRev_NMPH" : ".1",
					"Prjn.Learn.XCal.DRevMag_NMPH" : "-0.5",
					"Prjn.Learn.XCal.ThrP_NMPH" : ".5",
					"Prjn.Learn.NMPH" : "false",
				}},
			{Sel: "#Output", Desc: "color output layer turned on for color learning task",
				Params: params.Params{
					"Layer.Off": "false",
				}},
			{Sel: "#CategoryToHidden", Desc: "default connection (will change for non-shape learning task).",
				Params: params.Params{
					"Prjn.WtScale.Rel": "1",

					}},
			{Sel: "#OutputToOutput", Desc: "default connection (will change for non-shape learning task).",
				Params: params.Params{
					"Prjn.WtScale.Rel": "2",
						}},
			{Sel: "#OutputToHidden", Desc: "default connection (will change for non-shape learning task).",
			Params: params.Params{
				"Prjn.WtScale.Rel": "2",
					}},

			{Sel: "#SceneToHidden", Desc: "default connection (will change for non-shape learning task).",
				Params: params.Params{
					"Prjn.WtScale.Abs": "1",
					"Prjn.WtScale.Rel": "1",
				}},
		},

		"Sim": &params.Sheet{ // sim params apply to sim object
			{Sel: "Sim", Desc: "best params always finish in this time",
				Params: params.Params{
					"Sim.MaxEpcs": "8",
				}},
			},
	}},
	{Name: "TaskColorRecall", Desc: "params for the second, color learning task", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Prjn", Desc: "norm and momentum on works better, but wt bal is not better for smaller nets",
				Params: params.Params{
					"Prjn.Learn.XCal.SetLLrn" : "true",
					"Prjn.Learn.XCal.MLrn" : "0",
					"Prjn.Learn.XCal.LLrn" : "1",
					"Prjn.Learn.XCal.LTD_mult" : "3",
					"Prjn.Learn.XCal.DThr_NMPH" : ".3",
					"Prjn.Learn.XCal.DRev_NMPH" : ".45",
					"Prjn.Learn.XCal.DRevMag_NMPH" : "-0.3",
					"Prjn.Learn.XCal.DMaxMag_NMPH" : "0.3",
					"Prjn.Learn.XCal.ThrP_NMPH" : ".6",
					"Prjn.Learn.NMPH" : "true",
				}},
			{Sel: "Layer", Desc: "using default 1.8 inhib for all of network + no oscillation",
				Params: params.Params{
					"Layer.OscAmnt" : ".18", // was .18
					"Layer.Learn.AvgL.SetAveL" : "true",
					"Layer.Learn.AvgL.AveLFix" : ".65",
					"Layer.Learn.LearningMP" : "0",
					"Layer.Inhib.ActAvg.Fixed" : "true",
					"Layer.Inhib.ActAvg.Init" : ".15",

				}},

			{Sel: "#HiddenToHidden", Desc: "default connection (will change for non-shape learning task).",
				Params: params.Params{
					"Prjn.Learn.XCal.DThr_NMPH" : ".11", //.17
					"Prjn.Learn.XCal.DRev_NMPH" : "0.23",
					"Prjn.Learn.XCal.DRevMag_NMPH" : "-1.5",
					"Prjn.Learn.XCal.DMaxMag_NMPH" : "0.1",
					"Prjn.Learn.XCal.ThrP_NMPH" : ".4", //.5
					"Prjn.Learn.Lrate" : "1",
					"Prjn.WtScale.Abs": "1.8", //1.4
				    "Prjn.WtInit.Var" : ".05",

				}},

			{Sel: "#SceneToHidden", Desc: "default connection (will change for non-shape learning task).",
			Params: params.Params{
				"Prjn.Learn.XCal.DThr_NMPH" : "0.215",
				"Prjn.Learn.XCal.DRev_NMPH" : "0.4",
				"Prjn.Learn.XCal.DRevMag_NMPH" : "-2.5",
				"Prjn.Learn.XCal.DMaxMag_NMPH" : "0.3",
				"Prjn.Learn.XCal.ThrP_NMPH" : "0.6",
				"Prjn.WtScale.Abs": ".3", //.5
				"Prjn.Learn.Lrate": "1",
				}},

		{Sel: "#HiddenToScene", Desc: "default connection (will change for non-shape learning task).",
			Params: params.Params{
				"Prjn.Learn.XCal.DThr_NMPH" : "0.215",
				"Prjn.Learn.XCal.DRev_NMPH" : "0.4",
				"Prjn.Learn.XCal.DRevMag_NMPH" : "-2.5",
				"Prjn.Learn.XCal.DMaxMag_NMPH" : "0.3",
				"Prjn.Learn.XCal.ThrP_NMPH" : "0.6",
				"Prjn.WtScale.Abs": ".2",//.5
				"Prjn.Learn.Lrate": "1", //005
				}},
				
		{Sel: "#HiddenToOutput", Desc: "default connection (will change for non-shape learning task).",
			Params: params.Params{
				"Prjn.Learn.XCal.DThr_NMPH" : ".11", //.17
				"Prjn.Learn.XCal.DRev_NMPH" : "0.23",
				"Prjn.Learn.XCal.DRevMag_NMPH" : "-.01",
				"Prjn.Learn.XCal.DMaxMag_NMPH" : "0.5",
				"Prjn.Learn.XCal.ThrP_NMPH" : ".4", //.5
				"Prjn.Learn.Lrate": "1",
				}},

		{Sel: "#OutputToHidden", Desc: "default connection (will change for non-shape learning task).",
				Params: params.Params{
					"Prjn.Learn.XCal.DThr_NMPH" : ".11", //.17
					"Prjn.Learn.XCal.DRev_NMPH" : "0.23",
					"Prjn.Learn.XCal.DRevMag_NMPH" : "-.01",
					"Prjn.Learn.XCal.DMaxMag_NMPH" : "0.5",
					"Prjn.Learn.XCal.ThrP_NMPH" : ".4", //.5
					"Prjn.Learn.Lrate": "1",
			}},
	
		{Sel: "#OutputToOutput", Desc: "default connection (will change for non-shape learning task).",
			Params: params.Params{
				"Prjn.Learn.XCal.DThr_NMPH" : ".11", //.17
				"Prjn.Learn.XCal.DRev_NMPH" : "0.23",
				"Prjn.Learn.XCal.DRevMag_NMPH" : "-1.5",
				"Prjn.Learn.XCal.DMaxMag_NMPH" : "0.1",
				"Prjn.Learn.XCal.ThrP_NMPH" : ".4", //.5
			}},

		{Sel: "#CategoryToHidden", Desc: "default connection (will change for non-shape learning task).",
					Params: params.Params{
						"Prjn.Learn.XCal.DThr_NMPH" : "0.2", // was .2115
						"Prjn.Learn.XCal.DRev_NMPH" : "0.3", //.37
						"Prjn.Learn.XCal.DRevMag_NMPH" : "-.1",
						"Prjn.Learn.XCal.DMaxMag_NMPH" : "0.06",
						"Prjn.Learn.XCal.ThrP_NMPH" : "0.46", // .4
						"Prjn.Learn.Lrate": "1", //005
						"Prjn.WtScale.Abs": ".1", //.1
			}},

		{Sel: "#HiddenToCategory", Desc: "default connection (will change for non-shape learning task).",
					Params: params.Params{
						"Prjn.Learn.XCal.DThr_NMPH" : "0.2", // was .2115
						"Prjn.Learn.XCal.DRev_NMPH" : "0.3", //.37
						"Prjn.Learn.XCal.DRevMag_NMPH" : "-.1",
						"Prjn.Learn.XCal.DMaxMag_NMPH" : "0.06",
						"Prjn.Learn.XCal.ThrP_NMPH" : "0.46", // .4
						"Prjn.Learn.Lrate": "1", //005
						"Prjn.WtScale.Abs": ".1",//.3
			}},

		{Sel: "#Output", Desc: "color output layer turned on for color learning task",
			Params: params.Params{
				"Layer.Off": "false",
				"Layer.OscAmnt" : ".07", // was .07
				// "Layer.Inhib.Layer.Gi": "2.4",
				// "Layer.Inhib.Layer.FF": "1",
				// "Layer.Inhib.Layer.FB": "1",
				"Layer.Inhib.InhibType" : "KWTA",
				"Layer.Inhib.K_for_WTA" : "1",
				"Layer.Inhib.K_point" : ".75", //.75
				"Layer.Inhib.Target_diff" : ".03",
				"Layer.Act.Clamp.Gain": "1", // originally was .5
				"Layer.Act.Clamp.Hard": "false",

				}},

		{Sel: "#Hidden", Desc: "color output layer turned on for color learning task",
			Params: params.Params{
				"Layer.OscAmnt": ".067", // was .066
				"Layer.Inhib.Layer.Gi": "1", // 2.26
				"Layer.Inhib.InhibType" : "KWTA",
				"Layer.Inhib.K_for_WTA" : "6",
				"Layer.Inhib.K_max" : "10",
				"Layer.Inhib.K_point" : ".8", //was .75
				"Layer.Inhib.Target_diff" : ".02", //was .03
				"Layer.Act.XX1.Gain" : "100",
				"Layer.Inhib.Layer.FF": ".7", //.7
				"Layer.Inhib.Layer.FB": "1.6", //1.6
				"Layer.Inhib.Layer.MaxVsAvg": "1", //0
				"Layer.Learn.AvgL.SetAveL" : "true",
				"Layer.Learn.AvgL.AveLFix" : ".6",
				// "Layer.Act.XX1.Thr": "0.4965",
				// "Layer.Act.XX1.NVar": "0.03",
				// "Layer.Act.XX1.InterpRange": "0.05",
				// "Layer.Act.XX1.SigMult": "0.5",
			}},

			{Sel: "#Scene", Desc: "color output layer turned on for color learning task",
				Params: params.Params{
					"Layer.OscAmnt": ".2", // was .3
	        		"Layer.Act.Clamp.Gain": ".3", // originally was 1
	        		"Layer.Inhib.Layer.Gi": ".9", //
					// "Layer.Inhib.Layer.FF": ".8", //
					// "Layer.Inhib.Layer.FB": ".5", //

					"Layer.Inhib.InhibType" : "KWTA",
					"Layer.Inhib.K_for_WTA" : "1",
					"Layer.Inhib.K_point" : ".95", //was .75
					"Layer.Inhib.Target_diff" : ".2", //was .03

	        		"Layer.Inhib.Pool.On": "false",
	        		"Layer.Inhib.Pool.Gi": "1.8", // .66      }},
				}},
			{Sel: "#Category", Desc: "color output layer turned on for color learning task",
				Params: params.Params{
					"Layer.Act.Clamp.Gain": "2", // originally was .5
					"Layer.OscAmnt" : "0", // was .18
					"Layer.Inhib.InhibType" : "KWTA",
					"Layer.Inhib.K_for_WTA" : "1",
					"Layer.Inhib.K_point" : "0.75", //was .75

				}},
			 


			
		},
		"Sim": &params.Sheet{ // sim params apply to sim category
			{Sel: "Sim", Desc: "best params always finish in this time",
				Params: params.Params{
					"Sim.MaxEpcs": "20",
				}},
			},
	}},

	{Name: "TaskSceneRecall", Desc: "params for the Scene recall task", Sheets: params.Sheets{
		"Network": &params.Sheet{
			{Sel: "Prjn", Desc: "norm and momentum on works better, but wt bal is not better for smaller nets",
				Params: params.Params{
					"Prjn.Learn.XCal.SetLLrn" : "0",
					"Prjn.Learn.XCal.MLrn" : "1",
			}},
			{Sel: "Layer", Desc: "using default 1.8 inhib for all of network + no oscillation",
				Params: params.Params{
					"Layer.OscAmnt" : "0",
			}},
			{Sel: "#Output", Desc: "color output layer turned on for color learning task",
				Params: params.Params{
					"Layer.Act.Clamp.Hard": "true",
			}},

			{Sel: "#Hidden", Desc: "color output layer turned on for color learning task",
				Params: params.Params{
					"Layer.OscAmnt": "0",
			}},
			{Sel: "#Scene", Desc: "color output layer turned on for color learning task",
				Params: params.Params{
					"Layer.OscAmnt": "0",
			}},

			{Sel: "#SceneToHidden", Desc: "default connection (will change for non-shape learning task).",
				Params: params.Params{
					"Prjn.WtScale.Abs": "1",
					"Prjn.WtScale.Rel": "1",
			}},
		},
		"Sim": &params.Sheet{ // sim params apply to sim object
			{Sel: "Sim", Desc: "best params always finish in this time",
				Params: params.Params{
					"Sim.MaxEpcs": "1",
				}},
			},
	}},
}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This structure keeps all relevant
// state information organized and available without having to pass everything around
// as arguments to methods, and provides the core GUI interface (note the view tags
// for the fields which provide hints to how things should be displayed).
type Sim struct {
	Net          *leabra.Network   `view:"no-inline" desc:"the network -- click to view / edit parameters for layers, prjns, etc"`
	Pats         *etable.Table     `view:"no-inline" desc:"the training patterns to use"`
	TrnEpcLog    *etable.Table     `view:"no-inline" desc:"training epoch-level log data"`
	TstEpcLog    *etable.Table     `view:"no-inline" desc:"testing epoch-level log data"`
	TrnTrlLog    *etable.Table     `view:"no-inline" desc:"training trial-level log data"`
	TstTrlLog    *etable.Table     `view:"no-inline" desc:"testing trial-level log data"`
	TstErrLog    *etable.Table     `view:"no-inline" desc:"log of all test trials where errors were made"`
	TstErrStats  *etable.Table     `view:"no-inline" desc:"stats on test trials where errors were made"`
	TrnCycLog    *etable.Table     `view:"no-inline" desc:"training cycle-level log data"`
	TstCycLog    *etable.Table     `view:"no-inline" desc:"testing cycle-level log data"`
	RunLog       *etable.Table     `view:"no-inline" desc:"summary log of each run"`
	RunStats     *etable.Table     `view:"no-inline" desc:"aggregate stats on all runs"`
	Params       params.Sets       `view:"no-inline" desc:"full collection of param sets"`
	ParamSet     string            `desc:"which set of *additional* parameters to use -- always applies Base and optionaly this next if set"`
	ParamValues  map[string]string	   `desc:"values of parameters (used to control parameter values for parameter search)"`
	Tag          string            `desc:"extra tag string to add to any file names output from sim (e.g., weights files, log files, params for run)"`
	MaxRuns      int               `desc:"maximum number of model runs to perform"`
	MaxEpcs      int               `desc:"maximum number of epochs to run per model run"`
	LoopEpcs     int               `desc:"loop number of epochs to run per model run"`
	NZeroStop    int               `desc:"if a positive number, training will stop after this many epochs with zero SSE"`
	TrainEnv     env.FixedTable    `desc:"Training environment -- contains everything about iterating over input / output patterns over training"`
	TestEnv      env.FixedTable    `desc:"Testing environment -- manages iterating over testing"`
	CurrentTask  string 					 `desc:"Name of current task -- useful for logging"`
	CurrentTest  string 					 `desc:"Name of current test (either TestColorAll or TestSceneAll) -- useful for logging"`
	CurrentStimFile  string 					 `desc:"Name of current stim file used"`
	StimulusDir  string			   `desc:"Directory to use to load in the stimulus folders"`
	Time         leabra.Time       `desc:"leabra timing parameters and state"`
	ViewOn       bool              `desc:"whether to update the network view while running"`
	TrainUpdt    leabra.TimeScales `desc:"at what time scale to update the display during training?  Anything longer than Epoch updates at Epoch in this model"`
	TestUpdt     leabra.TimeScales `desc:"at what time scale to update the display during testing?  Anything longer than Epoch updates at Epoch in this model"`
	TestInterval int               `desc:"how often to run through all the test patterns, in terms of training epochs -- can use 0 or -1 for no testing"`
	LayStatNms      []string          `desc:"names of layers to collect more detailed stats on (avg act, etc)"`
	LayErrStatNms   []string          `desc:"names of layers to collect more detailed stats on CosDiff, error and SSE during training"`
	// OscAmnt  		 float64           `def:"0" desc:"describes the amount of oscillation for every layer -- applied when we run a task"`

	// statistics: note use float64 as that is best for etable.Table
	TrlErr        	map[string]float64 `inactive:"+" desc:"1 if trial was error, 0 if correct -- based on SSE = 0 (subject to .5 unit-wise tolerance)"`
	TrlSSE     		map[string]float64 `inactive:"+" desc:"current trial's sum squared error"`
	TrlAvgSSE  		map[string]float64 `inactive:"+" desc:"current trial's average sum squared error"`
	TrlCosDiff 		map[string]float64 `inactive:"+" desc:"current trial's cosine difference"`
	EpcSSE     		map[string]float64 `inactive:"+" desc:"last epoch's total sum squared error"`
	EpcAvgSSE  		map[string]float64 `inactive:"+" desc:"last epoch's average sum squared error (average over trials, and over units within layer)"`
	EpcPctErr     map[string]float64 `inactive:"+" desc:"last epoch's average TrlErr"`
	EpcPctCor     map[string]float64 `inactive:"+" desc:"1 - last epoch's average TrlErr"`
	EpcCosDiff 	  map[string]float64 `inactive:"+" desc:"last epoch's average cosine difference for output layer (a normalized error measure, maximum of 1 when the minus phase exactly matches the plus)"`
	EpcPerTrlMSec float64 `inactive:"+" desc:"how long did the epoch take per trial in wall-clock milliseconds"`
	FirstZero  	  int     `inactive:"+" desc:"epoch at when SSE first went to zero"`
	NZero         int     `inactive:"+" desc:"number of epochs in a row with zero SSE"`


	// internal state - view:"-"
	SumErr       map[string]float64											`view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumSSE       map[string]float64											`view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumAvgSSE    map[string]float64											`view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	SumCosDiff   map[string]float64											`view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	Win          *gi.Window										`view:"-" desc:"main GUI window"`
	NetView      *netview.NetView							`view:"-" desc:"the network viewer"`
	ToolBar      *gi.ToolBar									`view:"-" desc:"the master toolbar"`
	TrnEpcPlot   *eplot.Plot2D								`view:"-" desc:"the training epoch plot"`
	TstEpcPlot   *eplot.Plot2D								`view:"-" desc:"the testing epoch plot"`
	TstTrlPlot   *eplot.Plot2D								`view:"-" desc:"the test-trial plot"`
	TstCycPlot   *eplot.Plot2D								`view:"-" desc:"the test-cycle plot"`
	RunPlot      *eplot.Plot2D								`view:"-" desc:"the run plot"`
	TrnEpcFile   *os.File											`view:"-" desc:"log file"`
	TrnTrlFile   *os.File											`view:"-" desc:"log file for train trials"`
	TstTrlFile   *os.File											`view:"-" desc:"log file for test trials"`
	TrnCycFile   *os.File											`view:"-" desc:"log file for train cycles"`
	TstCycFile   *os.File											`view:"-" desc:"log file for test cycles"`
	RunFile      *os.File											`view:"-" desc:"log file"`
	ValsTsrs     map[string]*etensor.Float32	`view:"-" desc:"for holding layer values"`
	ValsArrays	 map[string]*[]float32				`view:"-" desc:"for holding layer values"`
	SaveWts      bool                    	    `view:"-" desc:"for command-line run only, auto-save final weights after each run"`
	DataDir		 string           						`view:"-" desc:"both command-line run and GUI run, name of the directory to save weights/activations to"`
	NoGui        bool           							`view:"-" desc:"if true, runing in no GUI mode"`
	LogSetParams bool           							`view:"-" desc:"if true, print message for all params that are set"`
	IsRunning    bool           							`view:"-" desc:"true if sim is running"`
	StopNow      bool           							`view:"-" desc:"flag to stop running"`
	NeedsNewRun  bool           							`view:"-" desc:"flag to initialize NewRun if last one finished"`
	DoRunColorTests  	 bool           							`view:"-" desc:"flag to run color tests, set to false to save time"`
	DoRunSceneTests  	 bool           							`view:"-" desc:"flag to run color tests, set to false to save time"`
	RndSeed      int64           							`view:"-" desc:"the current random seed"`
	LastEpcTime  time.Time     						    `view:"-" desc:"timer for last epoch"`
}

// this registers this Sim Type and gives it properties that e.g.,
// prompt for filename for save methods.
var KiT_Sim = kit.Types.AddType(&Sim{}, SimProps)

// TheSim is the overall state for this simulation
var TheSim Sim

// New creates new blank elements and initializes defaults
func (ss *Sim) New() {
	ss.Net = &leabra.Network{}
	ss.Pats = &etable.Table{}
	ss.TrnEpcLog = &etable.Table{}
	ss.TstEpcLog = &etable.Table{}
	ss.TrnTrlLog = &etable.Table{}
	ss.TrnTrlLog.SetMetaData("IsHeaderWritten", "No") // haven't written headers yet
	ss.TstTrlLog = &etable.Table{}
	ss.TstTrlLog.SetMetaData("IsHeaderWritten", "No") // haven't written headers yet
	ss.TrnCycLog = &etable.Table{}
	ss.TrnCycLog.SetMetaData("IsHeaderWritten", "No") // haven't written headers yet
	ss.TstCycLog = &etable.Table{}
	ss.TstCycLog.SetMetaData("IsHeaderWritten", "No") // haven't written headers yet
	ss.RunLog = &etable.Table{}
	ss.RunStats = &etable.Table{}
	ss.Params = ParamSets
	ss.RndSeed = time.Now().UnixNano()
	ss.ViewOn = true
	ss.DoRunColorTests = true
	ss.DoRunSceneTests = true
	ss.TrainUpdt = leabra.Cycle // leabra.Phase
	ss.TestUpdt = leabra.Cycle
	ss.TestInterval = 1
	ss.LayStatNms = []string{"Scene", "Hidden", "Output"}
	ss.LayErrStatNms = []string{"Scene", "Category", "Output"}
	ss.TrlErr = make(map[string]float64)
	ss.TrlSSE = make(map[string]float64)
	ss.TrlAvgSSE = make(map[string]float64)
	ss.TrlCosDiff = make(map[string]float64)
	ss.EpcSSE = make(map[string]float64)
	ss.EpcAvgSSE = make(map[string]float64)
	ss.EpcPctErr = make(map[string]float64)
	ss.EpcPctCor = make(map[string]float64)
	ss.EpcCosDiff = make(map[string]float64)
	ss.SumErr = make(map[string]float64)
	ss.SumSSE = make(map[string]float64)
	ss.SumAvgSSE = make(map[string]float64)
	ss.SumCosDiff = make(map[string]float64)

	ss.StimulusDir = "stim_file"
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Configs
func (ss *Sim) get_color_diff_StimFile_by_samediff_condition() string {
	hiddentooutput_prjn := ss.getPrjn("Hidden", "Output")
	SameDiffCondition := hiddentooutput_prjn.WtInit.SameDiffCondition
	stimfile := fmt.Sprintf("%s/color_diff_stim_%s.dat", ss.StimulusDir, SameDiffCondition)
	// fmt.Println("stimfile", stimfile)
	return stimfile
}

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	//ss.ConfigPats()
	ss.CurrentStimFile = fmt.Sprintf("%s/color_diff_stim.dat", ss.StimulusDir)
	ss.OpenPats()
	ss.ConfigEnv()
	ss.ConfigNet(ss.Net)
	ss.ConfigTrnEpcLog(ss.TrnEpcLog)
	ss.ConfigTstEpcLog(ss.TstEpcLog)
	ss.ConfigTstTrlLog(ss.TstTrlLog)
	ss.ConfigTrnTrlLog(ss.TrnTrlLog)
	ss.ConfigTrnCycLog(ss.TrnCycLog)
	ss.ConfigTstCycLog(ss.TstCycLog)
	ss.ConfigRunLog(ss.RunLog)

	// Reload ss.CurrentStimFile
	// We can't start loading ss.CurrentStimFile from the beginning because
	// we only know the NumOverlapUnits and NumTotalUnits after we run ss.ConfigNet(ss.Net)
	ss.CurrentStimFile = ss.get_color_diff_StimFile_by_samediff_condition()
}

func (ss *Sim) ConfigEnv() {
	if ss.MaxRuns == 0 { // allow user override
		ss.MaxRuns = 10
	}
	if ss.MaxEpcs == 0 { // allow user override
		ss.MaxEpcs = 50
		ss.NZeroStop = 0 // no NZeroStop to not stop early
	}

	ss.TrainEnv.Nm = "TrainEnv"
	ss.TrainEnv.Dsc = "training params and state"
	ss.TrainEnv.Table = etable.NewIdxView(ss.Pats)
	ss.TrainEnv.Validate()
	ss.TrainEnv.Run.Max = ss.MaxRuns // note: we are not setting epoch max -- do that manually

	ss.TestEnv.Nm = "TestEnv"
	ss.TestEnv.Dsc = "testing params and state"
	ss.TestEnv.Table = etable.NewIdxView(ss.Pats)
	ss.TestEnv.Sequential = true
	ss.TestEnv.Validate()

	// note: to create a train / test split of pats, do this:
	// all := etable.NewIdxView(ss.Pats)
	// splits, _ := split.Permuted(all, []float64{.8, .2}, []string{"Train", "Test"})
	// ss.TrainEnv.Table = splits.Splits[0]
	// ss.TestEnv.Table = splits.Splits[1]

	ss.TrainEnv.Init(0)
	ss.TestEnv.Init(0)
}

func (ss *Sim) SetParamSheetValuefromParamValues(ParamSheetDict map[string]string, ParamSheetKey, ParamName string) {
	if val, ok := ss.ParamValues[ParamName]; ok {
		ParamSheetDict[ParamSheetKey] = val
	}
	return
}

func (ss *Sim) ConfigParamValues() {
	if ss.ParamValues == nil { // in GUI mode
		return
	}

	// ss.Params

	// Layer_Params := ss.Params.SetByName("TaskColorRecall").SheetByName("Network").SelByName("Layer").Params
	// ss.SetParamSheetValuefromParamValues(Layer_Params, "Layer.Learn.AvgL.AveLFix", "Layer_ColorRecall_Learn_AvgL_AveLFix")
	//
	// ColorRecall_Prjn_Params := ss.Params.SetByName("TaskColorRecall").SheetByName("Network").SelByName("Prjn").Params
	// ss.SetParamSheetValuefromParamValues(ColorRecall_Prjn_Params, "Prjn.Learn.XCal.DRev", "TaskColorRecall_Prjn_Learn_XCal_DRev")
	//
	// Base_Layer_Params := ss.Params.SetByName("Base").SheetByName("Network").SelByName("Layer").Params
	// Base_Layer_Params["Layer.Learn.AvgL.SetAveL"] = strconv.FormatBool(ss.ParamValues["Base_Layer_Learn_AvgL_SetAveL"].(bool))
	// Base_Layer_Params["Layer.Learn.AvgL.AveLFix"] = strconv.FormatFloat(ss.ParamValues["Base_Layer_Learn_AvgL_AveLFix"].(float64), 'f', 4, 64)

	// Hidden_Layer_Params := ss.Params.SetByName("Base").SheetByName("Network").SelByName("#Hidden").Params
	// Hidden_Layer_Params["Prjn.Learn.AvgL.SetAveL"] = strconv.FormatBool(ss.ParamValues["Hidden_Layer_Learn_AvgL_SetAveL"].(bool))
	// Hidden_Layer_Params["Prjn.Learn.AvgL.AveLFix"] = strconv.FormatFloat(ss.ParamValues["Hidden_Layer_Learn_AvgL_AveLFix"].(float64), 'f', 4, 64)

	// HiddentoOutput_Prjn_Params := ss.Params.SetByName("Base").SheetByName("Network").SelByName("#HiddenToOutput").Params
	// HiddentoOutput_Prjn_Params["Prjn.Learn.XCal.SetAveL"] = strconv.FormatBool(ss.ParamValues["HiddentoOutput_Prjn_Learn_XCal_SetAveL"].(bool))
	// HiddentoOutput_Prjn_Params["Prjn.Learn.XCal.AveLFix"] = strconv.FormatFloat(ss.ParamValues["HiddentoOutput_Prjn_Learn_XCal_AveLFix"].(float64), 'f', 4, 64)




	// Hidden_ColorRecall_Params := ss.Params.SetByName("TaskColorRecall").SheetByName("Network").SelByName("#Hidden").Params
	// ss.SetParamSheetValuefromParamValues(Hidden_ColorRecall_Params, "Layer.OscAmnt", "Hidden_ColorRecall_Layer_OscAmnt")
	// Hidden_ColorRecall_Params["Layer.OscAmnt"] = ss.ParamValues["Hidden_ColorRecall_Layer_OscAmnt"]

	// Scene_ColorRecall_Params := ss.Params.SetByName("TaskColorRecall").SheetByName("Network").SelByName("#Scene").Params
	// ss.SetParamSheetValuefromParamValues(Scene_ColorRecall_Params, "Layer.OscAmnt", "Scene_ColorRecall_Layer_OscAmnt")

	// Scene_ColorRecall_Params := ss.Params.SetByName("TaskColorRecall").SheetByName("Network").SelByName("#Scene").Params
	// Scene_ColorRecall_Params["Layer.OscAmnt"] = ss.ParamValues["Scene_ColorRecall_Layer_OscAmnt"]
	//
	// Output_ColorRecall_Params := ss.Params.SetByName("TaskColorRecall").SheetByName("Network").SelByName("#Output").Params
	// ss.SetParamSheetValuefromParamValues(Output_ColorRecall_Params, "Layer.OscAmnt", "Output_ColorRecall_Layer_OscAmnt")
	// Output_ColorRecall_Params["Layer.OscAmnt"] = ss.ParamValues["Output_ColorRecall_Layer_OscAmnt"]

	// Hidden_ColorRecall_inhib_Params := ss.Params.SetByName("TaskColorRecall").SheetByName("Network").SelByName("#Hidden").Params
	// ss.SetParamSheetValuefromParamValues(Hidden_ColorRecall_inhib_Params, "Layer.Inhib.Layer.Gi", "Hidden_ColorRecall_Layer_gi")
	// Hidden_ColorRecall_inhib_Params["Layer.Inhib.Layer.Gi"] = ss.ParamValues["Hidden_ColorRecall_Layer_gi"]

	// Scene_ColorRecall_inhib_Params := ss.Params.SetByName("TaskColorRecall").SheetByName("Network").SelByName("#Scene").Params
	// ss.SetParamSheetValuefromParamValues(Scene_ColorRecall_inhib_Params, "Layer.Inhib.Layer.Gi", "Scene_ColorRecall_Layer_gi")


	// Output_ColorRecall_inhib_Params := ss.Params.SetByName("TaskColorRecall").SheetByName("Network").SelByName("#Output").Params
	// ss.SetParamSheetValuefromParamValues(Output_ColorRecall_inhib_Params, "Layer.Inhib.Layer.Gi", "Output_ColorRecall_Layer_gi")
	// Output_ColorRecall_inhib_Params["Layer.Inhib.Layer.Gi"] = ss.ParamValues["Output_ColorRecall_Layer_gi"]

	HiddNumOverlapUnits := ss.Params.SetByName("Base").SheetByName("Network").SelByName("Prjn").Params
	ss.SetParamSheetValuefromParamValues(HiddNumOverlapUnits, "Prjn.WtInit.NumOverlapUnits", "HiddNumOverlapUnits")

	same_diff_flag := ss.Params.SetByName("Base").SheetByName("Network").SelByName("#HiddenToOutput").Params
	ss.SetParamSheetValuefromParamValues(same_diff_flag, "Prjn.WtInit.SameDiffCondition", "same_diff_flag")

	// setting lrate for the main projections:
	hiddentohidden := ss.Params.SetByName("TaskColorRecall").SheetByName("Network").SelByName("#HiddenToHidden").Params
	ss.SetParamSheetValuefromParamValues(hiddentohidden, "Prjn.Learn.Lrate", "LRateOverAll")

	outputtohidden := ss.Params.SetByName("TaskColorRecall").SheetByName("Network").SelByName("#OutputToHidden").Params
	ss.SetParamSheetValuefromParamValues(outputtohidden, "Prjn.Learn.Lrate", "LRateOverAll")

	hiddentooutput := ss.Params.SetByName("TaskColorRecall").SheetByName("Network").SelByName("#HiddenToOutput").Params
	ss.SetParamSheetValuefromParamValues(hiddentooutput, "Prjn.Learn.Lrate", "LRateOverAll")

	scenetohidden := ss.Params.SetByName("TaskColorRecall").SheetByName("Network").SelByName("#SceneToHidden").Params
	ss.SetParamSheetValuefromParamValues(scenetohidden, "Prjn.Learn.Lrate", "LRateOverAll")

	hiddentoscene := ss.Params.SetByName("TaskColorRecall").SheetByName("Network").SelByName("#HiddenToScene").Params
	ss.SetParamSheetValuefromParamValues(hiddentoscene, "Prjn.Learn.Lrate", "LRateOverAll")

	categorytohidden := ss.Params.SetByName("TaskColorRecall").SheetByName("Network").SelByName("#CategoryToHidden").Params
	ss.SetParamSheetValuefromParamValues(categorytohidden, "Prjn.Learn.Lrate", "LRateOverAll")

	hiddentocategory := ss.Params.SetByName("TaskColorRecall").SheetByName("Network").SelByName("#HiddenToCategory").Params
	ss.SetParamSheetValuefromParamValues(hiddentocategory, "Prjn.Learn.Lrate", "LRateOverAll")
}


func (ss *Sim) ConfigNet(net *leabra.Network) {
	net.InitName(net, "favila")
	categoryLayer := net.AddLayer2D("Category", 1, 3, emer.Input)
	sceneLayer := net.AddLayer4D("Scene", 2, 1, 1, 3, emer.Input) // used to be 1,3,2,1 when 6 units
	// the product of the first two numbers specifying layer shape is the number of pools
	hiddenLayer := net.AddLayer2D("Hidden", 1, 50, emer.Hidden)
	outputLayer := net.AddLayer2D("Output", 1, 50, emer.Target)

	// use this to position layers relative to each other
	// default is Above, YAlign = Front, XAlign = Center
	sceneLayer.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Category", YAlign: relpos.Front, Space: 10})
	hiddenLayer.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "Scene", YAlign: relpos.Front, XAlign: relpos.Right, XOffset:20, YOffset: 1})
	outputLayer.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "Hidden", YAlign: relpos.Front, XAlign: relpos.Right, YOffset: 1})
	net.BidirConnectLayers(sceneLayer, hiddenLayer, prjn.NewFull())
	// net.BidirConnectLayers(categoryLayer, sceneLayer, prjn.NewFull())
	net.BidirConnectLayers(categoryLayer, hiddenLayer, prjn.NewFull())
	net.BidirConnectLayers(hiddenLayer, outputLayer, prjn.NewFull())
	net.ConnectLayers(hiddenLayer, hiddenLayer, prjn.NewFull(), emer.Lateral)
	// net.ConnectLayers(outputLayer, outputLayer, prjn.NewFull(), emer.Lateral)



	// net.ConnectLayers(sceneLayer, sceneLayer, prjn.NewFull(), emer.Lateral)

	// note: see emergent/prjn module for all the options on how to connect
	// NewFull returns a new prjn.Full connectivity pattern

	// note: can set these to do parallel threaded computation across multiple cpus
	// not worth it for this small of a model, but definitely helps for larger ones
	// if Thread {
	// 	hid2Lay.SetThread(1)
	// 	outLay.SetThread(1)
	// }

	// note: if you wanted to change a layer type from e.g., Target to Compare, do this:
	// outLay.SetType(emer.Compare)
	// that would mean that the output layer doesn't reflect target values in plus phase
	// and thus removes error-driven learning -- but stats are still computed.

	net.Defaults()
	ss.SetParams("Network", ss.LogSetParams) // only set Network params
	err := net.Build()
	if err != nil {
		log.Println(err)
		return
	}
	net.InitWts()
}

////////////////////////////////////////////////////////////////////////////////
// 	    Init, utils

// Init restarts the run, and initializes everything, including network weights
// and resets the epoch log table
func (ss *Sim) Init() {
	ss.NewRndSeed() // remove this if you want the same seed each time you press init.

	rand.Seed(ss.RndSeed)
	ss.ConfigEnv() // re-config env just in case a different set of patterns was
	// selected or patterns have been modified etc

	// It's important that we set the seed after we run ConfigEnv
	// We want to keep the random number generator between the GUI and the Cmdargs() the same

	ss.StopNow = false
	ss.SetParams("", ss.LogSetParams) // all sheets
	ss.ConfigParamValues()
	// ss.TEMPORARY_ORDER_SHUFFLING() // to be removed later
	ss.NewRun()

	ss.UpdateView(true)
}

// NewRndSeed gets a new random seed based on current time -- otherwise uses
// the same random seed for every run
func (ss *Sim) NewRndSeed() {
	ss.RndSeed = time.Now().UnixNano()
	// ss.RndSeed = 1637092089498501011
}

// Temporarily switch up the order of the train and test trials for testing purposes
// To be removed later
func (ss *Sim) TEMPORARY_ORDER_SHUFFLING() {
	// ss.TrainEnv.NewOrder()
	// ss.TestEnv.NewOrder()
	ss.TrainEnv.NewOrder()
	ss.TestEnv.NewOrder()
}

// Counters returns a string of the current counter state
// use tabs to achieve a reasonable formatting overall
// and add a few tabs at the end to allow for expansion..
func (ss *Sim) Counters(train bool) string {
	if train {
		return fmt.Sprintf("Training Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%v\tTask:\t%v\tTest:\t%v\t", ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur, ss.TrainEnv.Trial.Cur, ss.Time.Cycle, ss.TrainEnv.TrialName.Cur, ss.CurrentTask, ss.CurrentTest)
	} else {
		return fmt.Sprintf("Testing Run:\t%d\tEpoch:\t%d\tTrial:\t%d\tCycle:\t%d\tName:\t%v\tTask:\t%v\tTest:\t%v\t", ss.TestEnv.Run.Cur, ss.TestEnv.Epoch.Cur, ss.TestEnv.Trial.Cur, ss.Time.Cycle, ss.TestEnv.TrialName.Cur, ss.CurrentTask, ss.CurrentTest)
	}
}

func (ss *Sim) UpdateView(train bool) {
	if ss.NetView != nil && ss.NetView.IsVisible() {
		ss.NetView.Record(ss.Counters(train))
		// note: essential to use Go version of update when called from another goroutine
		ss.NetView.GoUpdate() // note: using counters is significantly slower..
	}
}

func (ss *Sim) printOscillation() {
	fmt.Printf("%+v\n", ss.Time.Cycle)

	for _, ly := range ss.Net.Layers {

		currLayer := ly.(*leabra.Layer)
						fmt.Println(ly.Name(), currLayer.Inhib.Layer.Gi, currLayer.OscAmnt)
					}
			}

//Runs the oscillation -- updates Gi based on the current cycle number.
// func (ss *Sim) RunOscillation() {
//
// 		for _, ly := range ss.Net.Layers {
//
// 			currLayer := ly.(*leabra.Layer)
// 			var new_inhib float64
//
// 			if currLayer.OscAmnt != 0 {
// 		    if ss.Time.Cycle < 25 { // 1st quarter has no change
// 		      new_inhib = currLayer.BaseGi
// 		      } else if ss.Time.Cycle >= 75 {// 4th quarter has no change
// 		        new_inhib = currLayer.BaseGi
// 		        } else { //second quarter is increased inhibition, 3rd quarter is decreased inhibition.
// 		          new_inhib = calc_oscill(float64(ss.Time.Cycle), currLayer.BaseGi, currLayer.OscAmnt)
// 							currLayer.Inhib.Layer.Gi = float32(new_inhib)
// 		        }
// 		    } else {
//
// 		    }
//
// 		}
//
// }

// note that this runOscillation currently has the oscillation run over 3 quartesr-- q2 - q4.
// originally it just ran q2-q3. When that was the case, there was an extra else if ss.Time.Cycle >= 75 {// 4th quarter has no change
// 		        new_inhib = currLayer.BaseGi
// also periodlength was 2 and not 3.
func (ss *Sim) RunOscillation() {

	for _, ly := range ss.Net.Layers {

		currLayer := ly.(*leabra.Layer)
		var new_inhib float64

		var cycleStartOsc int
		cycleStartOsc = 125

		if currLayer.OscAmnt != 0 {
			if ss.Time.Cycle < cycleStartOsc { // 1st quarter has no change
				new_inhib = currLayer.BaseGi
			} else {
				new_inhib = calc_oscill(float64(ss.Time.Cycle), currLayer.BaseGi, currLayer.OscAmnt, float64(cycleStartOsc))
				currLayer.Inhib.Layer.Gi = float32(new_inhib)
			}
		}
	}

}

//called by RunOscillations, this actually calculates the correct Gi value we want.
func calc_oscill(step_number float64, curr_inhib float64, oscill_amount float64, cycleStartOsc float64) float64 {

	const pi float64 = 3.14159265
	var periodLength float64 = 75 // this was because in the original version with 100 cycles, we wanted oscillations to start at the end of the first quarter.


  new_inhib := curr_inhib + oscill_amount*math.Sin(((2*pi) / periodLength) *(step_number - cycleStartOsc))


	if new_inhib >= curr_inhib {
		new_inhib = curr_inhib
	}
	return new_inhib
}


func sum(array []float32) float32 {
	result := float32(0)
	for _, v := range array {
		result += v
	}
	return result
}

func sumsqdiff(arr1, arr2 []float32) float64 {
	result := float64(0)
	for i := range arr1 {
		result += math.Pow(float64(arr1[i] - arr2[i]), float64(2))
	}
	return result
}

////////////////////////////////////////////////////////////////////////////////
// 	    Running the Network, starting bottom-up..
// Set layer variable
func (ss *Sim) SetHiddenVm() {
	hiddenLayer := ss.Net.LayerByName("Hidden").(leabra.LeabraLayer).AsLeabra()
	for ni := range hiddenLayer.Neurons {
		nrn := &hiddenLayer.Neurons[ni]
		if 32 <= ni && ni <= 33  {
			nrn.Vm = 0.56
		}
	}
}

func (ss *Sim) SetHiddenGe() {
	hiddenLayer := ss.Net.LayerByName("Hidden").(leabra.LeabraLayer).AsLeabra()
	for ni := range hiddenLayer.Neurons {
		nrn := &hiddenLayer.Neurons[ni]
		if 32 <= ni && ni <= 33  {
			nrn.Ge = 0.8
		}
	}
}

// AlphaCyc runs one alpha-cycle (100 msec, 4 quarters)			 of processing.
// External inputs must have already been applied prior to calling,
// using ApplyExt method on relevant layers (see TrainTrial, TestTrial).
// If train is true, then learning DWt or WtFmDWt calls are made.
// Handles netview updating within scope of AlphaCycle
func (ss *Sim) AlphaCyc(train bool) {
	// ss.Win.PollEvents() // this can be used instead of running in a separate goroutine
	viewUpdt := ss.TrainUpdt
	if !train {
		viewUpdt = ss.TestUpdt
	}

	// update prior weight changes at start, so any DWt values remain visible at end
	// you might want to do this less frequently to achieve a mini-batch update
	// in which case, move it out to the TrainTrial method where the relevant
	// counters are being dealt with.
	// if train {
	// 	ss.Net.WtFmDWt()
	// }

	//to see if GUI and cmdline are the same
	// if train {
		// val := *(ss.printlayerVariable("Output", "Ge"))
		// fmt.Println("Ge", train, "Run", ss.TrainEnv.Run.Cur, "Epoch", ss.TrainEnv.Epoch.Cur, "Trial", ss.TrainEnv.TrialName.Cur, "Ge", val)
	// }
	ss.Net.WtFmDWt()

	// val = *(ss.printSynapseVariable("Hidden", "Hidden", "DWt"))
	// fmt.Println("After WtFmDWt train", train, "Run", ss.TrainEnv.Run.Cur, "Epoch", ss.TrainEnv.Epoch.Cur, "Trial", ss.TrainEnv.TrialName.Cur, "Sum of DWt", sum(val))

	ss.Net.AlphaCycInit()
	ss.Time.AlphaCycStart()

	// At beginning of trial, apply targ to ext if plus only learning
	for _, ly := range ss.Net.Layers {
		layer := ly.(*leabra.Layer)
		layer_LearningMP := layer.Learn.LearningMP //
		if (layer_LearningMP == 0) {
			for ni := range layer.Neurons {
				nrn := &layer.Neurons[ni]
				if nrn.IsOff() {
					continue
				}
				if nrn.HasFlag(leabra.NeurHasTarg) { // will be clamped in plus only learning
					nrn.Ext = nrn.Targ
					nrn.SetFlag(leabra.NeurHasExt)
					// fmt.Println("NeurHasTarg", nrn.Ext)
				}
			}
		}
	}

	// hiddenLayer := ss.Net.LayerByName("Hidden").(leabra.LeabraLayer).AsLeabra()


	for qtr := 0; qtr < 4; qtr++ {
		for cyc := 0; cyc < ss.Time.CycPerQtr; cyc++ {
			ss.RunOscillation() // run the code for oscillations-- ie change Gi based on the cycle number.
			// ss.printOscillation()
			// // If you want to check on the oscillations for a specific layer across cycles, uncomment the following code:
			// if !train {
			// 	hiddenLayer := ss.Net.LayerByName("Hidden").(*leabra.Layer)
			// // fmt.Printf("%+v\n",ss.Time)
			// 	fmt.Println("Gi", ss.CurrentTest, ss.TestEnv.TrialName.Cur, ss.Time.Cycle, hiddenLayer.Neurons[0].Gi)
			// }
			// if train {
				// hiddentohidden := ss.getPrjn("Hidden", "Hidden")
				// hiddenLayer := ss.Net.LayerByName("Hidden").(*leabra.Layer)
				// fmt.Println("hiddentohidden", hiddentohidden.Learn.Lrate)
			// }



			ss.Net.Cycle(&ss.Time)

			// if train && (ss.TrainEnv.Epoch.Cur == 0) {
			// 	path := "./hidden_ffi.txt"
			// 	var file, _ = os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
			// 	defer file.Close()


			// 	lpl := hiddenLayer.Pools[0]

			// 	file.WriteString(fmt.Sprintln("epoch", ss.TrainEnv.Epoch.Cur,
			// 					"trial", ss.TrainEnv.TrialName.Cur, "cyc", qtr*ss.Time.CycPerQtr +cyc,
			// 					"hiddenLayer.Inhib.FFi", lpl.Inhib.FFi))
			// 	// save changes
			// 	file.Sync()
			// 	// fmt.Println("cyc", qtr*ss.Time.CycPerQtr + cyc, "hiddenLayer.Inhib.Pool", hiddenLayer.Inhib.Pool.Gi)
			// }
			// if train && (ss.TrainEnv.Epoch.Cur == 0) && (ss.TrainEnv.Trial.Cur == 1)   {
			// 	if 33 <= cyc && cyc < 38 && qtr == 1 {
			// 		ss.SetHiddenGe()
			// 	}
			// 	if 32 == cyc && qtr == 1 {
			// 		ss.SetHiddenVm()
			// 	}

			// }



			if (train) && (ss.TrnCycFile != nil) {
				ss.LogTrnCyc(ss.TrnCycLog, ss.Time.Cycle)
			} else if (ss.TstCycFile != nil) {
				ss.LogTstCyc(ss.TstCycLog, ss.Time.Cycle)
			}
			ss.Time.CycleInc()
			if ss.ViewOn {
				switch viewUpdt {
				case leabra.Cycle:
					if cyc != ss.Time.CycPerQtr-1 { // will be updated by quarter
						ss.UpdateView(train)
					}
				case leabra.FastSpike:
					if (cyc+1)%10 == 0 {
						ss.UpdateView(train)
					}
				}
			}
		}
		ss.Net.QuarterFinal(&ss.Time)
		ss.Time.QuarterInc()
		if ss.ViewOn {
			switch {
			case viewUpdt <= leabra.Quarter:
				ss.UpdateView(train)
			case viewUpdt == leabra.Phase:
				if qtr >= 2 {
					ss.UpdateView(train)
				}
			}
		}
	}

	// val := *ss.printSynapseVariable( "Hidden", "Hidden", "DWt")
	// fmt.Println("Epoch", ss.TrainEnv.Epoch.Cur, "Hidden wt", val )



	if train {
		ss.Net.DWt()
	}
	if ss.ViewOn && viewUpdt == leabra.AlphaCycle {
		ss.UpdateView(train)
	}
	if !train {
		ss.TstCycPlot.GoUpdate() // make sure up-to-date at end
	}
}

// ApplyInputs applies input patterns from given environment.
// It is good practice to have this be a separate method with appropriate
// args so that it can be used for various different contexts
// (training, testing, etc).
func (ss *Sim) ApplyInputs(en env.Env) {
	ss.Net.InitExt() // clear any existing inputs -- not strictly necessary if always
	// going to the same layers, but good practice and cheap anyway

	lays := []string{"Category", "Scene", "Output"}
	for _, lnm := range lays {
		ly := ss.Net.LayerByName(lnm).(*leabra.Layer).AsLeabra()
		pats := en.State(ly.Nm)
		if pats != nil {
			ly.ApplyExt(pats)
		}
	}
}

// TrainTrial runs one trial of training using TrainEnv
func (ss *Sim) TrainTrial() {
	if ss.NeedsNewRun {
		// fmt.Println("Starting new run!!")
		// fmt.Println("ss.CurrentTask", ss.CurrentTask)
		// fmt.Println("ss.TrainEnv.Run.Cur", ss.TrainEnv.Run.Cur)
		// ss.Init()
		// ss.NewRndSeed()
		// fmt.Println("Randseed", ss.RndSeed)
		// rand.Seed(ss.RndSeed)

		// ss.ConfigEnv() // re-config env just in case a different set of patterns was
		// selected or patterns have been modified etc
		// ss.StopNow = false
		// ss.SetParams("", ss.LogSetParams) // all sheets

		// ss.NewRun()
		// ss.UpdateView(true)

		// ss.SaveWeights("savingweightsfilenow.wts")
	}

	ss.TrainEnv.Step() // the Env encapsulates and manages all counter state
	// Key to query counters FIRST because current state is in NEXT epoch
	// if epoch counter has changed
	epc, _, chg := ss.TrainEnv.Counter(env.Epoch)
	if chg {
		ss.LogTrnEpc(ss.TrnEpcLog)
		if ss.ViewOn && ss.TrainUpdt > leabra.AlphaCycle {
			ss.UpdateView(true)
		}
		if ss.TestInterval > 0 && epc%ss.TestInterval == 0 { // note: epc is *next* so won't trigger first time
			ss.TestColorAll()
			ss.TestSceneAll()
		}
		if epc >= ss.MaxEpcs || (ss.NZeroStop > 0 && ss.NZero >= ss.NZeroStop) {
			ss.TrainEnv.Trial.Cur = -1 // Have to init the trial to -1 (to start the next task at trial 0)
			// because ss.TrainEnv.Step() was called at end of last epoch
			ss.StopNow = true
			return
		}

	}

	// HiddenL := ss.Net.LayerByName("Hidden").(leabra.LeabraLayer).AsLeabra()
	// fmt.Println("hidden LrnM", HiddenL.Learn.ActAvg.LrnM)
	//
	// fmt.Println("hidden LrnS", HiddenL.Learn.ActAvg.LrnS)

	// Reset the task type (inputs/targets)
	if ss.CurrentTask == "TaskColorWOOsc" {
		ss.CurrentStimFile = ss.get_color_diff_StimFile_by_samediff_condition()
		ss.OpenPats() // because we want to train without color but test with color
		// ss.SetParamsSet("TaskColorWOOsc", "", ss.LogSetParams) // all sheets: done to reset LearningMP

		categoryLayer := ss.Net.LayerByName("Category").(*leabra.Layer)
		categoryLayer.SetType(emer.Input)
		sceneLayer := ss.Net.LayerByName("Scene").(*leabra.Layer)
		sceneLayer.SetType(emer.Input)
		outputLayer := ss.Net.LayerByName("Output").(*leabra.Layer)
		outputLayer.SetType(emer.Target)

	} else if ss.CurrentTask == "TaskSceneRecall" {
		ss.CurrentStimFile = ss.get_color_diff_StimFile_by_samediff_condition()
		// ss.OpenPats() // because we want the network to guess scene given color and object

		categoryLayer := ss.Net.LayerByName("Category").(*leabra.Layer)
		categoryLayer.SetType(emer.Input)
		sceneLayer := ss.Net.LayerByName("Scene").(*leabra.Layer)
		sceneLayer.SetType(emer.Target)
		outputLayer := ss.Net.LayerByName("Output").(*leabra.Layer)
		outputLayer.SetType(emer.Input)

	} else if ss.CurrentTask == "TaskColorRecall" {
		ss.CurrentStimFile = fmt.Sprintf("%s/color_diff_stim_no_color.dat", ss.StimulusDir)
		ss.OpenPats() // because we want to train without color but test with color
		ss.SetParamsSet("TaskColorRecall", "", ss.LogSetParams) // all sheets

		categoryLayer := ss.Net.LayerByName("Category").(*leabra.Layer)
		categoryLayer.SetType(emer.Input)
		sceneLayer := ss.Net.LayerByName("Scene").(*leabra.Layer)
		sceneLayer.SetType(emer.Input)
		outputLayer := ss.Net.LayerByName("Output").(*leabra.Layer)
		outputLayer.SetType(emer.Target)
	}
	ss.setBaseGi()

	ss.ApplyInputs(&ss.TrainEnv)
	ss.AlphaCyc(true)   // train




	// //Section for ebug, get rid of *** //
 	// SceneL := ss.Net.LayerByName("Scene").(leabra.LeabraLayer).AsLeabra()
	// HiddenL := ss.Net.LayerByName("Hidden").(leabra.LeabraLayer).AsLeabra()
	// hiddenToScene := SceneL.RcvPrjns.SendName("Hidden").(*leabra.Prjn)
	// SceneToHidden := HiddenL.RcvPrjns.SendName("Scene").(*leabra.Prjn)
	// fmt.Println("hidden to face Learn.Xcal.LLrn", hiddenToScene.Learn.XCal.LLrn)
	// fmt.Println("face to hidden Learn.Xcal.LLrn", SceneToHidden.Learn.XCal.LLrn)
	// //Section for ebug, get rid of *** //







	ss.TrialStats(true) // accumulate
	ss.LogTrnTrl(ss.TrnTrlLog)
}


// RunEnd is called at the end of a run -- save weights, record final log, etc here
func (ss *Sim) RunEnd() {
	// Key to query counters FIRST because current state is in NEXT epoch
	// if epoch counter has changed
	_, _, chg := ss.TrainEnv.Counter(env.Epoch)
	// fmt.Println("In RunEnd() _, _, chg", chg)
	if chg {

		// done with training..
		ss.LogRun(ss.RunLog)
		// fmt.Println("In Run_End() prior ss.TrainEnv.Run.Cur", ss.TrainEnv.Run.Cur)
		isMaxRun := ss.TrainEnv.Run.Incr() // This function does two things: 1) increase run number, 2) output boolean of whether it matches max runs
		// fmt.Println("In Run_End() after ss.TrainEnv.Run.Cur", ss.TrainEnv.Run.Cur)

		if isMaxRun { // we are done!
			ss.StopNow = true
			return
		} else {
			ss.NeedsNewRun = true
			return
		}
	}

	if ss.SaveWts {
		fnm := ss.WeightsFileName()
		fmt.Printf("Saving Weights to: %v\n", fnm)
		ss.Net.SaveWtsJSON(gi.FileName(fnm))
	}
}

// NewRun intializes a new run of the model, using the TrainEnv.Run counter
// for the new run value
func (ss *Sim) NewRun() {
	run := ss.TrainEnv.Run.Cur
	ss.TrainEnv.Init(run)
	ss.TestEnv.Init(run)
	ss.Time.Reset()
	ss.Net.InitWts()
	// We use our own functions to Init Wts!
	ss.SaveTesselSpectoJSON()

	// ss.Net.WtBalCtr = 0

	// for _, ly := range ss.Net.Layers {
	// 	if ly.IsOff() {
	// 		continue
	// 	}

	// 	for pi := range ly.(*leabra.Layer).Pools {
	// 		pl := &ly.(*leabra.Layer).Pools[pi]
	// 		pl.ActAvg.ActMAvg = ly.(*leabra.Layer).Inhib.ActAvg.Init
	// 		pl.ActAvg.ActPAvg = ly.(*leabra.Layer).Inhib.ActAvg.Init
	// 		pl.ActAvg.ActPAvgEff = ly.(*leabra.Layer).Inhib.ActAvg.EffInit()
	// 	}

	// 	ly.(*leabra.Layer).InitActAvg()
	// 	ly.(*leabra.Layer).InitActs()
	// 	ly.(*leabra.Layer).CosDiff.Init()
	// }



	var wtsfile gi.FileName
	wtsfile = gi.FileName(ss.DataDir + "/tesselspec.wts")

	ss.LoadWeights(wtsfile)
	// ss.ReloadStimFiles()

	ss.Net.InitActs()
	ss.InitStats()
	ss.TrnEpcLog.SetNumRows(0)
	ss.TstEpcLog.SetNumRows(0)
	ss.NeedsNewRun = false
}

// InitStats initializes all the statistics, especially important for the
// cumulative epoch stats -- called at start of new run
func (ss *Sim) InitStats() {
	// accumulators

	ss.FirstZero = -1
	ss.NZero = 0
	// clear rest just to make Sim look initialized

	for _, lnm := range ss.LayErrStatNms {
		ss.SumErr[lnm] = 0
		ss.SumSSE[lnm] = 0
		ss.SumAvgSSE[lnm] = 0
		ss.SumCosDiff[lnm] = 0
		ss.TrlErr[lnm] = 0
		ss.TrlSSE[lnm] = 0
		ss.TrlAvgSSE[lnm] = 0
		ss.TrlCosDiff[lnm] = 0
		ss.EpcSSE[lnm] = 0
		ss.EpcAvgSSE[lnm] = 0
		ss.EpcPctErr[lnm] = 0
		ss.EpcPctCor[lnm] = 0
		ss.EpcCosDiff[lnm] = 0
	}
}

// TrialStats computes the trial-level statistics and adds them to the epoch accumulators if
// accum is true.  Note that we're accumulating stats here on the Sim side so the
// core algorithm side remains as simple as possible, and doesn't need to worry about
// different time-scales over which stats could be accumulated etc.
// You can also aggregate directly from log data, as is done for testing stats
func (ss *Sim) TrialStats(accum bool) {
	// if ss.CurrentTask == "TaskSceneRecall" {
	// 	sceneLayer := ss.Net.LayerByName("Scene").(leabra.LeabraLayer).AsLeabra()
	// 	ss.TrlCosDiff = float64(sceneLayer.CosDiff.Cos)
	// 	ss.TrlSSE, ss.TrlAvgSSE = sceneLayer.MSE(0.5) // 0.5 = per-unit tolerance -- right side of .5
	// } else {
	// 	outputLayer := ss.Net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()
	// 	ss.TrlCosDiff = float64(outputLayer.CosDiff.Cos)
	// 	ss.TrlSSE, ss.TrlAvgSSE = outputLayer.MSE(0.5) // 0.5 = per-unit tolerance -- right side of .5
	// }

	for _, lnm := range ss.LayErrStatNms {
		Layer := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		ss.TrlCosDiff[lnm] = float64(Layer.CosDiff.Cos)
		ss.TrlSSE[lnm], ss.TrlAvgSSE[lnm] = Layer.MSE(0.5) // 0.5 = per-unit tolerance -- right side of .5
		if ss.TrlSSE[lnm] > 0 {
			ss.TrlErr[lnm] = 1
		} else {
			ss.TrlErr[lnm] = 0
		}

		if accum {
			ss.SumErr[lnm] += ss.TrlErr[lnm]
			ss.SumSSE[lnm] += ss.TrlSSE[lnm]
			ss.SumAvgSSE[lnm] += ss.TrlAvgSSE[lnm]
			ss.SumCosDiff[lnm] += ss.TrlCosDiff[lnm]
		}
	}
}

// TrainEpoch runs training trials for remainder of this epoch
func (ss *Sim) TrainEpoch() {
	ss.StopNow = false
	curEpc := ss.TrainEnv.Epoch.Cur
	for {
		ss.TrainTrial()
		if ss.StopNow || ss.TrainEnv.Epoch.Cur != curEpc {
			break
		}
	}
	ss.Stopped()
}

// TrainRun runs training trials for remainder of run
func (ss *Sim) TrainRun() {
	ss.StopNow = false
	curRun := ss.TrainEnv.Run.Cur
	for {
		ss.TrainTrial()
		if ss.StopNow || ss.TrainEnv.Run.Cur != curRun {
			break
		}
	}
	ss.Stopped()
}

// VR custom code to set an initial Gi Value. This is only needed because the oscillations change
// Layer.Inhib.Layer.Gi at every cycle, and an initial Gi value is needed in order to calculate the sinusoidal function.
// If we used Layer.Inhib.Layer.Gi, it wouldn't a good input, since the value is always changing!.
//Run this function at the beginning of every task, run this code to set this BaseGI to be the same as the Inhib.Layer.Gi.
func (ss *Sim) setBaseGi() {
	for _, ly := range ss.Net.Layers {
		// fmt.Printf("%+v\n",ly)
		currLayer := ly.(*leabra.Layer)
		// currLayerName := currLayer.Nm
		currLayerInhib := currLayer.Inhib.Layer.Gi
		currLayer.BaseGi = float64(currLayerInhib)
	}
}

// Helper function for tasks: test all items & save weights and activations
func (ss *Sim) TestandSaveAfterTask() {
	// Run through every item once
	fmt.Println("Running test after task!!")
	// This is brittle, is there a way to get the number of test items?
	var num_tests int
	// if task == "TaskRetrievalPractice" {
	// 	num_tests = 1
	// } else if task == "TaskRestudy" {
	// 	num_tests = 3
	// } else {
	// 	num_tests = 4
	// }

	num_tests = 4
	for idx := 0; idx < num_tests; idx++ {
		// test the item
		if !ss.IsRunning {
			ss.IsRunning = true
			fmt.Printf("testing index: %v\n", idx)
			ss.TestItem(idx)
			ss.IsRunning = false
		}
	}
}



func check(e error) {
    if e != nil {
        panic(e)
    }
}

func(ss *Sim) WriteTaskParams() {
	paramstring := ss.Net.AllParams()
	fnm := ss.LogFileName("Params_" + ss.CurrentTask)
	paramfile, err := os.Create(fnm)

	if err != nil {
		log.Println(err)
		paramfile = nil
	} else {
		// fmt.Printf("Saving task params log to: %v\n", fnm)
		defer paramfile.Close()
	}
	_, err = paramfile.WriteString(paramstring)
	check(err)

}

func (ss *Sim) TaskColorWOOsc() {
	ss.CurrentStimFile = ss.get_color_diff_StimFile_by_samediff_condition()
	ss.OpenPats()
	ss.SetParamsSet("TaskColorWOOsc", "", ss.LogSetParams) // all sheets
	ss.setBaseGi()
	ss.CurrentTask = "TaskColorWOOsc"
	ss.WriteTaskParams()
	// fmt.Println("My maxepcs is!", ss.MaxEpcs, "with task", ss.CurrentTask)






	// ss.ConfigEnv()

	// Set the layer of oscillation like in GUI
	// overrides the ParamSet
	// if !(ss.NoGui) {
	// 	for _, ly := range ss.Net.Layers {
	// 		ly.(*leabra.Layer).OscAmnt = ss.OscAmnt
	// 	}
	// }


	categoryLayer := ss.Net.LayerByName("Category").(*leabra.Layer)
	categoryLayer.SetType(emer.Input)
	sceneLayer := ss.Net.LayerByName("Scene").(*leabra.Layer)
	sceneLayer.SetType(emer.Input)
	outputLayer := ss.Net.LayerByName("Output").(*leabra.Layer)
	outputLayer.SetType(emer.Target)

	ss.StopNow = false
	curRun := ss.TrainEnv.Run.Cur

	for {
		ss.TrainTrial()

		if ss.StopNow || ss.TrainEnv.Run.Cur != curRun {
			break
		}
	}
	ss.Stopped()

}



// TaskShapes is based on TrainRuns, and runs training trials for remainder of run
func (ss *Sim) TaskSceneRecall() {
	ss.CurrentTask = "TaskSceneRecall"

	categoryLayer := ss.Net.LayerByName("Category").(*leabra.Layer)
	categoryLayer.SetType(emer.Input)
	outputLayer := ss.Net.LayerByName("Output").(*leabra.Layer)
	outputLayer.SetType(emer.Input)
	sceneLayer := ss.Net.LayerByName("Scene").(*leabra.Layer)
	sceneLayer.SetType(emer.Target)


	ss.CurrentStimFile = ss.get_color_diff_StimFile_by_samediff_condition()
	ss.OpenPats()
	ss.SetParamsSet("TaskSceneRecall", "", ss.LogSetParams) // all sheets
	ss.setBaseGi()

	ss.WriteTaskParams()
	// fmt.Println("My maxepcs is!", ss.MaxEpcs, "with task", ss.CurrentTask)


	// ss.ConfigEnv()

	// Set the layer of oscillation like in GUI
	// overrides the ParamSet
	// if !(ss.NoGui) {
	// 	for _, ly := range ss.Net.Layers {
	// 		ly.(*leabra.Layer).OscAmnt = ss.OscAmnt
	// 	}
	// }









	ss.StopNow = false
	curRun := ss.TrainEnv.Run.Cur
	for {
		ss.TrainTrial()
		if ss.StopNow || ss.TrainEnv.Run.Cur != curRun {
			break
		}
	}
	ss.Stopped()

}

// TaskShapes is based on TrainRuns, and runs training trials for remainder of run
func (ss *Sim) TaskColorRecall() {
	ss.CurrentTask = "TaskColorRecall"
 // fmt.Println(ss.RndSeed)
	categoryLayer := ss.Net.LayerByName("Category").(*leabra.Layer)
  	categoryLayer.SetType(emer.Input)
	outputLayer := ss.Net.LayerByName("Output").(*leabra.Layer)
	outputLayer.SetType(emer.Target)
	sceneLayer := ss.Net.LayerByName("Scene").(*leabra.Layer)
	sceneLayer.SetType(emer.Input)

	ss.CurrentStimFile = fmt.Sprintf("%s/color_diff_stim_no_color.dat", ss.StimulusDir)
	ss.OpenPats()
	ss.SetParamsSet("TaskColorRecall", "", ss.LogSetParams) // all sheets
	ss.setBaseGi()


	ss.WriteTaskParams()
	// fmt.Println("My maxepcs is!", ss.MaxEpcs, "with task", ss.CurrentTask)


	// ss.ConfigEnv()

	// Set the layer of oscillation like in GUI
	// overrides the ParamSet
	// if !(ss.NoGui) {
	// 	for _, ly := range ss.Net.Layers {
	// 		ly.(*leabra.Layer).OscAmnt = ss.OscAmnt
	// 	}
	// }








	ss.StopNow = false
	curRun := ss.TrainEnv.Run.Cur
	for {
		ss.CurrentTask = "TaskColorRecall"
		ss.TrainTrial()
		if ss.StopNow || ss.TrainEnv.Run.Cur != curRun {
			break
		}
	}
	ss.Stopped()

}

func (ss *Sim) TaskInterleaveStudyAndSceneRecall() {

	for i := 0; i < ss.LoopEpcs; i++ {
			ss.TaskColorWOOsc()
			ss.TaskSceneRecall()
	}

}


// TaskRunAll implements a function for the GUI that runs all 4 tasks sequentially.
func (ss *Sim) TaskRunAll() {
	ss.NewRndSeed()
	rand.Seed(ss.RndSeed)

	// ss.ConfigEnv() // re-config env just in case a different set of patterns was
	// selected or patterns have been modified etc
	ss.SetParams("", ss.LogSetParams) // all sheets
	ss.ConfigParamValues()
	// ss.TEMPORARY_ORDER_SHUFFLING() // to be removed later
	ss.TrainEnv.NewOrder() // shuffling to match counter of CmdArg and GUI
	ss.TestEnv.NewOrder() // shuffling to match counter of CmdArg and GUI
	ss.NewRun()

	// ss.TaskColorWOOsc()
	ss.CurrentTask = "TaskColorWOOsc"
	ss.SetParamsSet("TaskColorRecall", "", ss.LogSetParams) // all sheets
	ss.TestColorAll()
	ss.TestSceneAll()



	// ss.TaskSceneRecall()
	// ss.TaskInterleaveStudyAndSceneRecall()



	ss.TaskColorRecall()

	// Run a test
	ss.StopNow = false
	// ss.TestAll()

	// ss.TaskShapes()
	// fmt.Println("Finished task shapes!")
	// // Save weights and activations for each test item?
	// // fmt.Println("Filename", task + "_testitem_" + strconv.Itoa(idx) + "start")
	// fnm := ss.WeightsFileName("color"  + "start")
	// fmt.Printf("Saving Weights to: %v\n", fnm)
	// ss.Net.SaveWtsJSON(gi.FileName(fnm))
	// ss.TaskColorRecall()
	// fmt.Println("Finished task color study!")
	//
	// fnm = ss.WeightsFileName("retrieve"  + "start")
	// fmt.Printf("Saving Weights to: %v\n", fnm)
	// ss.Net.SaveWtsJSON(gi.FileName(fnm))
	// ss.TaskRetrievalPractice()
	// fmt.Println("Finished task retrieval practice!")
	//
	// fnm = ss.WeightsFileName("restudy"  + "start")
	// fmt.Printf("Saving Weights to: %v\n", fnm)
	// ss.Net.SaveWtsJSON(gi.FileName(fnm))
	// ss.TaskRestudy()

	// End Run
	ss.RunEnd()
}

// Train runs the full training from this point onward
func (ss *Sim) Train() {
	ss.StopNow = false
	for {
		ss.TrainTrial()
		if ss.StopNow {
			break
		}
	}
	ss.Stopped()
}

// Stop tells the sim to stop running
func (ss *Sim) Stop() {
	ss.StopNow = true
}

// Stopped is called when a run method stops running -- updates the IsRunning flag and toolbar
func (ss *Sim) Stopped() {
	ss.IsRunning = false
	if ss.Win != nil {
		vp := ss.Win.WinViewport2D()
		if ss.ToolBar != nil {
			ss.ToolBar.UpdateActions()
		}
		vp.SetNeedsFullRender()
	}
}

// SaveWeights saves the network weights -- when called with giv.CallMethod
// it will auto-prompt for filename
func (ss *Sim) SaveWeights(filename gi.FileName) {
	ss.Net.SaveWtsJSON(filename)
}

// SaveWeights saves the network weights -- when called with giv.CallMethod
// it will auto-prompt for filename
func (ss *Sim) LoadWeights(filename gi.FileName) {
	ss.Net.OpenWtsJSON(filename)
}

////////////////////////////////////////////////////////////////////////////////////////////
// Testing

// TestTrial runs one trial of testing -- always sequentially presented inputs
func (ss *Sim) TestTrial(returnOnChg bool) {
	ss.TestEnv.Step()

	// Query counters FIRST
	_, _, chg := ss.TestEnv.Counter(env.Epoch)
	if chg {
		if ss.ViewOn && ss.TestUpdt > leabra.AlphaCycle {
			ss.UpdateView(false)
		}
		ss.LogTstEpc(ss.TstEpcLog)
		if returnOnChg {
			return
		}
	}

	ss.ApplyInputs(&ss.TestEnv)
	ss.AlphaCyc(false)   // !train

	ss.TrialStats(false) // !accumulate
	ss.LogTstTrl(ss.TstTrlLog)
	// ss.AnalyzeTstTrl()
}

// TestItem tests given item which is at given index in test item list
func (ss *Sim) TestItem(idx int) {
	cur := ss.TestEnv.Trial.Cur
	ss.TestEnv.Trial.Cur = idx
	ss.TestEnv.SetTrialName()
	ss.ApplyInputs(&ss.TestEnv)
	ss.AlphaCyc(false)   // !train
	ss.TrialStats(false) // !accumulate
	ss.TestEnv.Trial.Cur = cur
}

func (ss *Sim) SetUpTestTrial(StimFileName, ParamSetName, TestName string) {
	ss.CurrentStimFile = StimFileName
	ss.OpenPats() // so we always test the right pattern


 /// SHOULD THIS BE ADDED?
 	ss.SetParamsSet(ParamSetName, "", ss.LogSetParams) // all sheets

	for _, ly := range ss.Net.Layers {
		ly.(*leabra.Layer).Learn.LearningMP = 1 // so it's minus plus always.
	}

	for _, ly := range ss.Net.Layers {
		ly.(*leabra.Layer).OscAmnt = 0
	}
///

	ss.CurrentTest = TestName
}
// TestAll runs through the full set of testing items
func (ss *Sim) TestColorAll() {
	if ss.DoRunColorTests == false {
		return
	}

	ss.SetUpTestTrial(ss.get_color_diff_StimFile_by_samediff_condition(), "TaskColorRecall", "TestColorAll")

	ss.TestEnv.Init(ss.TrainEnv.Run.Cur)

	categoryLayer := ss.Net.LayerByName("Category").(*leabra.Layer)
	categoryLayer.SetType(emer.Input)
	sceneLayer := ss.Net.LayerByName("Scene").(*leabra.Layer)
	sceneLayer.SetType(emer.Input)
	outputLayer := ss.Net.LayerByName("Output").(*leabra.Layer)
	outputLayer.SetType(emer.Target)


	for {
		ss.TestTrial(true) // return on change -- don't wrap
		_, _, chg := ss.TestEnv.Counter(env.Epoch)
		if chg || ss.StopNow {
			break
		}
	}


	ss.CurrentTest = ""
}


func (ss *Sim) TestSceneAll() {
	if ss.DoRunSceneTests == false {
		return
	}

	ss.SetUpTestTrial(ss.get_color_diff_StimFile_by_samediff_condition(), "TaskColorRecall", "TestSceneAll")

	ss.TestEnv.Init(ss.TrainEnv.Run.Cur)

	categoryLayer := ss.Net.LayerByName("Category").(*leabra.Layer)
	categoryLayer.SetType(emer.Input)
	sceneLayer := ss.Net.LayerByName("Scene").(*leabra.Layer)
	sceneLayer.SetType(emer.Target)
	outputLayer := ss.Net.LayerByName("Output").(*leabra.Layer)
	outputLayer.SetType(emer.Input)

	for {
		ss.TestTrial(true) // return on change -- don't wrap
		_, _, chg := ss.TestEnv.Counter(env.Epoch)
		if chg || ss.StopNow {
			break
		}
	}

	ss.CurrentTest = ""
}


// RunTestAll runs through the full set of testing items, has stop running = false at end -- for gui
func (ss *Sim) RunTestAllPlusOnly() {

	fmt.Println("TEST")

	outputLayer := ss.Net.LayerByName("Output").(*leabra.Layer)
	outputLayer.SetType(emer.Hidden)
	ss.CurrentStimFile = fmt.Sprintf("%s/color_diff_stim_no_color.dat", ss.StimulusDir)
	ss.OpenPats()

	ss.StopNow = false
	ss.TestColorAll()
	ss.TestSceneAll()
	ss.Stopped()
	outputLayer.SetType(emer.Target)



}


// RunTestAll runs through the full set of testing items, has stop running = false at end -- for gui
func (ss *Sim) RunTestAll() {
	ss.StopNow = false
	ss.TestColorAll()
	ss.TestSceneAll()
	ss.Stopped()
}

/////////////////////////////////////////////////////////////////////////
//   Params setting

// ParamsName returns name of current set of parameters
func (ss *Sim) ParamsName() string {
	if ss.ParamSet == "" {
		return "Base"
	}
	return ss.ParamSet
}

// SetParams sets the params for "Base" and then current ParamSet.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func (ss *Sim) SetParams(sheet string, setMsg bool) error {
	if sheet == "" {
		// this is important for catching typos and ensuring that all sheets can be used
		ss.Params.ValidateSheets([]string{"Network", "Sim"})
	}
	err := ss.SetParamsSet("Base", sheet, setMsg)
	if ss.ParamSet != "" && ss.ParamSet != "Base" {
		err = ss.SetParamsSet(ss.ParamSet, sheet, setMsg)
	}
	return err
}

// SetParamsSet sets the params for given params.Set name.
// If sheet is empty, then it applies all avail sheets (e.g., Network, Sim)
// otherwise just the named sheet
// if setMsg = true then we output a message for each param that was set.
func (ss *Sim) SetParamsSet(setNm string, sheet string, setMsg bool) error {
	pset, err := ss.Params.SetByNameTry(setNm)
	if err != nil {
		return err
	}
	if sheet == "" || sheet == "Network" {
		netp, ok := pset.Sheets["Network"]
		if ok {
			ss.Net.ApplyParams(netp, setMsg)
		}
	}

	if sheet == "" || sheet == "Sim" {
		simp, ok := pset.Sheets["Sim"]
		if ok {
			simp.Apply(ss, setMsg)
		}
	}
	// note: if you have more complex environments with parameters, definitely add
	// sheets for them, e.g., "TrainEnv", "TestEnv" etc
	return err
}

func (ss *Sim) ConfigPats() {
	dt := ss.Pats
	dt.SetMetaData("name", "TrainPats")
	dt.SetMetaData("desc", "Training patterns")
	dt.SetFromSchema(etable.Schema{
		{"Name", etensor.STRING, nil, nil},
		{"Input", etensor.FLOAT32, []int{5, 5}, []string{"Y", "X"}},
		{"Output", etensor.FLOAT32, []int{5, 5}, []string{"Y", "X"}},
	}, 25)

	patgen.PermutedBinaryRows(dt.Cols[1], 6, 1, 0)
	patgen.PermutedBinaryRows(dt.Cols[2], 6, 1, 0)
	dt.SaveCSV("color_diff_stimuli_gen.dat", ',', true)
}

func (ss *Sim) OpenPats() {
	dt := ss.Pats
	dt.SetMetaData("name", "TrainPats")
	dt.SetMetaData("desc", "Training patterns")
	err := dt.OpenCSV(gi.FileName(ss.CurrentStimFile), etable.Tab)
	if err != nil {
		log.Println(err)
	}
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Logging

// ValsTsr gets value tensor of given name, creating if not yet made
func (ss *Sim) ValsTsr(name string) *etensor.Float32 {
	if ss.ValsTsrs == nil {
		ss.ValsTsrs = make(map[string]*etensor.Float32)
	}
	tsr, ok := ss.ValsTsrs[name]
	if !ok {
		tsr = &etensor.Float32{}
		ss.ValsTsrs[name] = tsr
	}
	return tsr
}

// ValsArray gets value array of given name, creating if not yet made
func (ss *Sim) ValsArray(name string) *[]float32 {
	if ss.ValsArrays == nil {
		ss.ValsArrays = make(map[string]*[]float32)
	}
	arr, ok := ss.ValsArrays[name]
	if !ok {
		arr = &[]float32{}
		ss.ValsArrays[name] = arr
	}
	return arr
}

// RunName returns a name for this run that combines Tag and Params -- add this to
// any file names that are saved.
func (ss *Sim) RunName() string {
	if ss.Tag != "" {
		return ss.Tag + "_" + ss.ParamsName()
	} else {
		return ss.ParamsName()
	}
}

// RunEpochName returns a string with the run and epoch numbers with leading zeros, suitable
// for using in weights file names.  Uses 3, 5 digits for each.
func (ss *Sim) RunEpochName(run, epc int) string {
	return fmt.Sprintf("%03d_%05d", run, epc)
}

// WeightsFileName returns default current weights file name
func (ss *Sim) WeightsFileName() string {
	if ss.CurrentTask != "" {
		return ss.DataDir + "/" + ss.Net.Nm + "_" + ss.RunName() + "_" + ss.RunEpochName(ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur) + "_" + ss.CurrentTask + ".wts"
	} else {
		return ss.DataDir + "/" + ss.Net.Nm + "_" + ss.RunName() + "_" + ss.RunEpochName(ss.TrainEnv.Run.Cur, ss.TrainEnv.Epoch.Cur) + ".wts"
	}
}

// LogFileName returns default log file name
func (ss *Sim) LogFileName(lognm string) string {
	return ss.DataDir + "/" + ss.Net.Nm + "_" + ss.RunName() + "_" + lognm + ".csv"
}


//////////////////////////////////////////////
//  TrnEpcLog

// LogTrnEpc adds data from current epoch to the TrnEpcLog table.
// computes epoch averages prior to logging.
func (ss *Sim) LogTrnEpc(dt *etable.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)

	// hiddenLayer := ss.Net.LayerByName("Hidden").(*leabra.Layer)
	// outputLayer := ss.Net.LayerByName("Output").(*leabra.Layer)

	epc := ss.TrainEnv.Epoch.Prv           // this is triggered by increment so use previous value
	nt := float64(ss.TrainEnv.Table.Len()) // number of trials in view
	for _, lnm := range ss.LayErrStatNms {
		ss.EpcSSE[lnm] = ss.SumSSE[lnm] / nt
		ss.SumSSE[lnm] = 0
		ss.EpcAvgSSE[lnm] = ss.SumAvgSSE[lnm] / nt
		ss.SumAvgSSE[lnm] = 0
		ss.EpcPctErr[lnm] = float64(ss.SumErr[lnm]) / nt
		ss.SumErr[lnm] = 0
		ss.EpcPctCor[lnm] = 1 - ss.EpcPctErr[lnm]
		ss.EpcCosDiff[lnm] = ss.SumCosDiff[lnm] / nt
		ss.SumCosDiff[lnm] = 0
		if ss.FirstZero < 0 && ss.EpcPctErr[lnm] == 0 {
			ss.FirstZero = epc
		}
		if ss.EpcPctErr[lnm] == 0 {
			ss.NZero++
		} else {
			ss.NZero = 0
		}
	}



	if ss.LastEpcTime.IsZero() {
		ss.EpcPerTrlMSec = 0
	} else {
		iv := time.Now().Sub(ss.LastEpcTime)
		ss.EpcPerTrlMSec = float64(iv) / (nt * float64(time.Millisecond))
	}
	ss.LastEpcTime = time.Now()

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellString("CurrentTask", row, ss.CurrentTask)
	for _, lnm := range ss.LayErrStatNms {
		dt.SetCellFloat(lnm+"_SSE", row, ss.EpcSSE[lnm])
		dt.SetCellFloat(lnm+"_AvgSSE", row, ss.EpcAvgSSE[lnm])
		dt.SetCellFloat(lnm+"_PctErr", row, ss.EpcPctErr[lnm])
		dt.SetCellFloat(lnm+"_PctCor", row, ss.EpcPctCor[lnm])
		dt.SetCellFloat(lnm+"_CosDiff", row, ss.EpcCosDiff[lnm])
	}

	dt.SetCellFloat("PerTrlMSec", row, ss.EpcPerTrlMSec)

	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		dt.SetCellFloat(ly.Nm+"_ActAvg", row, float64(ly.Pools[0].ActAvg.ActPAvgEff))
	}
	// note: essential to use Go version of update when called from another goroutine
	ss.TrnEpcPlot.GoUpdate()
	if ss.TrnEpcFile != nil {
		if ss.TrainEnv.Run.Cur == 0 && epc == 0 {
			dt.WriteCSVHeaders(ss.TrnEpcFile, etable.Tab)
		}
		dt.WriteCSVRow(ss.TrnEpcFile, row, etable.Tab)
	}
}

func (ss *Sim) ConfigTrnEpcLog(dt *etable.Table) {
	dt.SetMetaData("name", "TrnEpcLog")
	dt.SetMetaData("desc", "Record of performance over epochs of training")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"CurrentTask", etensor.STRING, nil, nil},
	}

	for _, lnm := range ss.LayErrStatNms {
		sch = append(sch, etable.Column{lnm + "_SSE", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_AvgSSE", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_PctErr", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_PctCor", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_CosDiff", etensor.FLOAT64, nil, nil})
	}
	sch = append(sch, etable.Schema{
		{"PerTrlMSec", etensor.FLOAT64, nil, nil},
	}...)
	for _, lnm := range ss.LayStatNms {
		sch = append(sch, etable.Column{lnm + "_ActAvg", etensor.FLOAT64, nil, nil})
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTrnEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Leabra Random Associator 25 Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	for _, lnm := range ss.LayErrStatNms {
		plt.SetColParams(lnm + "_SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
		plt.SetColParams(lnm + "_AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
		plt.SetColParams(lnm + "_PctErr", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1) // default plot
		plt.SetColParams(lnm + "_PctCor", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1) // default plot
		plt.SetColParams(lnm + "_CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	}

	plt.SetColParams("PerTrlMSec", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)

	for _, lnm := range ss.LayStatNms {
		plt.SetColParams(lnm+"_ActAvg", eplot.Off, eplot.FixMin, 0, eplot.FixMax, .5)
	}
	return plt
}
//////////////////////////////////////////////
//  TrnTrlLog
// LogTrnTrl adds data from current trial to the TrnTrlLog table.
// log always contains number of training items

// Print layer variable
func (ss *Sim) printlayerVariable(layer, variableName string) *[]float32 {
	Layer := ss.Net.LayerByName(layer).(leabra.LeabraLayer).AsLeabra()
	vals := make([]float32, len(Layer.Neurons))
	Layer.UnitVals(&vals, variableName)
	// fmt.Println(ss.TrainEnv.TrialName.Cur, variableName, layer, vals) // vals[(49*28+22):(49*29+40)])
	return &vals
}

// Print synapse variable
func (ss *Sim) printSynapseVariable(sendLayerName, recvLayerName, variableName string) *[]float32 {
	recvLay := ss.Net.LayerByName(recvLayerName).(leabra.LeabraLayer).AsLeabra()
	vals := make([]float32, 0)
	for _, prjn := range recvLay.RcvPrjns {
		if prjn.SendLay().(*leabra.Layer).Nm == sendLayerName {
			proj := prjn.(*leabra.Prjn)
			vals = make([]float32, len(proj.Syns))
			proj.SynVals(&vals, variableName)
			// fmt.Println(ss.TrainEnv.TrialName.Cur, variableName, sendLayerName, "->", recvLayerName, vals) // vals[(49*28+22):(49*29+40)])
		}
	}
	return &vals
}

// Get projection variable
func (ss *Sim) getPrjn(sendLayerName, recvLayerName string) *leabra.Prjn {
	recvLay := ss.Net.LayerByName(recvLayerName).(leabra.LeabraLayer).AsLeabra()
	proj := &leabra.Prjn{}
	for _, prjn := range recvLay.RcvPrjns {
		if prjn.SendLay().(*leabra.Layer).Nm == sendLayerName {
			proj = prjn.(*leabra.Prjn)
			// fmt.Println(ss.TrainEnv.TrialName.Cur, variableName, sendLayerName, "->", recvLayerName, vals) // vals[(49*28+22):(49*29+40)])
		}
	}
	return proj
}


// Save synapse variable into ss.ValsTsr("Input")
func (ss *Sim) saveSynapseVariable(sendLayerName, recvLayerName, variableName string) {
	recvLay := ss.Net.LayerByName(recvLayerName).(leabra.LeabraLayer).AsLeabra()
	ivt := ss.ValsTsr("Input")
	for _, prjn := range recvLay.RcvPrjns {
		if prjn.SendLay().(*leabra.Layer).Nm == sendLayerName {
			proj := prjn.(*leabra.Prjn)
			vals := make([]float32, len(proj.Syns))
			proj.SynVals(&vals, variableName)
			ivt.SetShape([]int{len(proj.Syns)}, nil, nil)
			for i, val := range vals {
				ivt.SetFloat1D(i, float64(val))
			}
		}
	}
}

func (ss *Sim) LogTrnTrl(dt *etable.Table) {
	categoryLayer := ss.Net.LayerByName("Category").(leabra.LeabraLayer).AsLeabra()
	sceneLayer := ss.Net.LayerByName("Scene").(leabra.LeabraLayer).AsLeabra()
	hiddenLayer := ss.Net.LayerByName("Hidden").(leabra.LeabraLayer).AsLeabra()
	outputLayer := ss.Net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()

	epc := ss.TrainEnv.Epoch.Cur

	trl := ss.TrainEnv.Trial.Cur
	// row := trl

	// if dt.Rows <= row {
	// 	dt.SetNumRows(row + 1)
	// }
	row := dt.Rows
	dt.SetNumRows(row + 1)

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellString("CurrentTask", row, ss.CurrentTask)
	// dt.SetCellString("CurrentTest", row, ss.CurrentTest)
	dt.SetCellFloat("Trial", row, float64(trl))
	dt.SetCellString("TrialName", row, ss.TrainEnv.TrialName.Cur)
	for _, lnm := range ss.LayErrStatNms {
		dt.SetCellFloat(lnm + "_Err", row, ss.TrlErr[lnm])
		dt.SetCellFloat(lnm + "_SSE", row, ss.TrlSSE[lnm])
		dt.SetCellFloat(lnm + "_AvgSSE", row, ss.TrlAvgSSE[lnm])
		dt.SetCellFloat(lnm + "_CosDiff", row, ss.TrlCosDiff[lnm])
	}


	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		dt.SetCellFloat(ly.Nm+"_ActM.Avg", row, float64(ly.Pools[0].ActM.Avg))
		dt.SetCellFloat(ly.Nm+"_Ge.Avg", row, float64(ly.Pools[0].ActM.Avg))
	}
	ivt := ss.ValsTsr("Input")
	ovt := ss.ValsTsr("Output")

	categoryLayer.UnitValsTensor(ivt, "AvgSLrn")
	dt.SetCellTensor("CatAvgSLrn", row, ivt)
	categoryLayer.UnitValsTensor(ivt, "Act")
	dt.SetCellTensor("CatAct", row, ivt)

	sceneLayer.UnitValsTensor(ivt, "AvgSLrn")
	dt.SetCellTensor("SceneAvgSLrn", row, ivt)
	sceneLayer.UnitValsTensor(ivt, "Act")
	dt.SetCellTensor("SceneAct", row, ivt)

	hiddenLayer.UnitValsTensor(ivt, "AvgSLrn")
	dt.SetCellTensor("HiddenAvgSLrn", row, ivt)
	hiddenLayer.UnitValsTensor(ivt, "AvgL")
	dt.SetCellTensor("HiddenAvgL", row, ivt)
	hiddenLayer.UnitValsTensor(ivt, "ActM")
	dt.SetCellTensor("HiddenActM", row, ivt)
	hiddenLayer.UnitValsTensor(ivt, "ActP")
	dt.SetCellTensor("HiddenActP", row, ivt)

	ss.saveSynapseVariable("Hidden", "Scene", "DWt")
	dt.SetCellTensor("HiddentoSceneDWt", row, ivt)
	ss.saveSynapseVariable("Scene", "Hidden", "DWt")
	dt.SetCellTensor("ScenetoHiddenDWt", row, ivt)
	ss.saveSynapseVariable("Hidden", "Hidden", "DWt")
	dt.SetCellTensor("HiddentoHiddenDWt", row, ivt)
	ss.saveSynapseVariable("Output", "Hidden", "DWt")
	dt.SetCellTensor("OutputtoHiddenDWt", row, ivt)
	ss.saveSynapseVariable("Hidden", "Output", "DWt")
	dt.SetCellTensor("HiddentoOutputDWt", row, ivt)


	outputLayer.UnitValsTensor(ovt, "AvgSLrn")
	dt.SetCellTensor("OutAvgSLrn", row, ovt)
	outputLayer.UnitValsTensor(ovt, "ActM")
	dt.SetCellTensor("OutActM", row, ovt)
	outputLayer.UnitValsTensor(ovt, "ActP")
	dt.SetCellTensor("OutActP", row, ovt)

	// note: essential to use Go version of update when called from another goroutine
	// ss.TstTrlPlot.GoUpdate()

	if ss.TrnTrlFile != nil {
		// if ss.TrainEnv.Run.Cur == 0 && epc == 0 {
		if dt.MetaData["IsHeaderWritten"] == "No" {
			dt.WriteCSVHeaders(ss.TrnTrlFile, etable.Tab)
			dt.MetaData["IsHeaderWritten"] = "Yes"
		}
		dt.WriteCSVRow(ss.TrnTrlFile, row, etable.Tab)
	}
}

func (ss *Sim) ConfigTrnTrlLog(dt *etable.Table) {
	categoryLayer := ss.Net.LayerByName("Category").(leabra.LeabraLayer).AsLeabra()
	sceneLayer := ss.Net.LayerByName("Scene").(leabra.LeabraLayer).AsLeabra()
	hiddenLayer := ss.Net.LayerByName("Hidden").(leabra.LeabraLayer).AsLeabra()
	outputLayer := ss.Net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()

	dt.SetMetaData("name", "TrnTrlLog")
	dt.SetMetaData("desc", "Record of training per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	nt := ss.TrainEnv.Table.Len() // number in view
	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"CurrentTask", etensor.STRING, nil, nil},
		// {"CurrentTest", etensor.STRING, nil, nil}, do we need current test for train trial?
		{"Trial", etensor.INT64, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
	}

	for _, lnm := range ss.LayErrStatNms {
		sch = append(sch, etable.Column{lnm + "_Err", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_SSE", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_AvgSSE", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_CosDiff", etensor.FLOAT64, nil, nil})
	}
	for _, lnm := range ss.LayStatNms {
		sch = append(sch, etable.Column{lnm + "_ActM.Avg", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_Ge.Avg", etensor.FLOAT64, nil, nil})
	}
	sch = append(sch, etable.Schema{
		{"CatAvgSLrn", etensor.FLOAT64, categoryLayer.Shp.Shp, nil},
		{"CatAct", etensor.FLOAT64, categoryLayer.Shp.Shp, nil},
		{"SceneAvgSLrn", etensor.FLOAT64, sceneLayer.Shp.Shp, nil},
		{"SceneAct", etensor.FLOAT64, sceneLayer.Shp.Shp, nil},
		{"HiddenAvgSLrn", etensor.FLOAT64, hiddenLayer.Shp.Shp, nil},
		{"HiddenAvgL", etensor.FLOAT64, hiddenLayer.Shp.Shp, nil},
		{"HiddenActM", etensor.FLOAT64, hiddenLayer.Shp.Shp, nil},
		{"HiddenActP", etensor.FLOAT64, hiddenLayer.Shp.Shp, nil},
		// {"HiddentoHiddenWeights", etensor.FLOAT64, []int{9900}, nil}, // 9900 is the length of the weights array
		// {"OutputtoHiddenWeights", etensor.FLOAT64, []int{5000}, nil}, // 5000 is the length of the weights array
		// {"HiddentoOutputWeights", etensor.FLOAT64, []int{5000}, nil}, // 5000 is the length of the weights array
		{"ScenetoHiddenDWt", etensor.FLOAT64, []int{600}, nil}, // 600 is the length of the weights array
		{"HiddentoSceneDWt", etensor.FLOAT64, []int{600}, nil}, // 600 is the length of the weights array
		{"HiddentoHiddenDWt", etensor.FLOAT64, []int{9900}, nil}, // 9900 is the length of the weights array
		{"OutputtoHiddenDWt", etensor.FLOAT64, []int{5000}, nil}, // 5000 is the length of the weights array
		{"HiddentoOutputDWt", etensor.FLOAT64, []int{5000}, nil}, // 5000 is the length of the weights array
		{"OutAvgSLrn", etensor.FLOAT64, outputLayer.Shp.Shp, nil},
		{"OutActM", etensor.FLOAT64, outputLayer.Shp.Shp, nil},
		{"OutActP", etensor.FLOAT64, outputLayer.Shp.Shp, nil},
	}...)
	dt.SetFromSchema(sch, nt)
}

//////////////////////////////////////////////
//  TstTrlLog

// LogTstTrl adds data from current trial to the TstTrlLog table.
// log always contains number of testing items
func (ss *Sim) LogTstTrl(dt *etable.Table) {
	categoryLayer := ss.Net.LayerByName("Category").(leabra.LeabraLayer).AsLeabra()
	sceneLayer := ss.Net.LayerByName("Scene").(leabra.LeabraLayer).AsLeabra()
	hiddenLayer := ss.Net.LayerByName("Hidden").(leabra.LeabraLayer).AsLeabra()
	outputLayer := ss.Net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()

	epc := ss.TrainEnv.Epoch.Prv // this is triggered by increment so use previous value

	trl := ss.TestEnv.Trial.Cur
	// row := trl

	// if dt.Rows <= row {
	// 	dt.SetNumRows(row + 1)
	// }
	row := dt.Rows
	dt.SetNumRows(row + 1)

	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellString("CurrentTask", row, ss.CurrentTask)
	dt.SetCellString("CurrentTest", row, ss.CurrentTest)
	dt.SetCellFloat("Trial", row, float64(trl))
	dt.SetCellString("TrialName", row, ss.TestEnv.TrialName.Cur)
	for _, lnm := range ss.LayErrStatNms {
		dt.SetCellFloat(lnm + "_Err", row, ss.TrlErr[lnm])
		dt.SetCellFloat(lnm + "_SSE", row, ss.TrlSSE[lnm])
		dt.SetCellFloat(lnm + "_AvgSSE", row, ss.TrlAvgSSE[lnm])
		dt.SetCellFloat(lnm + "_CosDiff", row, ss.TrlCosDiff[lnm])
	}


	for _, lnm := range ss.LayStatNms {
		ly := ss.Net.LayerByName(lnm).(leabra.LeabraLayer).AsLeabra()
		dt.SetCellFloat(ly.Nm+"_ActM.Avg", row, float64(ly.Pools[0].ActM.Avg))
		dt.SetCellFloat(ly.Nm+"_Ge.Avg", row, float64(ly.Pools[0].ActM.Avg))
	}
	ivt := ss.ValsTsr("Input")
	ovt := ss.ValsTsr("Output")

	categoryLayer.UnitValsTensor(ivt, "Act")
	dt.SetCellTensor("CatAct", row, ivt)
	sceneLayer.UnitValsTensor(ivt, "ActM")
	dt.SetCellTensor("SceneActM", row, ivt)
	sceneLayer.UnitValsTensor(ivt, "Act")
	dt.SetCellTensor("SceneAct", row, ivt)
	hiddenLayer.UnitValsTensor(ivt, "ActM")
	dt.SetCellTensor("HiddenActM", row, ivt)
	hiddenLayer.UnitValsTensor(ivt, "ActP")
	dt.SetCellTensor("HiddenActP", row, ivt)
	outputLayer.UnitValsTensor(ovt, "ActM")
	dt.SetCellTensor("OutActM", row, ovt)
	outputLayer.UnitValsTensor(ovt, "ActP")
	dt.SetCellTensor("OutActP", row, ovt)

	// note: essential to use Go version of update when called from another goroutine
	ss.TstTrlPlot.GoUpdate()

	if ss.TstTrlFile != nil {
		// if ss.TrainEnv.Run.Cur == 0 && epc == 0 {
		if dt.MetaData["IsHeaderWritten"] == "No" {
			dt.WriteCSVHeaders(ss.TstTrlFile, etable.Tab)
			dt.MetaData["IsHeaderWritten"] = "Yes"
		}
		dt.WriteCSVRow(ss.TstTrlFile, row, etable.Tab)
	}
}

func (ss *Sim) ConfigTstTrlLog(dt *etable.Table) {
	categoryLayer := ss.Net.LayerByName("Category").(leabra.LeabraLayer).AsLeabra()
	sceneLayer := ss.Net.LayerByName("Scene").(leabra.LeabraLayer).AsLeabra()
	hiddenLayer := ss.Net.LayerByName("Hidden").(leabra.LeabraLayer).AsLeabra()
	outputLayer := ss.Net.LayerByName("Output").(leabra.LeabraLayer).AsLeabra()

	dt.SetMetaData("name", "TstTrlLog")
	dt.SetMetaData("desc", "Record of testing per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	nt := ss.TestEnv.Table.Len() // number in view
	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"CurrentTask", etensor.STRING, nil, nil},
		{"CurrentTest", etensor.STRING, nil, nil},
		{"Trial", etensor.INT64, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
	}
	for _, lnm := range ss.LayErrStatNms {
		sch = append(sch, etable.Column{lnm + "_Err", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_SSE", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_AvgSSE", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_CosDiff", etensor.FLOAT64, nil, nil})
	}
	for _, lnm := range ss.LayStatNms {
		sch = append(sch, etable.Column{lnm + "_ActM.Avg", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_Ge.Avg", etensor.FLOAT64, nil, nil})
	}
	sch = append(sch, etable.Schema{
		{"CatAct", etensor.FLOAT64, categoryLayer.Shp.Shp, nil},
		{"SceneActM", etensor.FLOAT64, sceneLayer.Shp.Shp, nil},
		{"SceneAct", etensor.FLOAT64, sceneLayer.Shp.Shp, nil},
		{"HiddenActM", etensor.FLOAT64, hiddenLayer.Shp.Shp, nil},
		{"HiddenActP", etensor.FLOAT64, hiddenLayer.Shp.Shp, nil},
		{"OutActM", etensor.FLOAT64, outputLayer.Shp.Shp, nil},
		{"OutActP", etensor.FLOAT64, outputLayer.Shp.Shp, nil},
	}...)
	dt.SetFromSchema(sch, nt)
}

func (ss *Sim) ConfigTstTrlPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Leabra Color Diff Model Test Trial Plot"
	plt.Params.XAxisCol = "Trial"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Trial", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("TrialName", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	for _, lnm := range ss.LayErrStatNms {
		plt.SetColParams(lnm + "_Err", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
		plt.SetColParams(lnm + "_SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
		plt.SetColParams(lnm + "_AvgSSE", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0)
		plt.SetColParams(lnm + "_CosDiff", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1)
	}

	for _, lnm := range ss.LayStatNms {
		plt.SetColParams(lnm+"_ActM.Avg", eplot.Off, eplot.FixMin, 0, eplot.FixMax, .5)
	}

	plt.SetColParams("CatAct", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("SceneAct", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("OutActM", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	plt.SetColParams("OutActP", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	return plt
}

//////////////////////////////////////////////
// Categorize Hidden Units
func (ss *Sim) AnalyzeTstTrl() {
	hiddenLayer := ss.Net.LayerByName("Hidden").(leabra.LeabraLayer).AsLeabra()
	vals := make([]float32, len(hiddenLayer.Neurons))
	hiddenLayer.UnitVals(&vals, "ActM")
	fmt.Println(vals)
}

//////////////////////////////////////////////
//  TstEpcLog

func (ss *Sim) LogTstEpc(dt *etable.Table) {
	row := dt.Rows
	dt.SetNumRows(row + 1)

	trl := ss.TstTrlLog
	tix := etable.NewIdxView(trl)
	epc := ss.TrainEnv.Epoch.Prv // ?

	// note: this shows how to use agg methods to compute summary data from another
	// data table, instead of incrementing on the Sim
	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	for _, lnm := range ss.LayErrStatNms {
		dt.SetCellFloat(lnm + "_SSE", row, agg.Sum(tix, lnm + "_SSE")[0])
		dt.SetCellFloat(lnm + "_AvgSSE", row, agg.Mean(tix, lnm + "_AvgSSE")[0])
		dt.SetCellFloat(lnm + "_PctErr", row, agg.Mean(tix, lnm + "_Err")[0])
		dt.SetCellFloat(lnm + "_PctCor", row, 1-agg.Mean(tix, lnm + "_Err")[0])
		dt.SetCellFloat(lnm + "_CosDiff", row, agg.Mean(tix, lnm + "_CosDiff")[0])
	}


	trlix := etable.NewIdxView(trl)
	trlix.Filter(func(et *etable.Table, row int) bool {
		return et.CellFloat("Output_SSE", row) > 0 // include error trials for Output
	})
	ss.TstErrLog = trlix.NewTable()

	allsp := split.All(trlix)
	for _, lnm := range ss.LayErrStatNms {
		split.Agg(allsp, lnm + "_SSE", agg.AggSum)
		split.Agg(allsp, lnm + "_AvgSSE", agg.AggMean)
	}

	split.Agg(allsp, "CatAct", agg.AggMean)
	split.Agg(allsp, "SceneAct", agg.AggMean)
	split.Agg(allsp, "OutActM", agg.AggMean)
	split.Agg(allsp, "OutActP", agg.AggMean)

	ss.TstErrStats = allsp.AggsToTable(etable.AddAggName)

	// note: essential to use Go version of update when called from another goroutine
	ss.TstEpcPlot.GoUpdate()
}

func (ss *Sim) ConfigTstEpcLog(dt *etable.Table) {
	dt.SetMetaData("name", "TstEpcLog")
	dt.SetMetaData("desc", "Summary stats for testing trials")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
	}
	for _, lnm := range ss.LayErrStatNms {
		sch = append(sch, etable.Column{lnm + "_SSE", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_AvgSSE", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_PctErr", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_PctCor", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_CosDiff", etensor.FLOAT64, nil, nil})
	}
	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigTstEpcPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Leabra Color Difference Model Testing Epoch Plot"
	plt.Params.XAxisCol = "Epoch"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("Epoch", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	for _, lnm := range ss.LayErrStatNms {
		plt.SetColParams(lnm + "_SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
		plt.SetColParams(lnm + "_AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
		plt.SetColParams(lnm + "_PctErr", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1) // default plot
		plt.SetColParams(lnm + "_PctCor", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1) // default plot
		plt.SetColParams(lnm + "_CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	}
	return plt
}



//////////////////////////////////////////////
//  TrnCycLog

// LogTrnCyc adds data from current trial to the TrnCycLog table.
// log always contains number of testing items
func (ss *Sim) LogTrnCyc(dt *etable.Table, cyc int) {
	categoryLayer := ss.Net.LayerByName("Category").(*leabra.Layer)
	sceneLayer := ss.Net.LayerByName("Scene").(*leabra.Layer)
	hiddenLayer := ss.Net.LayerByName("Hidden").(*leabra.Layer)
	outputLayer := ss.Net.LayerByName("Output").(*leabra.Layer)

	epc := ss.TrainEnv.Epoch.Cur // this is NOT triggered by increment so use current value
	trl := ss.TrainEnv.Trial.Cur

	row := dt.Rows
	dt.SetNumRows(row + 1)
	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellString("CurrentTask", row, ss.CurrentTask)
	dt.SetCellFloat("Trial", row, float64(trl))
	dt.SetCellString("TrialName", row, string(ss.TrainEnv.TrialName.Cur))
	dt.SetCellFloat("Cycle", row, float64(cyc))

	ivt := ss.ValsTsr("Input")
	ovt := ss.ValsTsr("Output")

	categoryLayer.UnitValsTensor(ivt, "Act")
	dt.SetCellTensor("CatAct", row, ivt)
	sceneLayer.UnitValsTensor(ivt, "Act")
	dt.SetCellTensor("SceneAct", row, ivt)
	sceneLayer.UnitValsTensor(ivt, "AvgS")
	dt.SetCellTensor("SceneAvgS", row, ivt)
	sceneLayer.UnitValsTensor(ivt, "AvgM")
	dt.SetCellTensor("SceneAvgM", row, ivt)
	// sceneLayer.UnitValsTensor(ivt, "Ge")
	// dt.SetCellTensor("SceneGe", row, ivt)
	hiddenLayer.UnitValsTensor(ivt, "Act")
	dt.SetCellTensor("HiddenAct", row, ivt)
	// hiddenLayer.UnitValsTensor(ivt, "Ge")
	// dt.SetCellTensor("HiddenGe", row, ivt)
	hiddenLayer.UnitValsTensor(ivt, "AvgS")
	dt.SetCellTensor("HiddenAvgS", row, ivt)
	hiddenLayer.UnitValsTensor(ivt, "AvgM")
	dt.SetCellTensor("HiddenAvgM", row, ivt)
	outputLayer.UnitValsTensor(ovt, "Act")
	dt.SetCellTensor("OutAct", row, ovt)


	// note: essential to use Go version of update when called from another goroutine
	// ss.TstTrlPlot.GoUpdate()

	if ss.TrnCycFile != nil {
		// if ss.TrainEnv.Run.Cur == 0 && epc == 0 {
		if dt.MetaData["IsHeaderWritten"] == "No" {
			dt.WriteCSVHeaders(ss.TrnCycFile, etable.Tab)
			dt.MetaData["IsHeaderWritten"] = "Yes"
		}
		dt.WriteCSVRow(ss.TrnCycFile, row, etable.Tab)
	}
}

func (ss *Sim) ConfigTrnCycLog(dt *etable.Table) {
	categoryLayer := ss.Net.LayerByName("Category").(*leabra.Layer)
	sceneLayer := ss.Net.LayerByName("Scene").(*leabra.Layer)
	hiddenLayer := ss.Net.LayerByName("Hidden").(*leabra.Layer)
	outputLayer := ss.Net.LayerByName("Output").(*leabra.Layer)

	dt.SetMetaData("name", "TrnCycLog")
	dt.SetMetaData("desc", "Record of testing per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	nt := ss.TrainEnv.Table.Len() // number in view
	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"CurrentTask", etensor.STRING, nil, nil},
		{"Trial", etensor.INT64, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
		{"Cycle", etensor.INT64, nil, nil},
	}
	for _, lnm := range ss.LayStatNms {
		sch = append(sch, etable.Column{lnm + "_ActM.Avg", etensor.FLOAT64, nil, nil})
	}
	sch = append(sch, etable.Schema{
		{"CatAct", etensor.FLOAT64, categoryLayer.Shp.Shp, nil},
		{"SceneAct", etensor.FLOAT64, sceneLayer.Shp.Shp, nil},
		{"SceneAvgS", etensor.FLOAT64, sceneLayer.Shp.Shp, nil},
		{"SceneAvgM", etensor.FLOAT64, sceneLayer.Shp.Shp, nil},
		// {"SceneGe", etensor.FLOAT64, sceneLayer.Shp.Shp, nil},
		{"HiddenAct", etensor.FLOAT64, hiddenLayer.Shp.Shp, nil},
		// {"HiddenGe", etensor.FLOAT64, hiddenLayer.Shp.Shp, nil},
		{"HiddenAvgS", etensor.FLOAT64, hiddenLayer.Shp.Shp, nil},
		{"HiddenAvgM", etensor.FLOAT64, hiddenLayer.Shp.Shp, nil},
		{"OutAct", etensor.FLOAT64, outputLayer.Shp.Shp, nil},

	}...)
	dt.SetFromSchema(sch, nt)
}

//////////////////////////////////////////////
//  TstCycLog

// LogTstCyc adds data from current trial to the TstCycLog table.
// log always contains number of testing items
func (ss *Sim) LogTstCyc(dt *etable.Table, cyc int) {
	categoryLayer := ss.Net.LayerByName("Category").(*leabra.Layer)
	sceneLayer := ss.Net.LayerByName("Scene").(*leabra.Layer)
	hiddenLayer := ss.Net.LayerByName("Hidden").(*leabra.Layer)
	outputLayer := ss.Net.LayerByName("Output").(*leabra.Layer)

	epc := ss.TrainEnv.Epoch.Prv // this is triggered by increment so use previous value
	trl := ss.TestEnv.Trial.Cur

	row := dt.Rows
	dt.SetNumRows(row + 1)
	dt.SetCellFloat("Run", row, float64(ss.TrainEnv.Run.Cur))
	dt.SetCellFloat("Epoch", row, float64(epc))
	dt.SetCellString("CurrentTask", row, ss.CurrentTask)
	dt.SetCellString("CurrentTest", row, ss.CurrentTest)
	dt.SetCellFloat("Trial", row, float64(trl))
	dt.SetCellString("TrialName", row, string(ss.TestEnv.TrialName.Cur))
	dt.SetCellFloat("Cycle", row, float64(cyc))

	ivt := ss.ValsTsr("Input")
	ovt := ss.ValsTsr("Output")

	categoryLayer.UnitValsTensor(ivt, "Act")
	dt.SetCellTensor("CatAct", row, ivt)
	sceneLayer.UnitValsTensor(ivt, "Act")
	dt.SetCellTensor("SceneAct", row, ivt)
	sceneLayer.UnitValsTensor(ivt, "Ge")
	dt.SetCellTensor("SceneGe", row, ivt)
	hiddenLayer.UnitValsTensor(ivt, "Act")
	dt.SetCellTensor("HiddenAct", row, ivt)
	hiddenLayer.UnitValsTensor(ivt, "Ge")
	dt.SetCellTensor("HiddenGe", row, ivt)
	outputLayer.UnitValsTensor(ovt, "ActM")
	dt.SetCellTensor("OutActM", row, ovt)
	outputLayer.UnitValsTensor(ovt, "ActP")
	dt.SetCellTensor("OutActP", row, ovt)

	// note: essential to use Go version of update when called from another goroutine
	// ss.TstTrlPlot.GoUpdate()

	if ss.TstCycFile != nil {
		// if ss.TrainEnv.Run.Cur == 0 && epc == 0 {
		if dt.MetaData["IsHeaderWritten"] == "No" {
			dt.WriteCSVHeaders(ss.TstCycFile, etable.Tab)
			dt.MetaData["IsHeaderWritten"] = "Yes"
		}
		dt.WriteCSVRow(ss.TstCycFile, row, etable.Tab)
	}
}

func (ss *Sim) ConfigTstCycLog(dt *etable.Table) {
	categoryLayer := ss.Net.LayerByName("Category").(*leabra.Layer)
	sceneLayer := ss.Net.LayerByName("Scene").(*leabra.Layer)
	hiddenLayer := ss.Net.LayerByName("Hidden").(*leabra.Layer)
	outputLayer := ss.Net.LayerByName("Output").(*leabra.Layer)

	dt.SetMetaData("name", "TstCycLog")
	dt.SetMetaData("desc", "Record of testing per input pattern")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	nt := ss.TrainEnv.Table.Len() // number in view
	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Epoch", etensor.INT64, nil, nil},
		{"CurrentTask", etensor.STRING, nil, nil},
		{"CurrentTest", etensor.STRING, nil, nil},
		{"Trial", etensor.INT64, nil, nil},
		{"TrialName", etensor.STRING, nil, nil},
		{"Cycle", etensor.INT64, nil, nil},
	}
	for _, lnm := range ss.LayStatNms {
		sch = append(sch, etable.Column{lnm + "_ActM.Avg", etensor.FLOAT64, nil, nil})
	}
	sch = append(sch, etable.Schema{
		{"CatAct", etensor.FLOAT64, categoryLayer.Shp.Shp, nil},
		{"SceneAct", etensor.FLOAT64, sceneLayer.Shp.Shp, nil},
		{"SceneGe", etensor.FLOAT64, hiddenLayer.Shp.Shp, nil},
		{"HiddenAct", etensor.FLOAT64, hiddenLayer.Shp.Shp, nil},
		{"HiddenGe", etensor.FLOAT64, hiddenLayer.Shp.Shp, nil},
		{"OutActM", etensor.FLOAT64, outputLayer.Shp.Shp, nil},
		{"OutActP", etensor.FLOAT64, outputLayer.Shp.Shp, nil},
	}...)
	dt.SetFromSchema(sch, nt)
}

//////////////////////////////////////////////
//  RunLog

// LogRun adds data from current run to the RunLog table.
func (ss *Sim) LogRun(dt *etable.Table) {
	run := ss.TrainEnv.Run.Cur // this is NOT triggered by increment yet -- use Cur
	row := dt.Rows
	dt.SetNumRows(row + 1)

	epclog := ss.TrnEpcLog
	epcix := etable.NewIdxView(epclog)
	// compute mean over last N epochs for run level
	nlast := 10
	if nlast > epcix.Len()-1 {
		nlast = epcix.Len() - 1
	}
	epcix.Idxs = epcix.Idxs[epcix.Len()-nlast:]

	params := ss.RunName() // includes tag

	dt.SetCellFloat("Run", row, float64(run))
	dt.SetCellString("Seed", row, strconv.FormatInt(ss.RndSeed, 10))
	dt.SetCellString("Params", row, params)
	dt.SetCellFloat("FirstZero", row, float64(ss.FirstZero))
	for _, lnm := range ss.LayErrStatNms {
		dt.SetCellFloat(lnm + "_SSE", row, agg.Mean(epcix, lnm + "_SSE")[0])
		dt.SetCellFloat(lnm + "_AvgSSE", row, agg.Mean(epcix, lnm + "_AvgSSE")[0])
		dt.SetCellFloat(lnm + "_PctErr", row, agg.Mean(epcix, lnm + "_PctErr")[0])
		dt.SetCellFloat(lnm + "_PctCor", row, agg.Mean(epcix, lnm + "_PctCor")[0])
		dt.SetCellFloat(lnm + "_CosDiff", row, agg.Mean(epcix, lnm + "_CosDiff")[0])
	}


	runix := etable.NewIdxView(dt)
	spl := split.GroupBy(runix, []string{"Params"})
	split.Desc(spl, "FirstZero")
	split.Desc(spl, "Output_PctCor")
	ss.RunStats = spl.AggsToTable(etable.AddAggName)
	// note: essential to use Go version of update when called from another goroutine
	ss.RunPlot.GoUpdate()
	if ss.RunFile != nil {
		if row == 0 {
			dt.WriteCSVHeaders(ss.RunFile, etable.Tab)
		}
		dt.WriteCSVRow(ss.RunFile, row, etable.Tab)
	}
}

func (ss *Sim) ConfigRunLog(dt *etable.Table) {
	dt.SetMetaData("name", "RunLog")
	dt.SetMetaData("desc", "Record of performance at end of training")
	dt.SetMetaData("read-only", "true")
	dt.SetMetaData("precision", strconv.Itoa(LogPrec))

	sch := etable.Schema{
		{"Run", etensor.INT64, nil, nil},
		{"Seed", etensor.STRING, nil, nil},
		{"Params", etensor.STRING, nil, nil},
		{"FirstZero", etensor.FLOAT64, nil, nil},
	}

	for _, lnm := range ss.LayErrStatNms {
		sch = append(sch, etable.Column{lnm + "_SSE", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_AvgSSE", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_PctErr", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_PctCor", etensor.FLOAT64, nil, nil})
		sch = append(sch, etable.Column{lnm + "_CosDiff", etensor.FLOAT64, nil, nil})
	}

	dt.SetFromSchema(sch, 0)
}

func (ss *Sim) ConfigSaveWts(mode, saveDirName string) {
	var dirName string

	fmt.Println("------")
	fmt.Println("saveDirName", saveDirName)
	if mode == "experiment" {
		dirName = "./data/" + time.Now().Format("2006-01-02-15-04-05")
	} else if mode == "batch" {
		dirName = saveDirName
	}
	fmt.Println("DirName", dirName)
	os.Mkdir(dirName, 0770)
	ss.DataDir = dirName
}



func (ss *Sim) ConfigRunPlot(plt *eplot.Plot2D, dt *etable.Table) *eplot.Plot2D {
	plt.Params.Title = "Leabra Random Associator 25 Run Plot"
	plt.Params.XAxisCol = "Run"
	plt.SetTable(dt)
	// order of params: on, fixMin, min, fixMax, max
	plt.SetColParams("Run", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
	plt.SetColParams("FirstZero", eplot.On, eplot.FixMin, 0, eplot.FloatMax, 0) // default plot
	for _, lnm := range ss.LayErrStatNms {
		plt.SetColParams(lnm + "_SSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
		plt.SetColParams(lnm + "_AvgSSE", eplot.Off, eplot.FixMin, 0, eplot.FloatMax, 0)
		plt.SetColParams(lnm + "_PctErr", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1) // default plot
		plt.SetColParams(lnm + "_PctCor", eplot.On, eplot.FixMin, 0, eplot.FixMax, 1) // default plot
		plt.SetColParams(lnm + "_CosDiff", eplot.Off, eplot.FixMin, 0, eplot.FixMax, 1)
	}
	return plt
}

////////////////////////////////////////////////////////////////////////////////////////////
// 		Gui

// Equal tells whether a and b contain the same elements.
// A nil argument is equivalent to an empty slice.
func Equal(a, b []float32) bool {
    if len(a) != len(b) {
        return false
    }
    for i, v := range a {
        if v != b[i] {
            return false
        }
    }
    return true
}


func (ss *Sim) ZeroAvgSlrn() {
	for _, ly := range ss.Net.Layers {
		if ly.IsOff() {
			continue
		}
		lay := ly.(*leabra.Layer)
		for ni := range lay.Neurons {
			nrn := &lay.Neurons[ni]
			nrn.AvgSLrn = 0
			nrn.AvgSS = 0.15
			nrn.AvgS = 0.15
			nrn.AvgM = 0.15
			nrn.AvgL = 0.4
		}
	}
}
// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	width := 1600
	height := 1200

	gi.SetAppName("Color Differentiation")
	gi.SetAppAbout(`This demonstrates a basic Leabra model. See <a href="https://github.com/emer/emergent">emergent on GitHub</a>.</p>`)

	win := gi.NewMainWindow("favila", "Favila Model", width, height)
	ss.Win = win

	vp := win.WinViewport2D()
	updt := vp.UpdateStart()

	mfr := win.SetMainFrame()

	tbar := gi.AddNewToolBar(mfr, "tbar")
	tbar.SetStretchMaxWidth()
	ss.ToolBar = tbar

	split := gi.AddNewSplitView(mfr, "split")
	split.Dim = mat32.X
	split.SetStretchMax()

	sv := giv.AddNewStructView(split, "sv")
	sv.SetStruct(ss)

	tv := gi.AddNewTabView(split, "tv")

	nv := tv.AddNewTab(netview.KiT_NetView, "NetView").(*netview.NetView)
	nv.Var = "Act"
	// nv.Params.ColorMap = "Jet" // default is ColdHot
	// which fares pretty well in terms of discussion here:
	// https://matplotlib.org/tutorials/colors/colormaps.html

	nv.SetNet(ss.Net)
	ss.NetView = nv

	nv.Scene().Camera.Pose.Pos.Set(0, 1, 4) // more "head on" than default which is more "top down"
	nv.Scene().Camera.LookAt(mat32.Vec3{0, 0, 0}, mat32.Vec3{0, 1, 0})

	plt := tv.AddNewTab(eplot.KiT_Plot2D, "TrnEpcPlot").(*eplot.Plot2D)
	ss.TrnEpcPlot = ss.ConfigTrnEpcPlot(plt, ss.TrnEpcLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstTrlPlot").(*eplot.Plot2D)
	ss.TstTrlPlot = ss.ConfigTstTrlPlot(plt, ss.TstTrlLog)

	// plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstCycPlot").(*eplot.Plot2D)
	// ss.TstCycPlot = ss.ConfigTstCycPlot(plt, ss.TstCycLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "TstEpcPlot").(*eplot.Plot2D)
	ss.TstEpcPlot = ss.ConfigTstEpcPlot(plt, ss.TstEpcLog)

	plt = tv.AddNewTab(eplot.KiT_Plot2D, "RunPlot").(*eplot.Plot2D)
	ss.RunPlot = ss.ConfigRunPlot(plt, ss.RunLog)

	split.SetSplits(.3, .7)

	tbar.AddAction(gi.ActOpts{Label: "Init", Icon: "update", Tooltip: "Initialize everything including network weights, and start over.  Also applies current params.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Init()
		vp.SetNeedsFullRender()
	})

	tbar.AddAction(gi.ActOpts{Label: "Stop", Icon: "stop", Tooltip: "Interrupts running.  Hitting Train again will pick back up where it left off.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		ss.Stop()
	})

	tbar.AddAction(gi.ActOpts{Label: "Step Trial", Icon: "step-fwd", Tooltip: "Advances one training trial at a time.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			ss.StopNow = false
			ss.TrainTrial()
			ss.Stopped()

			ss.IsRunning = false
			vp.SetNeedsFullRender()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Step Epoch", Icon: "fast-fwd", Tooltip: "Advances one epoch (complete set of training patterns) at a time.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.TrainEpoch()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Run", Icon: "fast-fwd", Tooltip: "Advances one full training Run at a time.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.TrainRun()
		}
	})

// tbar.AddAction(gi.ActOpts{Label: "Task: Shape", Icon: "fast-fwd", Tooltip: "Advances one full training Run of shape learning task at a time.", UpdateFunc: func(act *gi.Action) {
// 	act.SetActiveStateUpdt(!ss.IsRunning)
// }}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
// 	if !ss.IsRunning {
// 		ss.IsRunning = true
// 		tbar.UpdateActions()
// 		go ss.TaskShapes()
// 	}
// })

tbar.AddAction(gi.ActOpts{Label: "Task: Color Study w/o Osc", Icon: "fast-fwd", Tooltip: "Advances one full training of color object face associations, but with no oscillations.", UpdateFunc: func(act *gi.Action) {
	act.SetActiveStateUpdt(!ss.IsRunning)
}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
	if !ss.IsRunning {
		ss.IsRunning = true
		tbar.UpdateActions()
		go ss.TaskColorWOOsc()
	}
})

tbar.AddAction(gi.ActOpts{Label: "SetParamSet TaskColorRecall", Icon: "fast-fwd", Tooltip: "Advances one full training of color object face associations, but with no oscillations.", UpdateFunc: func(act *gi.Action) {
	act.SetActiveStateUpdt(!ss.IsRunning)
}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
	if !ss.IsRunning {
		tbar.UpdateActions()
		go ss.SetParamsSet("TaskColorRecall", "", ss.LogSetParams) // all sheets
	}
})

tbar.AddAction(gi.ActOpts{Label: "Init Acts", Icon: "fast-fwd", Tooltip: "Advances one full training of color object face associations, but with no oscillations.", UpdateFunc: func(act *gi.Action) {
	act.SetActiveStateUpdt(!ss.IsRunning)
}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
	if !ss.IsRunning {
		tbar.UpdateActions()
		ss.ZeroAvgSlrn()
		ss.UpdateView(false)
		for _, ly := range ss.Net.Layers {
			if ly.IsOff() {
				continue
			}
			val := ss.printlayerVariable(ly.Name(), "AvgSLrn")
			fmt.Println(ly.Name(), *val)
		}
	}
})


tbar.AddAction(gi.ActOpts{Label: "Task: Color Study", Icon: "fast-fwd", Tooltip: "Advances one full training Run of shape learning task at a time.", UpdateFunc: func(act *gi.Action) {
	act.SetActiveStateUpdt(!ss.IsRunning)
}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
	if !ss.IsRunning {
		ss.IsRunning = true
		tbar.UpdateActions()
		go ss.TaskColorRecall()

	}
})

// tbar.AddAction(gi.ActOpts{Label: "Task: Scene Recall", Icon: "fast-fwd", Tooltip: "Advances one full training Run of face learning task at a time.", UpdateFunc: func(act *gi.Action) {
// 	act.SetActiveStateUpdt(!ss.IsRunning)
// }}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
// 	if !ss.IsRunning {
// 		ss.IsRunning = true
// 		tbar.UpdateActions()
// 		go ss.TaskSceneRecall()
// 	}
// })

// tbar.AddAction(gi.ActOpts{Label: "Task: Study/Scene", Icon: "fast-fwd", Tooltip: "interleaves color study and face recall tasks.", UpdateFunc: func(act *gi.Action) {
// 	act.SetActiveStateUpdt(!ss.IsRunning)
// }}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
// 	if !ss.IsRunning {
// 		ss.IsRunning = true
// 		tbar.UpdateActions()
// 		go ss.TaskInterleaveStudyAndSceneRecall()
// 	}
// })
// tbar.AddAction(gi.ActOpts{Label: "Task: Rp+ Retrieval", Icon: "fast-fwd", Tooltip: "Advances one full training Run of shape learning task at a time.", UpdateFunc: func(act *gi.Action) {
// 	act.SetActiveStateUpdt(!ss.IsRunning)
// }}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
// 	if !ss.IsRunning {
// 		ss.IsRunning = true
// 		tbar.UpdateActions()
// 		go ss.TaskRetrievalPractice()
// 	}
// })
// tbar.AddAction(gi.ActOpts{Label: "Task: Rp- Restudy", Icon: "fast-fwd", Tooltip: "Advances one full training Run of shape learning task at a time.", UpdateFunc: func(act *gi.Action) {
// 	act.SetActiveStateUpdt(!ss.IsRunning)
// }}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
// 	if !ss.IsRunning {
// 		ss.IsRunning = true
// 		tbar.UpdateActions()
// 		go ss.TaskRestudy()
// 	}
// })

tbar.AddAction(gi.ActOpts{Label: "Task: ALL OF THEM", Icon: "fast-fwd", Tooltip: "RUN EVERY TASK AT ONCE.", UpdateFunc: func(act *gi.Action) {
	act.SetActiveStateUpdt(!ss.IsRunning)
}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
	if !ss.IsRunning {
		ss.IsRunning = true
		tbar.UpdateActions()
		go ss.TaskRunAll()
		// ss.TaskRestudy()
	}
})

	tbar.AddSeparator("test")

	tbar.AddAction(gi.ActOpts{Label: "Test Color Trial", Icon: "step-fwd", Tooltip: "Runs the next testing trial.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			ss.StopNow = false

			ss.SetUpTestTrial("color_diff_stim.dat", "TaskColorRecall", "TestColorAll")

			categoryLayer := ss.Net.LayerByName("Category").(*leabra.Layer)
			categoryLayer.SetType(emer.Input)
			sceneLayer := ss.Net.LayerByName("Scene").(*leabra.Layer)
			sceneLayer.SetType(emer.Input)
			outputLayer := ss.Net.LayerByName("Output").(*leabra.Layer)
			outputLayer.SetType(emer.Target)

			ss.TestTrial(false) // don't return on change -- wrap
			// ss.CurrentTest = ""
			ss.Stopped()
			ss.IsRunning = false
			vp.SetNeedsFullRender()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Test Item", Icon: "step-fwd", Tooltip: "Prompts for a specific input pattern name to run, and runs it in testing mode.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		gi.StringPromptDialog(vp, "", "Test Item",
			gi.DlgOpts{Title: "Test Item", Prompt: "Enter the Name of a given input pattern to test (case insensitive, contains given string."},
			win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
				dlg := send.(*gi.Dialog)
				if sig == int64(gi.DialogAccepted) {
					val := gi.StringPromptDialogValue(dlg)
					idxs := ss.TestEnv.Table.RowsByString("Name", val, etable.Contains, etable.IgnoreCase)
					if len(idxs) == 0 {
						gi.PromptDialog(nil, gi.DlgOpts{Title: "Name Not Found", Prompt: "No patterns found containing: " + val}, gi.AddOk, gi.NoCancel, nil, nil)
					} else {
						if !ss.IsRunning {
							ss.IsRunning = true
							fmt.Printf("testing index: %v\n", idxs[0])
							ss.TestItem(idxs[0])
							ss.IsRunning = false
							vp.SetNeedsFullRender()
						}
					}
				}
			})
	})

	tbar.AddAction(gi.ActOpts{Label: "Test Color All", Icon: "fast-fwd", Tooltip: "Tests all of the testing trials.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			ss.StopNow = false
			ss.TestColorAll()
			ss.Stopped()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Test Scene All", Icon: "fast-fwd", Tooltip: "Tests all of the testing trials.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			ss.StopNow = false
			ss.TestSceneAll()
			ss.Stopped()
		}
	})

	tbar.AddAction(gi.ActOpts{Label: "Test All", Icon: "fast-fwd", Tooltip: "Tests all of the testing trials.", UpdateFunc: func(act *gi.Action) {
		act.SetActiveStateUpdt(!ss.IsRunning)
	}}, win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
		if !ss.IsRunning {
			ss.IsRunning = true
			tbar.UpdateActions()
			go ss.RunTestAll()
		}
	})

	tbar.AddSeparator("log")

	tbar.AddAction(gi.ActOpts{Label: "Reset RunLog", Icon: "reset", Tooltip: "Reset the accumulated log of all Runs, which are tagged with the ParamSet used"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.RunLog.SetNumRows(0)
			ss.RunPlot.Update()
		})

	tbar.AddSeparator("misc")

	tbar.AddAction(gi.ActOpts{Label: "New Seed", Icon: "new", Tooltip: "Generate a new initial random seed to get different results.  By default, Init re-establishes the same initial seed every time."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.NewRndSeed()
		})

	tbar.AddAction(gi.ActOpts{Label: "README", Icon: "file-markdown", Tooltip: "Opens your browser on the README file that contains instructions for how to run this model."}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			gi.OpenURL("https://github.com/emer/leabra/blob/master/examples/ra25/README.md")
		})

	vp.UpdateEndNoSig(updt)

	// main menu
	appnm := gi.AppName()
	mmen := win.MainMenu
	mmen.ConfigMenus([]string{appnm, "File", "Edit", "Window"})

	amen := win.MainMenu.ChildByName(appnm, 0).(*gi.Action)
	amen.Menu.AddAppMenu(win)

	emen := win.MainMenu.ChildByName("Edit", 1).(*gi.Action)
	emen.Menu.AddCopyCutPaste(win)

	// note: Command in shortcuts is automatically translated into Control for
	// Linux, Windows or Meta for MacOS
	// fmen := win.MainMenu.ChildByName("File", 0).(*gi.Action)
	// fmen.Menu.AddAction(gi.ActOpts{Label: "Open", Shortcut: "Command+O"},
	// 	win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
	// 		FileViewOpenSVG(vp)
	// 	})
	// fmen.Menu.AddSeparator("csep")
	// fmen.Menu.AddAction(gi.ActOpts{Label: "Close Window", Shortcut: "Command+W"},
	// 	win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
	// 		win.Close()
	// 	})

	inQuitPrompt := false
	gi.SetQuitReqFunc(func() {
		if inQuitPrompt {
			return
		}
		inQuitPrompt = true
		gi.PromptDialog(vp, gi.DlgOpts{Title: "Really Quit?",
			Prompt: "Are you <i>sure</i> you want to quit and lose any unsaved params, weights, logs, etc?"}, gi.AddOk, gi.AddCancel,
			win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
				if sig == int64(gi.DialogAccepted) {
					gi.Quit()
				} else {
					inQuitPrompt = false
				}
			})
	})

	// gi.SetQuitCleanFunc(func() {
	// 	fmt.Printf("Doing final Quit cleanup here..\n")
	// })

	inClosePrompt := false
	win.SetCloseReqFunc(func(w *gi.Window) {
		if inClosePrompt {
			return
		}
		inClosePrompt = true
		gi.PromptDialog(vp, gi.DlgOpts{Title: "Really Close Window?",
			Prompt: "Are you <i>sure</i> you want to close the window?  This will Quit the App as well, losing all unsaved params, weights, logs, etc"}, gi.AddOk, gi.AddCancel,
			win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
				if sig == int64(gi.DialogAccepted) {
					gi.Quit()
				} else {
					inClosePrompt = false
				}
			})
	})

	win.SetCloseCleanFunc(func(w *gi.Window) {
		go gi.Quit() // once main window is closed, quit
	})

	win.MainMenuUpdated()
	return win
}

// These props register Save methods so they can be used
var SimProps = ki.Props{
	"CallMethods": ki.PropSlice{
		{"SaveWeights", ki.Props{
			"desc": "save network weights to file",
			"icon": "file-save",
			"Args": ki.PropSlice{
				{"File Name", ki.Props{
					"ext": ".wts,.wts.gz",
				}},
			},
		}},
		{"SaveParams", ki.Props{
			"desc": "save parameters to file",
			"icon": "file-save",
			"Args": ki.PropSlice{
				{"File Name", ki.Props{
					"ext": ".params",
				}},
			},
		}},
	},
}



func CreateFlagVariable(flag_name string, flag_type string, default_value interface{}, description string) (interface{}) {
	// parameters
	// flag_name: name of the flag
	// flag_type: type of the flag (either string, bool, float64, or defaults to int)
	// default_value: default value of the flag
	// description: comment to yourself
	// returns 2 arguments
	// first return value is the pointer to the variable that holds the flag value
	// second return value is whether we used nil as a default argument
	switch flag_type {
	case "string":
		var variable string
		if default_value != nil {
			flag.StringVar(&variable, flag_name, default_value.(string), description)
		} else {
			flag.StringVar(&variable, flag_name, "", description)
		}
		return &variable
	case "bool":
		var variable bool
		if default_value != nil {
			flag.BoolVar(&variable, flag_name, default_value.(bool), description)
		} else {
			flag.BoolVar(&variable, flag_name, false, description)
		}
		return &variable
	case "float64":
		var variable float64
		if default_value != nil {
			flag.Float64Var(&variable, flag_name, default_value.(float64), description)
		} else {
			flag.Float64Var(&variable, flag_name, -1.0, description)
		}
		return &variable
	default: // default handling is of int type
		var variable int
		if default_value != nil {
			flag.IntVar(&variable, flag_name, default_value.(int), description)
		} else {
			flag.IntVar(&variable, flag_name, -1, description)
		}
		return &variable
	}
}

// Convert arbitrary types to string
func convert_to_string(value interface{}) string {
	switch (value).(type) {
	case string:
		return value.(string)
	case bool:
		return strconv.FormatBool((value).(bool))
	case float64:
		return strconv.FormatFloat((value).(float64), 'f', 4, 64)
	case int:
		return strconv.Itoa(value.(int))
	default: // default handling is of int type
		return strconv.FormatInt((value).(int64), 10)
	}
}

func (ss *Sim) CmdArgs() (err error) {
	ss.NoGui = true
	flag.StringVar(&ss.ParamSet, "params", "", "ParamSet name to use -- must be valid name as listed in compiled-in params or loaded params")
	flag.StringVar(&ss.Tag, "tag", "", "extra tag to add to file names saved from this run")
	flag.IntVar(&ss.MaxRuns, "runs", 10, "number of runs to do (note that MaxEpcs is in paramset)")
	flag.BoolVar(&ss.LogSetParams, "setparams", false, "if true, print a record of each parameter that is set")
	flag.BoolVar(&ss.SaveWts, "wts", false, "if true, save final weights after each run")

	mode := CreateFlagVariable("mode", "string", "experiment", "in batch mode or experimentation mode? (save to different directories)").(*string)
	saveDirName := CreateFlagVariable("saveDirName", "string", "", "name of directory to save log files to (default is current time); allows external control").(*string)
	note := CreateFlagVariable("note", "string", "", "user note -- describe the run params etc").(*string)
	saveEpcLog := CreateFlagVariable("epclog", "bool", true, "if true, save train epoch log to file").(*bool)
	saveRunLog := CreateFlagVariable("runlog", "bool", true, "if true, save run epoch log to file").(*bool)
	appendRuntoSheet := CreateFlagVariable("appendRuntoSheet", "bool", false, "if true, save run metadata to ").(*bool)
	saveTrnTrlLog := CreateFlagVariable("trntrllog", "bool", true, "if true, save train trial log to file").(*bool)
	saveTstTrlLog := CreateFlagVariable("tsttrllog", "bool", true, "if true, save test trial log to file").(*bool)
	saveTrnCycLog := CreateFlagVariable("trncyclog", "bool", false, "if true, save train cycle log to file").(*bool)
	saveTstCycLog := CreateFlagVariable("tstcyclog", "bool", false, "if true, save test cycle log to file").(*bool)

	// nogui := CreateFlagVariable("nogui", "bool", true, "if not passing any other args and want to run nogui, use nogui").(*bool)

	// Adjusting Parameter space
	CreateFlagVariable("Base_Layer_Learn_AvgL_SetAveL", "bool", nil, "if true, set a fixed AveL to use in BCM component. Default should be false so it's dynamically updated.")
	CreateFlagVariable("Base_Layer_Learn_AvgL_AveLFix", "float64", nil, "fixed AveL value to use, if not being dynamically updated. Only used if SetAveL")
	CreateFlagVariable("Hidden_Layer_Learn_AvgL_SetAveL", "bool", nil, "if true, set a fixed AveL to use in BCM component. Default should be false so it's dynamically updated.")
	CreateFlagVariable("Hidden_Layer_Learn_AvgL_AveLFix", "float64", nil, "fixed AveL value to use, if not being dynamically updated. Only used if SetAveL")

	CreateFlagVariable("Layer_ColorRecall_Learn_AvgL_AveLFix", "float64", nil, "fixed AveL value to use, if not being dynamically updated. Only used if SetAveL")

	CreateFlagVariable("Hidden_ColorRecall_Layer_OscAmnt", "float64", nil, "oscAmount, the amplitude of sinusoidal function on base gi")
	CreateFlagVariable("Scene_ColorRecall_Layer_OscAmnt", "float64", nil, "oscAmount, the amplitude of sinusoidal function on base gi")
	CreateFlagVariable("Output_ColorRecall_Layer_OscAmnt", "float64", nil, "oscAmount, the amplitude of sinusoidal function on base gi")

	CreateFlagVariable("Hidden_ColorRecall_Layer_gi", "float64", nil, "gi")
	CreateFlagVariable("Scene_ColorRecall_Layer_gi", "float64", nil, "gi")
	CreateFlagVariable("Output_ColorRecall_Layer_gi", "float64", nil, "gi")

	// CreateFlagVariable("Output_ColorRecall_Layer_gi", "float64", nil, "inhibitory gi level")

	CreateFlagVariable("TaskColorRecall_Prjn_Learn_XCal_LTD_mult", "float64", nil, "multiplication factor for LTD portion of XCAL function. Default is 1 so that there's no change")
	CreateFlagVariable("TaskColorRecall_Prjn_Learn_XCal_DRev", "float64", nil, "constant that determines the point where the function reverses direction")

	CreateFlagVariable("Base_Prjn_Learn_Norm_On", "bool", nil, "norm and momentum on works better, but wt bal is not better for smaller nets")
	CreateFlagVariable("Base_Prjn_Learn_Momentum_On", "bool", nil, "norm and momentum on works better, but wt bal is not better for smaller nets")

	CreateFlagVariable("TaskColorWOOsc_OutputtoHidden_Prjn_WtScale_Rel", "float64", nil, "Weight scale from color output layer to hidden layer")

	CreateFlagVariable("HiddNumOverlapUnits", "int", 2, "numer of overlapping units")
	CreateFlagVariable("same_diff_flag", "string", nil, "same or different condition")
	CreateFlagVariable("LRateOverAll", "float64", 1.0, "overall lrate")

	flag.Parse()

	// Adjust Parameter values
	ss.ParamValues = make(map[string]string)

	flag.Visit(func(fl *flag.Flag) {
		ss.ParamValues[fl.Name] = fl.Value.String()
	})

	// fmt.Println("mode",  mode)
	ss.ConfigSaveWts(*mode, *saveDirName)
	// Replace ss.Init()
	// We do not want to run ss.NewRun() now, the TaskRunAll() function will run NewRun() for us
	rand.Seed(ss.RndSeed)

	ss.ConfigEnv() // re-config env just in case a different set of patterns was
	// selected or patterns have been modified etc
	ss.StopNow = false
	ss.SetParams("", ss.LogSetParams) // all sheets
	ss.ConfigParamValues()
	ss.UpdateView(true)

	if *note != "" {
		fmt.Printf("note: %s\n", note)
	}
	if ss.ParamSet != "" {
		fmt.Printf("Using ParamSet: %s\n", ss.ParamSet)
	}

	if *saveEpcLog {
		var err error
		fnm := ss.LogFileName("epc")
		ss.TrnEpcFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.TrnEpcFile = nil
		} else {
			fmt.Printf("Saving epoch log to: %v\n", fnm)
			defer ss.TrnEpcFile.Close()
		}
	}
	if *saveRunLog {
		var err error
		fnm := ss.LogFileName("run")
		ss.RunFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.RunFile = nil
		} else {
			fmt.Printf("Saving run log to: %v\n", fnm)
			defer ss.RunFile.Close()
		}
	}

	if *saveTrnTrlLog {
		var err error
		fnm := ss.LogFileName("trntrl")
		ss.TrnTrlFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.TrnTrlFile = nil
		} else {
			fmt.Printf("Saving train trial log to: %v\n", fnm)
			defer ss.TrnTrlFile.Close()
		}
	}

	if *saveTstTrlLog {
		var err error
		fnm := ss.LogFileName("tsttrl")
		ss.TstTrlFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.TstTrlFile = nil
		} else {
			fmt.Printf("Saving test trial log to: %v\n", fnm)
			defer ss.TstTrlFile.Close()
		}
	}



	if *saveTrnCycLog {
		var err error
		fnm := ss.LogFileName("trncyc")
		ss.TrnCycFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.TrnCycFile = nil
		} else {
			fmt.Printf("Saving train cycle log to: %v\n", fnm)
			defer ss.TrnCycFile.Close()
		}
	}

	if *saveTstCycLog {
		var err error
		fnm := ss.LogFileName("tstcyc")
		ss.TstCycFile, err = os.Create(fnm)
		if err != nil {
			log.Println(err)
			ss.TstCycFile = nil
		} else {
			fmt.Printf("Saving test cycle log to: %v\n", fnm)
			defer ss.TstCycFile.Close()
		}
	}

	if ss.SaveWts {
		fmt.Printf("Saving final weights per run\n")
	}



	fmt.Printf("Running %d Runs\n", ss.MaxRuns)

	for i := 0; i < ss.MaxRuns; i++ { // run maxruns number of time
		ss.TaskRunAll()
	}

	if *appendRuntoSheet{
		path := "./experiment_log.txt"
		writer, err := os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0644)
		if err != nil {
			return err
		}
		defer writer.Close()
		writer.WriteString("{\n")

		hiddentohidden_prjn := ss.getPrjn("Hidden", "Hidden")
		ss.ParamValues["DataDir"] = ss.DataDir
		ss.ParamValues["HiddentoHidden_NumOverlapUnits"] = convert_to_string(hiddentohidden_prjn.WtInit.NumOverlapUnits)
		ss.ParamValues["HiddentoHidden_NumUniqueUnits"] = convert_to_string(hiddentohidden_prjn.WtInit.NumTotalUnits)

		for key, val := range ss.ParamValues {
			// Convert each key/value pair in m to a string
			writeline := fmt.Sprintf("\"%s\": \"%s\"\n", key, val)
		// 	// Do whatever you want to do with the string;
		// 	// in this example I just print out each of them.
			if _, err := writer.WriteString(writeline); err != nil {
				log.Println(err)
			}
		}
		writer.WriteString("},\n")
	}

	return
}

// modLikePython is a helper function that generalizes modulo arithmetic to
// negative divisors, convenient for implementing periodic boundary conditions
func modLikePython(d, m int) int {
   var res int = d % m
   if ((res < 0 && m > 0) || (res > 0 && m < 0)) {
      return res + m
   }
   return res
}

// GenerateWeights generates weights according to the strategy specified
//  - "Rand": Use the Learn.WtInit params to generate random numbers
//  - "TesselSpec": Use the TesselSpec function to generate a distance-based function
// array_len specifies how long the array is (i.e. how many connections in prjn)
// ri specifies what unit this is (only relevant for the distance-based TesselSpec function)
func GenerateWeights( strategy string, pj *leabra.Prjn, array_len int, ri int) (arr []float32) {
	arr = make([]float32, 0, array_len)

	var NumUniqueUnits = pj.WtInit.NumTotalUnits - pj.WtInit.NumOverlapUnits

	if (NumUniqueUnits < 0) || (pj.WtInit.NumOverlapUnits < 0) {
		panic("Prjn's NumUniqueUnits and NumOverlapUnits cannot be negative")
	}
	if (pj.WtInit.NumOverlapUnits != 2) || (pj.WtInit.NumTotalUnits != 6) {
		panic("In Favila experiment model, NumOverlapUnits must be 2 and NumTotalUnits must be 6")
	}
	minunitid_item1 := 25 - NumUniqueUnits
	maxunitid_item1 := 25 + pj.WtInit.NumOverlapUnits
	minunitid_item2 := 25
	maxunitid_item2 := 25 + pj.WtInit.NumOverlapUnits + NumUniqueUnits

	if strategy == "Rand" {
		for ci := 0; ci < array_len; ci++ {
			arr = append(arr, float32(pj.WtInit.Gen(-1)))
		}
	} else if strategy == "TesselSpec" {
		for ci := 0; ci < array_len; ci++ {
			if (ci == modLikePython(ri + 0, array_len)) || (ci == modLikePython(ri - 1, array_len)) {
				arr = append(arr, 0.9) //was .7
			} else if (ci == modLikePython(ri + 1, array_len)) || (ci == modLikePython(ri - 2, array_len)) {
				arr = append(arr, 0.9) //was .5
			} else if (ci == modLikePython(ri + 2, array_len)) || (ci == modLikePython(ri - 3, array_len)) {
				arr = append(arr, 0.9) //was .3
			} else if (ci == modLikePython(ri + 3, array_len)) || (ci == modLikePython(ri - 4, array_len)) {
				arr = append(arr, 0.9) //was .1
			} else {
				arr = append(arr, 0)
			}
		}
	} else if strategy == "One-to-One" {
		for ci := 0; ci < array_len; ci++ {
			arr = append(arr, 0)
		}

		arr[ri] = 0.99
	} else if strategy == "MedShared" {

		for ci := 0; ci < array_len; ci++ {
			arr = append(arr, float32(pj.WtInit.Gen(-1)))
		}

		for ci := 0; ci < array_len; ci++ {
			arr = append(arr, 0)
		}

		if minunitid_item1 <= ri && ri < maxunitid_item1 {
			arr[4] = .99
		}
		if minunitid_item2 <= ri && ri < maxunitid_item2 {
			arr[1] = .99
		}
	} else if strategy == "RandomHiddenToHidden" {
		for ci := 0; ci < array_len; ci++ {
			arr = append(arr, float32(pj.WtInit.Gen(-1)))
		}

		maxunitid_item1 -= 1 // Self-connection not in recurrent weight array
		maxunitid_item2 -= 1 // Self-connection not in recurrent weight array

		if minunitid_item1 <= ri && ri < maxunitid_item1 {
			for ci := minunitid_item1; ci < maxunitid_item1; ci++ {
				arr[ci] = .99
			}

		}
		if minunitid_item2 <= ri && ri < maxunitid_item2 {
			for ci := minunitid_item2; ci < maxunitid_item2; ci++ {
				arr[ci] = .99
			}
		}
	} else if strategy == "SparseHiddenToHidden" {
		for ci := 0; ci < array_len; ci++ {
			if erand.ZeroOne(-1) < pj.WtInit.SparseMix {
				arr = append(arr, float32(pj.WtInit.Gen(-1)))
			} else {
				// make sure that the weight is greater than 0 and less than 1
				sparseweight := math.Max(0, math.Min(erand.UniformMeanRange(pj.WtInit.SecondModeMean, pj.WtInit.SecondModeVar, -1), 1))
				arr = append(arr, float32(sparseweight))
			}
		}
		maxunitid_item1 -= 1 // Self-connection not in recurrent weight array
		maxunitid_item2 -= 1 // Self-connection not in recurrent weight array

		if minunitid_item1 <= ri && ri < maxunitid_item1 {
			for ci := minunitid_item1; ci < maxunitid_item1; ci++ {
				arr[ci] = .99
			}
			for ci := maxunitid_item1; ci < maxunitid_item2; ci++ {
				arr[ci] = .99
			}

		}
		if minunitid_item2 <= ri && ri < maxunitid_item2 {
			for ci := minunitid_item1; ci < minunitid_item2; ci++ {
				arr[ci] = .99
			}
			for ci := minunitid_item2; ci < maxunitid_item2; ci++ {
				arr[ci] = .99
			}
		}
	} else if strategy == "CategoryToHidden" {
		for ci := 0; ci < array_len; ci++ {
			arr = append(arr, float32(pj.WtInit.Gen(-1)))
		}
		if minunitid_item1 <= ri && ri < maxunitid_item2 {
			arr[1] = .99
		}
	} else if strategy == "HiddenToOutput" {
		SameDiffCondition := pj.WtInit.SameDiffCondition


		for ci := 0; ci < array_len; ci++ {
			// arr = append(arr, 0)
			arr = append(arr, float32(pj.WtInit.Gen(-1)))

		}

		if SameDiffCondition == "Different" {
			if ri == 25 {
				//
				// for ci := 0; ci < array_len; ci++ {
				// 		arr[ci]= float32(pj.WtInit.Gen(-1))
				// }

				for ci := minunitid_item1; ci < maxunitid_item1; ci++ {
					arr[ci] = .99
				}
			}
			if ri == 29 {
				//
				// for ci := 0; ci < array_len; ci++ {
				// 		arr[ci]= float32(pj.WtInit.Gen(-1))
				// }

				for ci := minunitid_item2; ci < maxunitid_item2; ci++ {
					arr[ci] = .99
				}
			}
		} else if SameDiffCondition == "Same" {
			if ri == 25 {
				//
				// for ci := 0; ci < array_len; ci++ {
				// 	arr[ci]= float32(pj.WtInit.Gen(-1))
				// }

				for ci := minunitid_item1; ci < maxunitid_item1; ci++ {
					arr[ci] = .99
				}
				for ci := minunitid_item2; ci < maxunitid_item2; ci++ {
					arr[ci] = .99
				}
			}
		} else {
			panic(fmt.Sprintf("SameDiffCondition: %s not supported", SameDiffCondition))
		}
	}

	return arr
}

// SliceIndex is a helper function that returns the first elements
// that satisfies a condition
func SliceIndex(limit int, predicate func(i int) bool) int {
    for i := 0; i < limit; i++ {
        if predicate(i) {
            return i
        }
    }
    return -1
}

// This function functions as InitWts in Network
func (ss *Sim) SaveTesselSpectoJSON() {
	// layerGenStrategy := ss.GenerateStrategy()

	net_out := weights.Network{}
	net_out.Network = ss.Net.Nm
	net_out.Layers = []weights.Layer{}
	// For each layer, save layer name (i.e. receiver name) and list of receiving projections
	// For each projection, save sender name, number of units in sender, and weight statistics
	// 		Initialize weight statistics only if receiver is after sender
	// 		Reciprocal connections initialized in following for loop
	for _, ly := range ss.Net.Layers {
		ly_out := weights.Layer{}
		ly_out.Layer = ly.(*leabra.Layer).Nm
		ly_out.Prjns = []weights.Prjn{}
		for _, pj := range ly.(*leabra.Layer).RcvPrjns {
			pj_out := weights.Prjn{}
			pj_out.From = pj.SendLay().Name()
			pj_out.Rs = []weights.Recv{}
			nr := len(pj.RecvLay().(*leabra.Layer).Neurons)


			for ri := 0; ri < nr; ri++ {
				recv_out := weights.Recv{}
				recv_out.Ri = ri
				nc := int(pj.(*leabra.Prjn).RConN[ri])
				st := int(pj.(*leabra.Prjn).RConIdxSt[ri])

				recv_out.N = nc

				si_arr := make([]int, 0, nc)
				for ci := 0; ci < nc; ci++ {
					si := int(pj.(*leabra.Prjn).RConIdx[st+ci])
					si_arr = append(si_arr, si)
				}
				recv_out.Si = si_arr

				// layerGenStrategy is structured like GenStrategy[receiver][sender]
				strategy := pj.(*leabra.Prjn).WtInit.InitStrategy
				var wt_arr []float32
				if pj.RecvLay().Index() >= pj.SendLay().Index() {
					wt_arr = GenerateWeights(strategy, pj.(*leabra.Prjn), nc, ri)
					recv_out.Wt = wt_arr
				}




				pj_out.Rs = append(pj_out.Rs, recv_out)

			}

			ly_out.Prjns = append(ly_out.Prjns, pj_out)
		}

		net_out.Layers = append(net_out.Layers, ly_out)
	}

	// Symmetrize Weights
	for _, ly := range ss.Net.Layers {
		for _, pj := range ly.(*leabra.Layer).RcvPrjns {
			sendLayIdx := pj.SendLay().Index()
			recvLayIdx := pj.RecvLay().Index()
			sendName := pj.SendLay().(*leabra.Layer).Nm
			recvName := pj.RecvLay().(*leabra.Layer).Nm
			if recvLayIdx > sendLayIdx {
				continue
			}

			// recv = cat
			// send = hid
			// copy weights from the reciprocal connection (initialized earlier)
			recv_prjns := net_out.Layers[recvLayIdx].Prjns // list of prjns in layer
			senderIdx := SliceIndex(len(recv_prjns), func (i int) bool {return recv_prjns[i].From == sendName})
			recv_prjn := recv_prjns[senderIdx]

			// where to copy weights from
			send_prjns := net_out.Layers[sendLayIdx].Prjns // list of prjns in layer
			receiverIdx := SliceIndex(len(send_prjns), func (i int) bool {return send_prjns[i].From == recvName})
			send_prjn := send_prjns[receiverIdx]

			nrecv := len(pj.RecvLay().(*leabra.Layer).Neurons)
			nsend := len(pj.SendLay().(*leabra.Layer).Neurons)
			if sendName != recvName {
				// Symmetrize backward connections
				// for each neuron in layer, create weight array
				for ri := 0; ri < nrecv; ri++ {
					wt_arr := make([]float32, 0, nsend)
					for si := 0; si < nsend; si++ {
						wt_arr = append(wt_arr, send_prjn.Rs[si].Wt[ri])
					}

					// fmt.Println("recv_neuron %+v", recv_neuron)
					recv_prjn.Rs[ri].Wt = wt_arr

				}
			} else {
				// Symmetrize lateral connections
				for ri := 0; ri < nrecv; ri++ {
					for si := 0; si < ri; si++ {
						sym := send_prjn.Rs[si].Wt[ri - 1]
						recv_prjn.Rs[ri].Wt[si] = sym
					}

					// fmt.Println("recv_neuron %+v", recv_neuron)
					// recv_prjn.Rs[ri].Wt = wt_arr

				}

			}
		}
	}



	// fp, err := os.Open("myweightencoding")
	// enc := json.NewEncoder(fp)
	// enc.Encode(net_out)
	file, _ := json.Marshal(net_out)

	_ = ioutil.WriteFile(ss.DataDir + "/tesselspec.wts", file, 0644)




	return
}

func (ss *Sim) ReloadStimFiles() (err error) {
	ss.CurrentStimFile = fmt.Sprintf("%s/color_diff_stim.dat", ss.StimulusDir)
	// Start reading from the file with a reader.
	file, err := ioutil.ReadFile(ss.CurrentStimFile)
	if err != nil {
		log.Fatalln(err)
	}

	//hiddentohidden_prjn := ss.getPrjn("Hidden", "Hidden")
	hiddentooutput_prjn := ss.getPrjn("Hidden", "Output")
	//var NumUniqueUnits = hiddentohidden_prjn.WtInit.NumTotalUnits - hiddentohidden_prjn.WtInit.NumOverlapUnits
	SameDiffCondition := hiddentooutput_prjn.WtInit.SameDiffCondition
	var minunitid_item1, maxunitid_item1, zero_rest_item1, minunitid_item2, maxunitid_item2, zero_rest_item2 int
	if SameDiffCondition == "Different" {
		minunitid_item1 = 25
		maxunitid_item1 = 1
		zero_rest_item1 = 50 - maxunitid_item1 - minunitid_item1
		minunitid_item2 = 29
		maxunitid_item2 = 1
		zero_rest_item2 = 50 - maxunitid_item2 - minunitid_item2
	} else if SameDiffCondition == "Same" {
		minunitid_item1 = 25
		maxunitid_item1 = 1
		zero_rest_item1 = 50 - maxunitid_item1 - minunitid_item1
		minunitid_item2 = 25
		maxunitid_item2 = 1
		zero_rest_item2 = 50 - maxunitid_item2 - minunitid_item2
	}



    lines := strings.Split(string(file), "\n")
	// Write into file
	for i, line := range lines {
		if i == 1 {
			writeline := line[:27] + strings.Repeat("0\t", minunitid_item1)
			writeline += strings.Repeat("0.9\t", maxunitid_item1)
			writeline += strings.Repeat("0\t", zero_rest_item1 - 1)
			writeline += "0"
			lines[i] = writeline

		} else if i == 2 {
			writeline := line[:27] + strings.Repeat("0\t", minunitid_item2)
			writeline += strings.Repeat("0.9\t", maxunitid_item2)
			writeline += strings.Repeat("0\t", zero_rest_item2 - 1)
			writeline += "0"
			lines[i] = writeline
		}
	}
	output := strings.Join(lines, "\n")
	write_name := fmt.Sprintf("%s/color_diff_stim_%s.dat", ss.StimulusDir, SameDiffCondition)
    err = ioutil.WriteFile(write_name, []byte(output), 0644)
	// err = ioutil.WriteFile(ss.CurrentStimFile, []byte(output), 0644)

    if err != nil {
		log.Fatalln(err)
	}
    return
}
