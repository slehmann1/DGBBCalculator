## DGBB Calculator: Loadings For Deep Groove Ball Bearings
Determines loadings that may be applied to two deep groove ball bearings mounted onto a shaft and achieve acceptable requirements. Considers static safety factors and basic rating life modified by reliability as defined by [ISO 76:2006](https://www.iso.org/standard/38101.html) and [ISO 281:2007](https://www.iso.org/standard/38102.html) respectively, whilst also following the methodology outlined in the [SKF rolling bearing catalogue](https://www.skf.com/binaries/pub12/Images/0901d196802809de-Rolling-bearings---17000_1-EN_tcm_12-121486.pdf).

### Methodology Overview
This calculator considers two normal tolerance deep groove ball bearings arranged on a shaft in a standard locating/non-locating bearing arrangement as specified by the following free body diagram:

![FBD](https://github.com/slehmann1/DGBBCalculator/blob/main/Supporting%20Info/FreeBodyDiagram.png?raw=true)
The boundary of maximum acceptable axial and radial loads is determined for a specified bearing size. This is determined based on factor of safety against static loading and factor of safety for rating life. The factor of safety for rating life is not modified by a<sub>iso</sub> or a<sub>skf</sub>, but a separate class for determining a<sub>skf</sub> is provided. The boundary is limited by SKF's recommendations for limiting bearing loads under indeterminate load conditions; the final boundary is then plotted.

The acceptable loading boundary is overlaid by points defined by specific load cases. These are determined by converting loads applied to the shaft into radial and axial forces applied to each bearing. Time variant loads are supported through a duty cycle methodology.

An example result is:
![Sample Result](https://github.com/slehmann1/DGBBCalculator/blob/main/Supporting%20Info/SampleResults.png?raw=true)
#### Dependencies
Written in python with the following dependencies:  Numpy, Pandas, and MatPlotLib.

#### Limitations
Whilst this tool can be useful for sizing of deep groove ball bearings, all of the assumptions made within it should be understood. An understanding of bearing theory should be grasped before attempting to use this tool; there are multiple factors that can influence bearing selection outside of the parameters this calculator considers. Some of many examples of these factors include temperature, lubrication, and mounting and operating condition. It would generally be considered poor engineering practice to design cars based on online and unverified tools.
