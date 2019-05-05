# CSharpWpfAIExamples
This repository contains C# written Deep Neural Network (DNN) image processing WPF examples. All examples are very simple and clear to know. There are very few project in this repo now but I will expand the number of projects continuously as my time allows.
# Projects
- FaceDetection: 
  - Use: Emgu.CV.Dnn and res10_300x300_ssd_iter_140000.caffemodel
  - Example: https://youtu.be/LBLWCOjqfqE
- MultiPerson2DPoseEstimation
  - Use: TensorFlow and frozen_model.bytes (75)
  - Example: https://youtu.be/A8VtRkL1L-A 
  - Requirements: https://github.com/migueldeicaza/TensorFlowSharp/
- CameraObjectDetection:
  - Use: Emgu.CV.Dnn and ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03 but it dos not too accurate.
         For more accuracy you can use all weights and configs from https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API 
         site.
  - Example: ... in progress
# All projects requirements
  - Visual Studio C#
  - .NET 4.7.2
  - Emgu CV https://www.nuget.org/packages/EMGU.CV/4.0.1.3373
# Usage
- Download this repo
- Open in Visual Studio EmguCVAIExamples.sln
- It can miss some dependencies: 
    - In Visual Studio go to Tools/NuGet Package Manager/Manage NuGet Packages for Solution...
    - Refactor all packages.
- Done. Set startup project and run.

The CommonCollection project is for general using for all projects. It is part of all projects (it will be an auto compiled dll file).
