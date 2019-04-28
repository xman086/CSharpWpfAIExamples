# CSharpWpfAIExamples
This repository contains C# written Deep Neural Network (DNN) image processing WPF examples. All examples are very simple and clear to know. There are very few project in this repo now but I will expand the number of projects continuously as my time allows.
# Projects
- FaceDetection: 
  - Using: res10_300x300_ssd_iter_140000.caffemodel
  - Example: https://youtu.be/LBLWCOjqfqE
# Requirements
  - Visual Studio C#
  - .NET 4.7.2
  - Emgu CV https://www.nuget.org/packages/EMGU.CV/4.0.1.3373
# Usage
- Download this repo
- Open in Visual Studio EmguCVAIExamples.sln
- It can miss some dependencies: 
    - In Visual Studio go to Tools/NuGet Package Manager/Manage NuGet Packages for Solutin...
    - Uninstall EMGU.CV and ZedGraph
    - Install EMGU.CV
- Done. Start CameraFaceDetection or other project.

The CommonCollection project is for general using for all projects. It is part of all projects (it will be auto compiled dll file).
