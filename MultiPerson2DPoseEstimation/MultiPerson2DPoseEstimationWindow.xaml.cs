using Emgu.CV;
using System;
using System.ComponentModel;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Input;
using TensorFlow;
using static PoseNet;

namespace MultiPerson2DPoseEstimation
{
    /// <summary>
    /// Interaction logic for MultiPerson2DPoseEstimationWindow.xaml
    /// </summary>
    public partial class MultiPerson2DPoseEstimationWindow : Window
    {
        #region Fields:
        /// <summary>
        /// Original frame from camera.
        /// </summary>
        private Mat frame = new Mat();

        /// <summary>
        /// The main camera reader.
        /// </summary>
        private VideoCapture camera;

        /// <summary>
        /// Resolution X.
        /// </summary>
        private int resolutionX = 640;

        /// <summary>
        /// Resolution Y.
        /// </summary>
        private int resolutionY = 480;

        /// <summary>
        /// Camera index from the local queue.
        /// You have to chose the camera index.
        /// </summary>
        private int cameraIndex = 0;

        #region AI section:
        /// <summary>
        /// Image size for detection.
        /// </summary>
        private int detectionSize = 337;

        /// <summary>
        /// resolutionX/detectionSize.
        /// </summary>
        private float xRate = 1.0f;

        /// <summary>
        /// resolutionY/detectionSize.
        /// </summary>
        private float yRate = 1.0f;

        /// <summary>
        /// Resized frame.
        /// </summary>
        private Mat resizedFrame = new Mat();

        /// <summary>
        /// Tensorflow model path.
        /// </summary>
        private string modelPath = "Models\\frozen_model.bytes";

        /// <summary>
        /// Main PoseNet object.
        /// </summary>
        private PoseNet posenet = new PoseNet();

        /// <summary>
        /// Tensorflow session.
        /// </summary>
        private TFSession session;

        /// <summary>
        /// Tensorflow model.
        /// </summary>
        private TFGraph graph;

        /// <summary>
        /// Human joint pairs.
        /// </summary>
        string[,] jointPairs = new string[,]
        {
            { "leftWrist", "leftElbow" }, { "leftElbow", "leftShoulder" }, { "leftShoulder", "rightShoulder" },
            { "rightShoulder", "rightElbow" }, { "rightElbow", "rightWrist" }, { "leftShoulder", "leftHip" }, 
            { "rightShoulder", "rightHip" }, { "leftHip", "rightHip" }, { "leftHip", "leftKnee" },
            { "leftKnee", "leftAnkle" }, { "rightHip", "rightKnee" }, { "rightKnee", "rightAnkle" },
            { "nose", "leftEye" }, { "leftEye", "leftEar" }, { "nose", "rightEye" }, { "rightEye", "rightEar" },
        };

        /// <summary>
        /// Color of joints.
        /// </summary>
        private Pen jointColor = new Pen(Color.Green, 3);

        /// <summary>
        /// Color of skeleton.
        /// </summary>
        private Pen skeletonColor = new Pen(Color.Orange, 5);
        #endregion End of AI section.
        #endregion End of fields.

        /// <summary>
        /// ctor.
        /// </summary>
        public MultiPerson2DPoseEstimationWindow()
        {
            InitializeComponent();
            
            xRate = resolutionX / (float)detectionSize;
            yRate = resolutionY / (float)detectionSize;

            TFSessionOptions TFOptions = new TFSessionOptions();
            
            unsafe
            {
                //It will only work if you install the tensorflow-batteries-windows-x64-gpu NuGet package.
                //WARNING: It just is compatible with NVDA video card which can run CUDA.
                //         You can need to copy the cudnn64_7.dll to the execution folder from the cudnn.zip and install cudnn properly, link below:
                //         https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html
                //
                //https://github.com/migueldeicaza/TensorFlowSharp/issues/206
                byte[] GPUConfig = new byte[] { 0x32, 0x0b, 0x09, 0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0xd3, 0x3f, 0x20, 0x01 };

                fixed (void* ptr = &GPUConfig[0])
                {
                    TFOptions.SetConfig(new IntPtr(ptr), GPUConfig.Length);
                }
            }
            
            graph = new TFGraph();
            graph.Import(File.ReadAllBytes(modelPath));
            session = new TFSession(graph);
        }

        #region Methods:
        /// <summary>
        /// Loaded event handler method.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Window_Loaded(object sender, RoutedEventArgs e)
        {
            this.Width = resolutionX;
            this.Height = resolutionY;
            camera = new VideoCapture(cameraIndex);
            camera.SetCaptureProperty(Emgu.CV.CvEnum.CapProp.FrameWidth, resolutionX);
            camera.SetCaptureProperty(Emgu.CV.CvEnum.CapProp.FrameHeight, resolutionY);
            camera.ImageGrabbed += Camera_ImageGrabbed;
            camera.Start();
        }
        
        /// <summary>
        /// Receive a frame from camera.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Camera_ImageGrabbed(object sender, EventArgs e)
        {
            if (camera.Retrieve(frame))
            {
                CvInvoke.Flip(frame, frame, Emgu.CV.CvEnum.FlipType.Horizontal);
                CvInvoke.Resize(frame, resizedFrame, new System.Drawing.Size(detectionSize, detectionSize), 0, 0);

                TFTensor tensor = TransformInput(resizedFrame.Bitmap);
                TFSession.Runner runner = session.GetRunner();

                runner.AddInput(graph["image"][0], tensor);
                runner.Fetch(
                    graph["heatmap"][0],
                    graph["offset_2"][0],
                    graph["displacement_fwd_2"][0],
                    graph["displacement_bwd_2"][0]
                );

                var result = runner.Run();
                var heatmap = (float[,,,])result[0].GetValue(jagged: false);
                var offsets = (float[,,,])result[1].GetValue(jagged: false);
                var displacementsFwd = (float[,,,])result[2].GetValue(jagged: false);
                var displacementsBwd = (float[,,,])result[3].GetValue(jagged: false);

                Pose[] poses = posenet.DecodeMultiplePoses(
                                           heatmap, offsets,
                                           displacementsFwd,
                                           displacementsBwd,
                                           outputStride: 16, maxPoseDetections: 100,
                                           scoreThreshold: 0.5f, nmsRadius: 20);

                Drawing(frame, poses);

                Dispatcher.Invoke(new Action(() =>
                {
                    img.Source = frame.Bitmap.BitmapToBitmapSource();
                }));
            }
        }

        /// <summary>
        /// Draw the poses.
        /// </summary>
        /// <param name="image"></param>
        /// <param name="poses"></param>
        /// <param name="poseChain"></param>
        private void Drawing(Mat image, Pose[] poses, bool poseChain = false)
        {
            if (poses.Length > 0)
            {
                using (Graphics g = Graphics.FromImage(frame.Bitmap))
                {
                    for (int i = 0; i < poses.Length; i++)
                    {
                        Pose pose = poses[i];

                        if (pose.score > 0.15f)
                        {
                            for (int j = 0; 
                                j < (poseChain ? posenet.poseChain.GetLength(0) : jointPairs.GetLength(0)); 
                                j++)
                            {
                                Keypoint point1 = pose.keypoints.FirstOrDefault(item => item.part.Equals(
                                    poseChain ? posenet.poseChain[j].Item1 : jointPairs[j,0]));
                                Keypoint point2 = pose.keypoints.FirstOrDefault(item => item.part.Equals(
                                    poseChain ? posenet.poseChain[j].Item2 : jointPairs[j, 1]));
                          
                                if (!point1.IsEmpty && point1.score >= 0.02)
                                {
                                    if (!point2.IsEmpty && point2.score >= 0.02)
                                    {
                                        g.DrawLine(skeletonColor, point1.position.X * xRate, point1.position.Y * 
                                            yRate, point2.position.X * xRate, point2.position.Y * yRate);
                                    }

                                    g.DrawEllipse(jointColor, point1.position.X * xRate, point1.position.Y * yRate, 3, 3);
                                }

                                if (!point2.IsEmpty && point2.score >= 0.02)
                                {
                                    g.DrawEllipse(jointColor, point2.position.X * xRate, point2.position.Y * yRate, 3, 3);
                                }
                            }
                        }
                    }
                }
            }
        }

        /// <summary>
        /// Convert bitmap data array to TFTensor format.
        /// </summary>
        /// <param name="bitmap"></param>
        /// <returns></returns>
        private TFTensor TransformInput(Bitmap bitmap)
        {
            BitmapData bitmapData = bitmap.LockBits(new Rectangle(0, 0, bitmap.Width, bitmap.Height), ImageLockMode.ReadOnly, bitmap.PixelFormat);
            var length = bitmapData.Stride * bitmapData.Height;

            byte[] bytes = new byte[length];
            
            int strideWithoutReserved = bitmapData.Stride - bitmapData.Reserved;

            Marshal.Copy(bitmapData.Scan0, bytes, 0, length);
            bitmap.UnlockBits(bitmapData);

            float[] floatValues = new float[bitmap.Width * bitmap.Height * 3];

            int idx = 0;

            for (int i = 0; i < bytes.Length; i++)
            {
                if (i == strideWithoutReserved)
                {
                    //Reserved byte.
                    continue;
                }

                if ((i - strideWithoutReserved) % bitmapData.Stride == 0)
                {
                    //Reserved byte.
                    continue;
                }

                floatValues[idx] = bytes[i] * (2.0f / 255.0f) - 1.0f;
                idx++;
            }

            TFShape shape = new TFShape(1, bitmap.Width, bitmap.Height, 3);
            return TFTensor.FromBuffer(shape, floatValues, 0, floatValues.Length);
        }

        /// <summary>
        /// Dispose necessary objects.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Window_Closing(object sender, CancelEventArgs e)
        {
            camera.Stop();
            camera.Dispose();
            session.Dispose();
            graph.Dispose();
        }

        /// <summary>
        /// MouseDown event handler method.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Window_MouseDown(object sender, MouseButtonEventArgs e)
        {
            if (e.LeftButton == MouseButtonState.Pressed)
            {
                DragMove();
            }
        }

        /// <summary>
        /// KeyDown event handler method.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Window_KeyDown(object sender, KeyEventArgs e)
        {
            if (e.Key == Key.Escape)
            {
                Close();
            }
        }

        /// <summary>
        /// MouseDoubleClick event handler method.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Window_MouseDoubleClick(object sender, MouseButtonEventArgs e)
        {
            if (this.WindowState == WindowState.Normal)
            {
                this.WindowState = WindowState.Maximized;
            }
            else
            {
                this.WindowState = WindowState.Normal;
            }
        }
        #endregion End of methods.
    }
}
