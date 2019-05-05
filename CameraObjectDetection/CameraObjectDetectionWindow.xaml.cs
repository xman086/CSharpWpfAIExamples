using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Dnn;
using Emgu.CV.Structure;
using System;
using System.ComponentModel;
using System.Globalization;
using System.IO;
using System.Windows;
using System.Windows.Input;
using Backend = Emgu.CV.Dnn.Backend;

namespace CameraObjectDetection
{
    /// <summary>
    /// Interaction logic for CameraObjectDetectionWindow.xaml
    /// </summary>
    public partial class CameraObjectDetectionWindow : Window
    {
        #region Fields:
        /// <summary>
        /// Original frame from camera.
        /// </summary>
        private Image<Bgr, byte> frame;

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
        private int detectionSize = 300;

        /// <summary>
        /// This class allows to create and manipulate comprehensive artificial neural networks.
        /// </summary>
        private Net net;

        /// <summary>
        /// Tensorflow graph.
        /// ssd_mobilenet_v1_ppn_shared_box_predictor_300x300_coco14_sync_2018_07_03
        /// You can use weights and configs from https://github.com/opencv/opencv/wiki/TensorFlow-Object-Detection-API
        /// Note the coco_labels.txt will be wrong to RCCN weights and configs.
        /// </summary>
        private string graphPath = @"Models\frozen_inference_graph.pb";

        /// <summary>
        /// Tensorflow graph config.
        /// </summary>
        private string configPath = @"Models\config.pbtxt";

        /// <summary>
        /// Label texts.
        /// </summary>
        private string labelPath = @"Models\coco_labels.txt";

        /// <summary>
        /// Label array.
        /// </summary>
        private string[] labels;
        #endregion End of AI section.
        #endregion End of fields.

        public CameraObjectDetectionWindow()
        {
            InitializeComponent();

            frame = new Image<Bgr, byte>(resolutionX, resolutionY);
            net = DnnInvoke.ReadNetFromTensorflow(graphPath, configPath);
            labels = File.ReadAllLines(labelPath);
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
            camera.SetCaptureProperty(CapProp.FrameWidth, resolutionX);
            camera.SetCaptureProperty(CapProp.FrameHeight, resolutionY);
            net.SetPreferableTarget(Target.Cpu); // Target.OpenCL, ... (depend on graphic card)
            net.SetPreferableBackend(Backend.OpenCV);
            camera.ImageGrabbed += Camera_ImageGrabbed;
            camera.Start();
        }

        /// <summary>
        /// Receve an image from camera.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Camera_ImageGrabbed(object sender, EventArgs e)
        {
            camera.Retrieve(frame);
            Mat blobs = DnnInvoke.BlobFromImage(frame, 1.0, new System.Drawing.Size(detectionSize, detectionSize), swapRB: true);
            net.SetInput(blobs);
            Mat outp = net.Forward();
            float[,,,] boxes = outp.GetData() as float[,,,];

            for (int i = 0; i < boxes.GetLength(2); i++)
            {
                int classID = Convert.ToInt32(boxes[0, 0, i, 1]);
                
                float confidence = Convert.ToSingle(
                    boxes[0, 0, i, 2].ToString().Replace(",", "."), CultureInfo.InvariantCulture);

                if (confidence < 0.6)
                {
                    continue;
                }

                float Xstart = Convert.ToSingle(
                    boxes[0, 0, i, 3].ToString().Replace(",", "."), CultureInfo.InvariantCulture) * resolutionX;
                float Ystart = Convert.ToSingle(
                    boxes[0, 0, i, 4].ToString().Replace(",", "."), CultureInfo.InvariantCulture) * resolutionY;
                float Xend = Convert.ToSingle(
                    boxes[0, 0, i, 5].ToString().Replace(",", "."), CultureInfo.InvariantCulture) * resolutionX;
                float Yend = Convert.ToSingle(
                    boxes[0, 0, i, 6].ToString().Replace(",", "."), CultureInfo.InvariantCulture) * resolutionY;

                System.Drawing.Rectangle rect = new System.Drawing.Rectangle
                {
                    X = (int)Xstart,
                    Y = (int)Ystart,
                    Height = (int)(Yend - Ystart),
                    Width = (int)(Xend - Xstart)
                };

                string label = labels[classID - 1];

                frame.Draw(rect, new Bgr(0, 255, 0), 2);
                frame.Draw(new System.Drawing.Rectangle((int)Xstart, 
                    (int)Ystart - 35, label.Length * 18, 35), new Bgr(0, 255, 0), -1);
                CvInvoke.PutText(frame, label, new System.Drawing.Point((int)Xstart, 
                    (int)Ystart - 10), FontFace.HersheySimplex, 1.0, new MCvScalar(0, 0, 0), 2);
            }

            Dispatcher.Invoke(new Action(() =>
            {
                img.Source = frame.Bitmap.BitmapToBitmapSource();
            }));
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
            net.Dispose();
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
