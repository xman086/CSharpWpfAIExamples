using Emgu.CV;
using Emgu.CV.Dnn;
using Emgu.CV.Structure;
using System;
using System.ComponentModel;
using System.Globalization;
using System.Windows;
using System.Windows.Input;

namespace CameraFaceDetection
{
    /// <summary>
    /// Interaction logic for FaceDetectionWindow.xaml
    /// </summary>
    public partial class FaceDetectionWindow : Window
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
        /// resolutionX/detectionSize.
        /// </summary>
        private float xRate = 1.0f;

        /// <summary>
        /// resolutionY/detectionSize.
        /// </summary>
        private float yRate = 1.0f;

        /// <summary>
        /// This class allows to create and manipulate comprehensive artificial neural networks.
        /// </summary>
        private Net net;
        
        /// <summary>
        /// prototxt path.
        /// </summary>
        private string protoPath = "Models\\deploy.prototxt";

        /// <summary>
        /// caffemodel path.
        /// </summary>
        private string caffemodelPath = "Models\\res10_300x300_ssd_iter_140000.caffemodel";
        #endregion End of AI section.
        #endregion End of fields.

        /// <summary>
        /// Face detection.
        /// </summary>
        public FaceDetectionWindow()
        {
            InitializeComponent();

            frame = new Image<Bgr, byte>(resolutionX, resolutionY);
            xRate = resolutionX / (float)detectionSize;
            yRate = resolutionY / (float)detectionSize;
            net = DnnInvoke.ReadNetFromCaffe(protoPath, caffemodelPath);
        }

        #region Methods:
        /// <summary>
        /// Receive an image from camera.
        /// </summary>
        /// <param name="sender"></param>
        /// <param name="e"></param>
        private void Camera_ImageGrabbed(object sender, EventArgs e)
        {
            camera.Retrieve(frame);
            
            //CvInvoke.Flip(frame, frame, Emgu.CV.CvEnum.FlipType.Horizontal);
            Mat blobs = DnnInvoke.BlobFromImage(frame, 1.0, new System.Drawing.Size(detectionSize, detectionSize));
            net.SetInput(blobs);
            Mat detections = net.Forward();

            float[,,,] detectionsArrayInFloats = detections.GetData() as float[,,,];

            for (int i = 0; i < detectionsArrayInFloats.GetLength(2); i++)
            {
                if (Convert.ToSingle(detectionsArrayInFloats[0, 0, i, 2], CultureInfo.InvariantCulture) > 0.4)
                {
                    float Xstart = Convert.ToSingle(detectionsArrayInFloats[0, 0, i, 3], 
                        CultureInfo.InvariantCulture) * detectionSize * xRate;
                    float Ystart = Convert.ToSingle(detectionsArrayInFloats[0, 0, i, 4], 
                        CultureInfo.InvariantCulture) * detectionSize * yRate;
                    float Xend = Convert.ToSingle(detectionsArrayInFloats[0, 0, i, 5], 
                        CultureInfo.InvariantCulture) * detectionSize * xRate;
                    float Yend = Convert.ToSingle(detectionsArrayInFloats[0, 0, i, 6], 
                        CultureInfo.InvariantCulture) * detectionSize * yRate;

                    System.Drawing.Rectangle rect = new System.Drawing.Rectangle
                    {
                        X = (int)Xstart,
                        Y = (int)Ystart,
                        Height = (int)(Yend - Ystart),
                        Width = (int)(Xend - Xstart)
                    };
                    
                    frame.Draw(rect, new Bgr(0, 255, 0), 2);
                }
            }

            Dispatcher.Invoke(new Action(() =>
            {
                img.Source = frame.Bitmap.BitmapToBitmapSource();
            }));
        }
        
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
        private void Window_MouseDown(object sender, System.Windows.Input.MouseButtonEventArgs e)
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
