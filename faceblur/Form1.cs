using System;
using System.Drawing;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Dnn;
using Emgu.CV.Structure;
using Timer = System.Windows.Forms.Timer;

namespace WinFormsApp1
{
    public partial class Form1 : Form
    {
        private VideoCapture capture;
        private Net net;
        private Timer timer;

        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            capture = new VideoCapture(0);
            capture.Set(CapProp.FrameHeight, 480);

            net = DnnInvoke.ReadNetFromCaffe(
                "deploy.prototxt",
                "res10_300x300_ssd_iter_140000.caffemodel"
            );

            timer = new Timer();
            timer.Interval = 30;
            timer.Tick += ProcessFrame;
            timer.Start();
        }

        private void ProcessFrame(object sender, EventArgs e)
        {
            var frame = capture.QueryFrame();
            if (frame == null) return;

            var image = frame.ToImage<Bgr, byte>();

            var blob = DnnInvoke.BlobFromImage(
                image,
                1.0,
                new Size(300, 300),
                new MCvScalar(104, 177, 123),
                false,
                false
            );

            net.SetInput(blob);
            var detections = net.Forward();

            float[,,,] data = (float[,,,])detections.GetData();

            int h = image.Height;
            int w = image.Width;

            for (int i = 0; i < data.GetLength(2); i++)
            {
                float confidence = data[0, 0, i, 2];

                if (confidence > 0.5)
                {
                    int x1 = (int)(data[0, 0, i, 3] * w);
                    int y1 = (int)(data[0, 0, i, 4] * h);
                    int x2 = (int)(data[0, 0, i, 5] * w);
                    int y2 = (int)(data[0, 0, i, 6] * h);

                    var faceRect = new Rectangle(x1, y1, x2 - x1, y2 - y1);
                    faceRect.Intersect(new Rectangle(0, 0, image.Width, image.Height));

                    if (faceRect.Width > 0 && faceRect.Height > 0)
                    {
                        var faceROI = image.GetSubRect(faceRect);

                        var small = faceROI.Resize(0.05, Inter.Nearest);
                        var pixelated = small.Resize(faceROI.Width, faceROI.Height, Inter.Nearest);

                        pixelated.CopyTo(faceROI);
                    }
                }
            }

            pictureBox1.Image?.Dispose();
            pictureBox1.Image = new Bitmap(image.Width, image.Height, image.Mat.Step,
    System.Drawing.Imaging.PixelFormat.Format24bppRgb,
    image.Mat.DataPointer);
        }
    }
}