using System.Drawing;
using System.Drawing.Imaging;
using System.Windows.Media;
using System.Windows.Media.Imaging;

/// <summary>
/// Class of all extension methods.
/// </summary>
public static class Extensions
{
    /// <summary>
    /// Convert Bitmap image to (WPF/Image control source) BitmapSource.
    /// </summary>
    /// <param name="bitmap"></param>
    /// <returns></returns>
    public static BitmapSource BitmapToBitmapSource(this Bitmap bitmap, System.Windows.Media.PixelFormat? pixelFormat = null)
    {
        if (bitmap == null)
        {
            return null;
        }

        if (pixelFormat is null)
        {
            switch (bitmap.PixelFormat)
            {
                // It contains only the most commonly used types:
                // You can get an exception for example if you use
                // a grayscale image (if you don't have a pixelFormat value).
                case System.Drawing.Imaging.PixelFormat.Format24bppRgb:
                    {
                        pixelFormat = PixelFormats.Bgr24;
                        break;
                    }
                case System.Drawing.Imaging.PixelFormat.Format32bppRgb:
                    {
                        pixelFormat = PixelFormats.Bgr32;
                        break;
                    }
                case System.Drawing.Imaging.PixelFormat.Format32bppArgb:
                    {
                        pixelFormat = PixelFormats.Bgra32;
                        break;
                    }
                default:
                    break;
            }
        }
        
        BitmapData bitmapData = bitmap.LockBits(new Rectangle(0, 0, bitmap.Width, bitmap.Height),
            ImageLockMode.ReadOnly, bitmap.PixelFormat);

        BitmapSource bitmapSource = BitmapSource.Create(
            bitmapData.Width,
            bitmapData.Height,
            bitmap.HorizontalResolution,
            bitmap.VerticalResolution,
            (System.Windows.Media.PixelFormat)pixelFormat,
            null,
            bitmapData.Scan0,
            bitmapData.Stride * bitmapData.Height,
            bitmapData.Stride);

        bitmap.UnlockBits(bitmapData);
        bitmapSource.Freeze();

        return bitmapSource;
    }
}