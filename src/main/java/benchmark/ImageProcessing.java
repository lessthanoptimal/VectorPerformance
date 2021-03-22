package benchmark;

import boofcv.struct.convolve.Kernel1D_F32;
import boofcv.struct.image.GrayF32;

public class ImageProcessing {
    public static void horizontal(Kernel1D_F32 kernel,
                                  GrayF32 image, GrayF32 dest ) {
        final float[] dataSrc = image.data;
        final float[] dataDst = dest.data;
        final float[] dataKer = kernel.data;

        final int offset = kernel.getOffset();
        final int kernelWidth = kernel.getWidth();

        final int width = image.getWidth();

        //CONCURRENT_BELOW BoofConcurrency.loopFor(0, image.height, i -> {
        for( int i = 0; i < image.height; i++ ) {
            int indexDst = dest.startIndex + i*dest.stride + offset;
            int j = image.startIndex + i*image.stride;
            final int jEnd = j+width-(kernelWidth-1);

            for (; j < jEnd; j++) {
                float total = 0;
                int indexSrc = j;
                for (int k = 0; k < kernelWidth; k++) {
                    total += (dataSrc[indexSrc++])*dataKer[k];
                }
                dataDst[indexDst++] = total;
            }
        }
        //CONCURRENT_ABOVE });
    }

    public static void vertical( Kernel1D_F32 kernel,
                                 GrayF32 image, GrayF32 dest )
    {
        final float[] dataSrc = image.data;
        final float[] dataDst = dest.data;
        final float[] dataKer = kernel.data;

        final int offset = kernel.getOffset();
        final int kernelWidth = kernel.getWidth();

        final int imgWidth = dest.getWidth();
        final int imgHeight = dest.getHeight();
        final int yEnd = imgHeight - (kernelWidth - offset - 1);

        //CONCURRENT_BELOW BoofConcurrency.loopFor(offset, yEnd, y -> {
        for( int y = offset; y < yEnd; y++ ) {
            int indexDst = dest.startIndex + y*dest.stride;
            int i = image.startIndex + (y - offset)*image.stride;
            final int iEnd = i + imgWidth;

            for (; i < iEnd; i++) {
                float total = 0;
                int indexSrc = i;
                for (int k = 0; k < kernelWidth; k++) {
                    total += (dataSrc[indexSrc])*dataKer[k];
                    indexSrc += image.stride;
                }
                dataDst[indexDst++] = total;
            }
        }
        //CONCURRENT_ABOVE });
    }

    public static void horizontal_vector(Kernel1D_F32 kernel,
                                         GrayF32 image, GrayF32 dest ) {
        final float[] dataSrc = image.data;
        final float[] dataDst = dest.data;
        final float[] dataKer = kernel.data;

        final int offset = kernel.getOffset();
        final int kernelWidth = kernel.getWidth();

        final int width = image.getWidth();

        //CONCURRENT_BELOW BoofConcurrency.loopFor(0, image.height, i -> {
        for( int i = 0; i < image.height; i++ ) {
            int indexDst = dest.startIndex + i*dest.stride + offset;
            int j = image.startIndex + i*image.stride;
            final int jEnd = j+width-(kernelWidth-1);

            for (; j < jEnd; j++) {
                float total = 0;
                int indexSrc = j;
                for (int k = 0; k < kernelWidth; k++) {
                    total += (dataSrc[indexSrc++])*dataKer[k];
                }
                dataDst[indexDst++] = total;
            }
        }
        //CONCURRENT_ABOVE });
    }
}
