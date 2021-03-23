package benchmark;

import boofcv.alg.misc.ImageMiscOps;
import boofcv.factory.filter.kernel.FactoryKernel;
import boofcv.struct.convolve.Kernel1D_F32;
import boofcv.struct.image.GrayF32;
import boofcv.struct.image.GrayU8;
import boofcv.testing.BoofTesting;
import org.ejml.UtilEjml;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Test;

import java.util.Random;

class ImageProcessingTest {
    int imageSize = 100;
    int kernelSize = 11;
    Random rand = new Random(345);

    @Test void horizontal_vector() {
        GrayF32 src = new GrayF32(imageSize, imageSize);
        GrayF32 expected = src.createSameShape();
        GrayF32 found = src.createSameShape();

        Kernel1D_F32 kernel = FactoryKernel.random1D_F32(kernelSize,kernelSize/2,0.0f,1.0f,rand);

        ImageMiscOps.fillUniform(src, rand, -1, 1);

        ImageProcessing.horizontal(kernel, src, expected);
        ImageProcessing.horizontal_vector(kernel, src, found);

        BoofTesting.assertEquals(expected, found, UtilEjml.TEST_F32);
    }

    @Disabled
    @Test void threshold_vector_v1() {
        GrayU8 src = new GrayU8(imageSize, imageSize);
        GrayU8 expected = src.createSameShape();
        GrayU8 found = src.createSameShape();

        ImageMiscOps.fillUniform(src, rand, 0, 255);

        ImageProcessing.threshold(src, expected, 125);
        ImageProcessing.threshold_vector_v1(src, found, 125);

        BoofTesting.assertEquals(expected, found, UtilEjml.TEST_F32);
    }

    @Disabled
    @Test void threshold_vector_v2() {
        GrayU8 src = new GrayU8(imageSize, imageSize);
        GrayU8 expected = src.createSameShape();
        GrayU8 found = src.createSameShape();

        ImageMiscOps.fillUniform(src, rand, 0, 255);

        ImageProcessing.threshold(src, expected, 125);
        ImageProcessing.threshold_vector_v2(src, found, 125);

        BoofTesting.assertEquals(expected, found, UtilEjml.TEST_F32);
    }
}