package benchmark;

import boofcv.struct.convolve.Kernel1D_F32;
import boofcv.struct.image.GrayF32;
import boofcv.struct.image.GrayU8;
import jdk.incubator.vector.ByteVector;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

public class ImageProcessing {
    static final VectorSpecies<Float> FSPECIES_64 = FloatVector.SPECIES_64;
    static final VectorSpecies<Float> FSPECIES = FloatVector.SPECIES_PREFERRED;

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

    public static void horizontal_vector(Kernel1D_F32 kernel,
                                         GrayF32 image, GrayF32 dest ) {
        final float[] dataSrc = image.data;
        final float[] dataDst = dest.data;
        final float[] dataKer = kernel.data;

        final int offset = kernel.getOffset();
        final int kernelWidth = kernel.getWidth();

        final int width = image.getWidth();

        VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

        for( int i = 0; i < image.height; i++ ) {
            int indexDst = dest.startIndex + i*dest.stride + offset;
            int j = image.startIndex + i*image.stride;
            final int jEnd = j+width-(kernelWidth-1);

            for (; j < jEnd; j++) {
                float total = 0;
                int k = 0;
                for (; k < SPECIES.loopBound(kernelWidth); k += SPECIES.length()) {
                    var vsrc = FloatVector.fromArray(SPECIES, dataSrc, j+k);
                    var vker = FloatVector.fromArray(SPECIES, dataKer, k);
                    total += vsrc.mul(vker).reduceLanes(VectorOperators.ADD);
                }
                for (; k < kernelWidth; k++) {
                    total += (dataSrc[j+k])*dataKer[k];
                }
                dataDst[indexDst++] = total;
            }
        }
    }

    public static GrayU8 threshold( GrayU8 input, GrayU8 output, int threshold ) {
        //CONCURRENT_BELOW BoofConcurrency.loopFor(0, input.height, y -> {
        for( int y = 0; y < input.height; y++ ) {
            int indexIn = input.startIndex + y*input.stride;
            int indexOut = output.startIndex + y*output.stride;

            for( int i = input.width; i>0; i-- ) {
                output.data[indexOut++] = (byte)((input.data[indexIn++]& 0xFF) <= threshold ? 1 : 0);
            }
        }
        //CONCURRENT_ABOVE });

        return output;
    }

    public static GrayU8 threshold_vector( GrayU8 input, GrayU8 output, int threshold ) {

        VectorSpecies<Byte> SPECIES = ByteVector.SPECIES_PREFERRED;

        // Vector applies threshold by writing to booleans
        boolean[] tmp = new boolean[input.width];

        for( int y = 0; y < input.height; y++ ) {
            int indexIn = input.startIndex + y*input.stride;
            int indexOut = output.startIndex + y*output.stride;

            int i = 0;
            for(; i < SPECIES.loopBound(input.width); i += SPECIES.length() ) {
                var vinput = ByteVector.fromArray(SPECIES, input.data, indexIn+i);
                vinput.compare(VectorOperators.LE, threshold).intoArray(tmp, i);
                // NOTE: This will yield incorrect results because JDK doesn't support unsigned comparisions
            }
            for (int vectorIdx = 0; vectorIdx < i; vectorIdx++) {
                output.data[indexOut+vectorIdx] = (byte)(tmp[vectorIdx] ? 1 : 0);
            }

            for(; i < input.width; i++ ) {
                output.data[indexOut+i] = (byte)((input.data[indexIn+i]& 0xFF) <= threshold ? 1 : 0);
            }
        }

        return output;
    }
}
