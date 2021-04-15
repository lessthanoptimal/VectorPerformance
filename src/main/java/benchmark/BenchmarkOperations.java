package benchmark;

import boofcv.alg.filter.convolve.ConvolveImageNoBorder;
import boofcv.alg.misc.ImageMiscOps;
import boofcv.concurrency.BoofConcurrency;
import boofcv.factory.filter.kernel.FactoryKernelGaussian;
import boofcv.struct.convolve.Kernel1D_F32;
import boofcv.struct.image.GrayF32;
import boofcv.struct.image.GrayU16;
import boofcv.struct.image.GrayU8;
import org.ejml.data.DMatrixRMaj;
import org.ejml.data.ZMatrixRMaj;
import org.ejml.dense.row.CommonOps_DDRM;
import org.ejml.dense.row.RandomMatrices_DDRM;
import org.ejml.dense.row.RandomMatrices_ZDRM;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;
import org.openjdk.jmh.runner.options.TimeValue;

import java.util.Random;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@Warmup(iterations = 2)
@Measurement(iterations = 5)
@State(Scope.Benchmark)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Fork(value = 1)
public class BenchmarkOperations {
    final static int MAX_PIXEL_U16 = 2000;

    @State(Scope.Thread)
    public static class MatrixState {
        @Param({"4","1000"}) // 4x4 matrices are very common in computer vision. 1000 is a medium sized matrix
        public int size;

        DMatrixRMaj A = new DMatrixRMaj(1, 1);
        DMatrixRMaj B = new DMatrixRMaj(1, 1);
        DMatrixRMaj C = new DMatrixRMaj(1, 1);

        ZMatrixRMaj CA = new ZMatrixRMaj(1, 1);
        ZMatrixRMaj CB = new ZMatrixRMaj(1, 1);
        ZMatrixRMaj CC = new ZMatrixRMaj(1, 1);

        @Setup(Level.Trial)
        public void setup() {
            Random rand = new Random(345);

            A.reshape(size, size);
            B.reshape(size, size);
            C.reshape(size, size);
            RandomMatrices_DDRM.fillUniform(A, -1, 1, rand);
            RandomMatrices_DDRM.fillUniform(B, -1, 1, rand);
            RandomMatrices_DDRM.fillUniform(C, -1, 1, rand);

            CA.reshape(size, size);
            CB.reshape(size, size);
            CC.reshape(size, size);
            RandomMatrices_ZDRM.fillUniform(CA, -1, 1, rand);
            RandomMatrices_ZDRM.fillUniform(CB, -1, 1, rand);
            RandomMatrices_ZDRM.fillUniform(CC, -1, 1, rand);
        }
    }

    @State(Scope.Thread)
    public static class FloatImageState {
        @Param({"5","31"})
        public int kernelSize;

        GrayF32 src = new GrayF32(1200,800);
        GrayF32 dst = src.createSameShape();

        Kernel1D_F32 kernel;

        @Setup(Level.Trial)
        public void setup() {
            // When calling a BoofCV function make sure it doesn't run concurrent code
            BoofConcurrency.USE_CONCURRENT = false;

            Random rand = new Random(345);
            ImageMiscOps.fillUniform(src, rand, 0, 255);

            kernel = FactoryKernelGaussian.gaussian1D(GrayF32.class, -1, kernelSize/2);
        }
    }

    @State(Scope.Thread)
    public static class ByteImageState {
        GrayU8 src = new GrayU8(1200,800);
        GrayU8 dst = src.createSameShape();

        @Setup(Level.Trial)
        public void setup() {
            // When calling a BoofCV function make sure it doesn't run concurrent code
            BoofConcurrency.USE_CONCURRENT = false;

            Random rand = new Random(345);
            ImageMiscOps.fillUniform(src, rand, 0, 255);
        }
    }

    @State(Scope.Thread)
    public static class ShortImageState {
        GrayU16 src = new GrayU16(1200,800);
        GrayU16 dst = src.createSameShape();
        int[] histogram = new int[MAX_PIXEL_U16];

        @Setup(Level.Trial)
        public void setup() {
            // When calling a BoofCV function make sure it doesn't run concurrent code
            BoofConcurrency.USE_CONCURRENT = false;

            Random rand = new Random(345);
            ImageMiscOps.fillUniform(src, rand, 0, MAX_PIXEL_U16);
        }
    }

//    @Benchmark public void matrix_mult_real(MatrixState state) {
//        MatrixMultiplication.mult_ikj(state.A, state.B, state.C);
//    }
//
//    @Benchmark public void matrix_mult_real_ejml(MatrixState state) {
//        // There is specialized code for small matrices here and if large enough a block matrix will kick in
//        CommonOps_DDRM.mult(state.A, state.B, state.C);
//    }
//
//    @Benchmark public void matrix_mult_real_vectors(MatrixState state) {
//        MatrixMultiplication.mult_ikj_vector(state.A, state.B, state.C);
//    }
//
//    @Benchmark public void matrix_mult_complex(MatrixState state) {
//        MatrixMultiplication.mult_ikj(state.CA, state.CB, state.CC);
//    }
//
//    @Benchmark public void matrix_mult_complex_vector(MatrixState state) {
//        MatrixMultiplication.mult_ikj_vector(state.CA, state.CB, state.CC);
//    }
//
//    @Benchmark public void convolve_horizontal(FloatImageState state) {
//        ImageProcessing.horizontal(state.kernel, state.src, state.dst);
//    }
//
//    @Benchmark public void convolve_horizontal_vector(FloatImageState state) {
//        ImageProcessing.horizontal_vector(state.kernel, state.src, state.dst);
//    }
//
//    @Benchmark public void convolve_horizontal_boofcv(FloatImageState state) {
//        // If possible this method will run an unrolled kernel
//        ConvolveImageNoBorder.horizontal(state.kernel, state.src, state.dst);
//    }

    @Benchmark public void mean_horizontal(ByteImageState state) {
        ImageProcessing.mean_horizontal(state.src, state.dst, 5, 11);
    }

    @Benchmark public void mean_horizontal_vector(ByteImageState state) {
        ImageProcessing.mean_horizontal_vector(state.src, state.dst, 5, 11);
    }

//    @Benchmark public void image_threshold(ByteImageState state) {
//        ImageProcessing.threshold(state.src, state.dst, 125);
//    }
//
//    @Benchmark public void image_threshold_vector_v1(ByteImageState state) {
//        ImageProcessing.threshold_vector_v1(state.src, state.dst, 125);
//    }
//
//    @Benchmark public void image_threshold_vector_v2(ByteImageState state) {
//        ImageProcessing.threshold_vector_v2(state.src, state.dst, 125);
//    }
//
//    @Benchmark public void histogram(ShortImageState state) {
//        ImageProcessing.histogram(state.src, 0, state.histogram);
//    }

    public static void main(String[] args) throws RunnerException {
        Options opt = new OptionsBuilder()
                .include(BenchmarkOperations.class.getSimpleName())
                .warmupTime(TimeValue.seconds(1))
                .measurementTime(TimeValue.seconds(1))
                .build();
        new Runner(opt).run();
    }
}
