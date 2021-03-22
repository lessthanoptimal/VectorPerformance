package benchmark;

import boofcv.alg.filter.convolve.ConvolveImageNoBorder;
import boofcv.alg.misc.ImageMiscOps;
import boofcv.concurrency.BoofConcurrency;
import boofcv.factory.filter.kernel.FactoryKernelGaussian;
import boofcv.struct.convolve.Kernel1D_F32;
import boofcv.struct.image.GrayF32;
import boofcv.struct.image.GrayU8;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.RandomMatrices_DDRM;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.runner.Runner;
import org.openjdk.jmh.runner.RunnerException;
import org.openjdk.jmh.runner.options.Options;
import org.openjdk.jmh.runner.options.OptionsBuilder;

import java.util.Random;
import java.util.concurrent.TimeUnit;

@BenchmarkMode(Mode.AverageTime)
@Warmup(iterations = 2)
@Measurement(iterations = 3)
@State(Scope.Benchmark)
@OutputTimeUnit(TimeUnit.NANOSECONDS)
@Fork(value = 1)
public class BenchmarkOperations {
    @State(Scope.Thread)
    public static class MatrixState {
        @Param({"10","1000"})
        public int size;

        DMatrixRMaj A = new DMatrixRMaj(1, 1);
        DMatrixRMaj B = new DMatrixRMaj(1, 1);
        DMatrixRMaj C = new DMatrixRMaj(1, 1);

        @Setup(Level.Trial)
        public void setup() {
            Random rand = new Random(345);

            A.reshape(size, size);
            B.reshape(size, size);
            C.reshape(size, size);

            RandomMatrices_DDRM.fillUniform(A, -1, 1, rand);
            RandomMatrices_DDRM.fillUniform(B, -1, 1, rand);
            RandomMatrices_DDRM.fillUniform(C, -1, 1, rand);
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

    @Benchmark public void matrix_mult_ejml(MatrixState state) {
        MatrixMultiplication.mult_reorder(state.A, state.B, state.C);
    }

    @Benchmark public void matrix_mult_vectors(MatrixState state) {
        MatrixMultiplication.mult_reorder_vector(state.A, state.B, state.C);
    }

    @Benchmark public void convolve_horizontal(FloatImageState state) {
        ImageProcessing.horizontal(state.kernel, state.src, state.dst);
    }

    @Benchmark public void convolve_horizontal_vector(FloatImageState state) {
        ImageProcessing.horizontal_vector(state.kernel, state.src, state.dst);
    }

    @Benchmark public void convolve_horizontal_boofcv(FloatImageState state) {
        // If possible this method will run an unrolled kernel
        ConvolveImageNoBorder.horizontal(state.kernel, state.src, state.dst);
    }

    @Benchmark public void image_threshold(ByteImageState state) {
        ImageProcessing.threshold(state.src, state.dst, 125);
    }

    @Benchmark public void image_threshold_vector(ByteImageState state) {
        ImageProcessing.threshold_vector(state.src, state.dst, 125);
    }

    public static void main(String[] args) throws RunnerException {
        Options opt = new OptionsBuilder()
                .include(BenchmarkOperations.class.getSimpleName())
                .build();
        new Runner(opt).run();
    }
}
