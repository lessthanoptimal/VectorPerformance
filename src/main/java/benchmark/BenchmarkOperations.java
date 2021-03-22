package benchmark;

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
@OutputTimeUnit(TimeUnit.MILLISECONDS)
@Warmup(iterations = 2)
@Measurement(iterations = 3)
@State(Scope.Benchmark)
@Fork(value=1)
public class BenchmarkOperations {
    @Param({"500"})
    public int size;

    DMatrixRMaj A = new DMatrixRMaj(1,1);
    DMatrixRMaj B = new DMatrixRMaj(1,1);
    DMatrixRMaj C = new DMatrixRMaj(1,1);

    @Setup public void setup() {
        Random rand = new Random(345);

        A.reshape(size, size);
        B.reshape(size, size);
        C.reshape(size, size);

        RandomMatrices_DDRM.fillUniform(A,-1,1,rand);
        RandomMatrices_DDRM.fillUniform(B,-1,1,rand);
        RandomMatrices_DDRM.fillUniform(C,-1,1,rand);
    }

    @Benchmark public void ejml() {
        MatrixMultiplication.mult_reorder(A,B,C);
    }

    @Benchmark public void vectors() {
        MatrixMultiplication.mult_reorder_vector(A,B,C);
    }

    public static void main( String[] args ) throws RunnerException {
        Options opt = new OptionsBuilder()
                .include(BenchmarkOperations.class.getSimpleName())
                .build();

        new Runner(opt).run();
    }
}
