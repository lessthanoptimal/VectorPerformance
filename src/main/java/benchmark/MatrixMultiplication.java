package benchmark;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;
import org.ejml.data.DMatrix1Row;
import org.ejml.data.ZMatrixRMaj;

/**
 * @author Peter Abeles
 */
public class MatrixMultiplication {
    static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    /**
     * Matrix multiplication with IKJ ordering from EJML. This is designed to minimize cache misses and is a
     * top performer in internal benchmarks. For larger matrices EJML switches to a block multiplication, which
     * is excessively complex for this benchmark.
     */
    public static void mult_ikj(DMatrix1Row A, DMatrix1Row B, DMatrix1Row C) {
        C.reshape(A.numRows, B.numCols);

        // Note to people looking at this code. It might look like there is a bunch of unnecessary hand optimizations.
        // You might be right, but this code goes back to probably Java 1.7 and when you're writing high
        // performance code that needs to be high performance on a wide range of platforms and JVM versions,
        // the general pattern is the more hand holding you do for the JVM the better your code will run even in
        // resource constrained environments.
        //
        // Having said that, the code could probably be cleaned up but doing that and proving it won't cause
        // a performance regression on some ancient JVM on some underpowered ARM processor is not trivial. So if it's
        // not broken don't fix it. Earlier attempts to follow "good practices" have resulted in improvements
        // being reverted, even after careful micro benchmarking.

        final int endOfKLoop = B.numRows * B.numCols;

        for (int i = 0; i < A.numRows; i++) {
            int indexCbase = i * C.numCols;
            int indexA = i * A.numCols;

            // need to assign C.data to a value initially
            int indexB = 0;
            int indexC = indexCbase;
            int end = indexB + B.numCols;

            double valA = A.data[indexA++];

            while (indexB < end) {
                C.data[indexC++] = valA * B.data[indexB++];
            }

            // now add to it
            while (indexB != endOfKLoop) { // k loop
                indexC = indexCbase;
                end = indexB + B.numCols;

                valA = A.data[indexA++];

                while (indexB < end) { // j loop
                    C.data[indexC++] += valA * B.data[indexB++];
                }
            }
        }
    }

    public static void mult_ikj_simple(DMatrix1Row A, DMatrix1Row B, DMatrix1Row C) {
        C.reshape(A.numRows, B.numCols);

        for (int i = 0; i < A.numRows; i++) {
            int indexCbase = i * C.numCols;

            // Initialize the row in C
            {
                double valA = A.data[i * A.numCols];
                for (int j = 0; j < B.numCols; j++) {
                    C.data[indexCbase + j] = valA * B.data[j];
                }
            }

            // Now sum up the final results
            for (int k = 1; k < B.numRows; k++) {
                int indexC = indexCbase;
                int indexB = k * B.numCols;

                double valA = A.data[i * A.numCols + k];
                for (int j = 0; j < B.numCols; j++) {
                    C.data[indexC++] += valA * B.data[indexB++];
                }
            }
        }
    }

    public static void mult_ikj_vector(DMatrix1Row A, DMatrix1Row B, DMatrix1Row C) {
        C.reshape(A.numRows, B.numCols);

        for (int i = 0; i < A.numRows; i++) {
            int indexCbase = i * C.numCols;
            {
                double valA = A.data[i * A.numCols];
                int j;
                for (j = 0; j < SPECIES.loopBound(B.numCols); j += SPECIES.length()) {
                    var vb = DoubleVector.fromArray(SPECIES, B.data, j);
                    vb.mul(valA).intoArray(C.data, indexCbase + j);
                }
                for (; j < B.numCols; j++) {
                    C.data[indexCbase + j] = valA * B.data[j];
                }
            }

            for (int k = 1; k < B.numRows; k++) {
                int indexB = k * B.numCols;

                double valA = A.data[i * A.numCols + k];

                int j;
                for (j = 0; j < SPECIES.loopBound(B.numCols); j += SPECIES.length()) {
                    var vb = DoubleVector.fromArray(SPECIES, B.data, indexB + j);
                    var vc = DoubleVector.fromArray(SPECIES, C.data, indexCbase + j);
                    vc.add(vb.mul(valA)).intoArray(C.data, indexCbase + j);
                }

                for (; j < B.numCols; j++) {
                    C.data[indexCbase + j] += valA * B.data[indexB + j];
                }
            }
        }
    }

    // Matrix multiplication for a complex matrix
    public static void mult_ikj(ZMatrixRMaj A, ZMatrixRMaj B, ZMatrixRMaj C) {
        double realA, imagA;

        int indexCbase = 0;
        int strideA = A.getRowStride();
        int strideB = B.getRowStride();
        int strideC = C.getRowStride();
        int endOfKLoop = B.numRows * strideB;

        for (int i = 0; i < A.numRows; i++) {
            int indexA = i * strideA;

            // need to assign c.data to a value initially
            int indexB = 0;
            int indexC = indexCbase;
            int end = indexB + strideB;

            realA = A.data[indexA++];
            imagA = A.data[indexA++];

            while (indexB < end) {
                double realB = B.data[indexB++];
                double imagB = B.data[indexB++];

                C.data[indexC++] = realA * realB - imagA * imagB;
                C.data[indexC++] = realA * imagB + imagA * realB;
            }

            // now add to it
            while (indexB != endOfKLoop) { // k loop
                indexC = indexCbase;
                end = indexB + strideB;

                realA = A.data[indexA++];
                imagA = A.data[indexA++];

                while (indexB < end) { // j loop
                    double realB = B.data[indexB++];
                    double imagB = B.data[indexB++];

                    C.data[indexC++] += realA * realB - imagA * imagB;
                    C.data[indexC++] += realA * imagB + imagA * realB;
                }
            }
            indexCbase += strideC;
        }
    }

    public static void mult_ikj_vector(ZMatrixRMaj A, ZMatrixRMaj B, ZMatrixRMaj C) {
        double realA, imagA;

        int indexCbase = 0;
        int strideA = A.getRowStride();
        int strideB = B.getRowStride();
        int strideC = C.getRowStride();
        int endOfKLoop = B.numRows * strideB;

        final int speciesLength = SPECIES.length();
        double[] multiRealA = new double[speciesLength];
        double[] multiImagA = new double[speciesLength];

        if (speciesLength % 2 != 0)
            throw new RuntimeException("Code below assumes an even length");

        for (int i = 0; i < A.numRows; i++) {
            int indexA = i * strideA;

            // need to assign c.data to a value initially
            int indexB = 0;
            int indexC = indexCbase;
            int end = indexB + strideB;

            realA = A.data[indexA++];
            imagA = A.data[indexA++];

            for (; indexB < SPECIES.loopBound(B.numCols); indexB += SPECIES.length()) {
                var vb = DoubleVector.fromArray(SPECIES, B.data, indexB);
                vb.mul(realA).intoArray(multiRealA, 0);
                vb.mul(imagA).intoArray(multiImagA, 0);

                // TODO figure out how to use shuffle to re-order the arrays quickly
                for (int j = 0; j < speciesLength; j += 2) {
                    C.data[indexC++] = multiRealA[j] - multiImagA[j + 1];
                    C.data[indexC++] = multiRealA[j + 1] + multiImagA[j];
                }
            }

            while (indexB < end) {
                double realB = B.data[indexB++];
                double imagB = B.data[indexB++];

                C.data[indexC++] = realA * realB - imagA * imagB;
                C.data[indexC++] = realA * imagB + imagA * realB;
            }

            // now add to it
            while (indexB != endOfKLoop) { // k loop
                indexC = indexCbase;
                end = indexB + strideB;

                realA = A.data[indexA++];
                imagA = A.data[indexA++];

                for (; indexB < SPECIES.loopBound(B.numCols); indexB += SPECIES.length()) {
                    var vb = DoubleVector.fromArray(SPECIES, B.data, indexB);
                    vb.mul(realA).intoArray(multiRealA, 0);
                    vb.mul(imagA).intoArray(multiImagA, 0);

                    for (int j = 0; j < speciesLength; j += 2) {
                        C.data[indexC++] += multiRealA[j] - multiImagA[j + 1];
                        C.data[indexC++] += multiRealA[j + 1] + multiImagA[j];
                    }
                }

                while (indexB < end) { // j loop
                    double realB = B.data[indexB++];
                    double imgB = B.data[indexB++];

                    C.data[indexC++] += realA * realB - imagA * imgB;
                    C.data[indexC++] += realA * imgB + imagA * realB;
                }
            }
            indexCbase += strideC;
        }
    }
}
