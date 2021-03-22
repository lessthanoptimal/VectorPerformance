package benchmark;

import jdk.incubator.vector.DoubleVector;
import jdk.incubator.vector.VectorSpecies;
import org.ejml.data.DMatrix1Row;

/**
 * @author Peter Abeles
 */
public class MatrixMultiplication {
    static final VectorSpecies<Double> SPECIES = DoubleVector.SPECIES_PREFERRED;

    /**
     * Matrix multiplication with IKJ ordering from EJML. This is designed to minimize cache misses and is a
     * top performer in internal benchmarks.
     */
    public static void mult_reorder(DMatrix1Row A, DMatrix1Row B, DMatrix1Row C) {
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

    public static void mult_reorder_simple(DMatrix1Row A, DMatrix1Row B, DMatrix1Row C) {
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

    public static void mult_reorder_vector(DMatrix1Row A, DMatrix1Row B, DMatrix1Row C) {
        C.reshape(A.numRows, B.numCols);

        final int endOfKLoop = B.numRows * B.numCols;

        for (int i = 0; i < A.numRows; i++) {
            int indexCbase = i * C.numCols;
            int indexA = i * A.numCols;

            // need to assign C.data to a value initially
            int indexB = 0;

            double valA = A.data[indexA++];

            for (int j = 0; j < B.numCols; j += SPECIES.length()) {
                var m = SPECIES.indexInRange(i, B.numCols);
                var vb = DoubleVector.fromArray(SPECIES, B.data, j, m);
                var vc = vb.mul(valA);
                vc.intoArray(C.data, indexCbase + j, m);
            }

//            // now add to it
//            for (int k = B.numCols; k < endOfKLoop; k++) {
//                valA = A.data[indexA++];
//
//                for (int j = 0; j < B.numCols; j += SPECIES.length()) {
//                    var m = SPECIES.indexInRange(i, B.numCols);
//                    var vb = DoubleVector.fromArray(SPECIES, B.data, k, m);
//                    var vc = DoubleVector.fromArray(SPECIES, C.data, indexCbase, m);
//                    vb.mul(valA).add(vc).intoArray(C.data,indexCbase);
//                }
//            }
//            while (indexB != endOfKLoop) { // k loop
//                indexC = indexCbase;
//                end = indexB + B.numCols;
//
//                valA = A.data[indexA++];
//
//                while (indexB < end) { // j loop
//                    C.data[indexC++] += valA * B.data[indexB++];
//                }
//            }
        }
    }
}
