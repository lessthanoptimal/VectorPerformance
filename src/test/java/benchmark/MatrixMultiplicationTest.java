package benchmark;

import org.ejml.UtilEjml;
import org.ejml.data.DMatrixRMaj;
import org.ejml.data.ZMatrixRMaj;
import org.ejml.dense.row.MatrixFeatures_DDRM;
import org.ejml.dense.row.MatrixFeatures_ZDRM;
import org.ejml.dense.row.RandomMatrices_DDRM;
import org.ejml.dense.row.RandomMatrices_ZDRM;
import org.ejml.sparse.csc.RandomMatrices_DSCC;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.ejml.UtilEjml.assertTrue;

/**
 * @author Peter Abeles
 */
class MatrixMultiplicationTest {
    Random rand = new Random(3453);

    @Test void simpleCompareToEJML() {
        int N = 10;
        DMatrixRMaj A = RandomMatrices_DDRM.rectangle(N, N, rand);
        DMatrixRMaj B = RandomMatrices_DDRM.rectangle(N, N, rand);
        DMatrixRMaj found = RandomMatrices_DDRM.rectangle(N, N, rand);
        DMatrixRMaj expected = found.copy();

        MatrixMultiplication.mult_ikj_simple(A,B,found);
        MatrixMultiplication.mult_ikj(A,B,expected);

        assertTrue(MatrixFeatures_DDRM.isIdentical(found, expected, UtilEjml.TEST_F64));
    }

    @Test void vectorCompareToSimple() {
        int N = 10;
        DMatrixRMaj A = RandomMatrices_DDRM.rectangle(N, N, rand);
        DMatrixRMaj B = RandomMatrices_DDRM.rectangle(N, N, rand);
        DMatrixRMaj found = RandomMatrices_DDRM.rectangle(N, N, rand);
        DMatrixRMaj expected = found.copy();

        MatrixMultiplication.mult_ikj_vector(A,B,found);
        MatrixMultiplication.mult_ikj_simple(A,B,expected);

        assertTrue(MatrixFeatures_DDRM.isIdentical(found, expected, UtilEjml.TEST_F64));
    }

    @Test void vectorCompareToSimple_complex() {
        int N = 10;
        ZMatrixRMaj A = RandomMatrices_ZDRM.rectangle(N, N, rand);
        ZMatrixRMaj B = RandomMatrices_ZDRM.rectangle(N, N, rand);
        ZMatrixRMaj found = RandomMatrices_ZDRM.rectangle(N, N, rand);
        ZMatrixRMaj expected = found.copy();

        MatrixMultiplication.mult_ikj_vector(A,B,found);
        MatrixMultiplication.mult_ikj(A,B,expected);

        assertTrue(MatrixFeatures_ZDRM.isIdentical(found, expected, UtilEjml.TEST_F64));
    }
}