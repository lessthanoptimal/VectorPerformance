package benchmark;

import org.ejml.UtilEjml;
import org.ejml.data.DMatrixRMaj;
import org.ejml.dense.row.MatrixFeatures_DDRM;
import org.ejml.dense.row.RandomMatrices_DDRM;
import org.junit.jupiter.api.Test;

import java.util.Random;

import static org.ejml.UtilEjml.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

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

        MatrixMultiplication.mult_reorder_simple(A,B,found);
        MatrixMultiplication.mult_reorder(A,B,expected);

        found.print();
        expected.print();

        assertTrue(MatrixFeatures_DDRM.isIdentical(found, expected, UtilEjml.TEST_F64));
    }

    @Test void vectorCompareToEJML() {
        int N = 10;
        DMatrixRMaj A = RandomMatrices_DDRM.rectangle(N, N, rand);
        DMatrixRMaj B = RandomMatrices_DDRM.rectangle(N, N, rand);
        DMatrixRMaj found = RandomMatrices_DDRM.rectangle(N, N, rand);
        DMatrixRMaj expected = found.copy();

        MatrixMultiplication.mult_reorder_simple(A,B,found);
        MatrixMultiplication.mult_reorder_vector(A,B,expected);

        found.print();
        expected.print();

        assertTrue(MatrixFeatures_DDRM.isIdentical(found, expected, UtilEjml.TEST_F64));
    }

    @Test void foo() {
        fail("implement");
    }
}