Proof of concept testing using new Vector API. Vectorized code is compared against already optimized code from EJML and BoofCV.

* Matrix Multiplication IKJ Order
* TODO Image Convolution
* TODO Image Thresholding

To run the benchmark just type the command below.
```bash
./gradlew runtimeBenchmark
```

Here are the results I get on my computer. It's about 2x faster. For matrix multiplication, well written native matrix
multiplication tend to run about 2.5x to 3.5x faster than non Vectorized Java code. So this will close the gap
significantly. Maybe someone who understands this new API better can make it run even faster!

```
Benchmark                    (size)  Mode  Cnt    Score     Error  Units
BenchmarkOperations.ejml       1000  avgt    3  619.884 ±  51.834  ms/op
BenchmarkOperations.vectors    1000  avgt    3  338.798 ± 411.409  ms/op
```

If you load this up in your favorite IDE (in my case IntelliJ) you're highly likely to experience issues. This
is using bleeding edge version of Gradle with a bleeding edge JDK, and a new API.

Author:
Peter Abeles