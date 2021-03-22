Proof of concept testing using new Vector API ([JEP 338](https://openjdk.java.net/jeps/338)). Vectorized code is compared against already optimized code from 
[EJML](https://ejml.org) and [BoofCV](https://boofcv.org).

* [Matrix Multiplication](https://en.wikipedia.org/wiki/Matrix_multiplication) IKJ Order (double)
* [Image Convolution](https://boofcv.org/index.php?title=Example_Image_Blur) (float)
* [Image Thresholding](https://boofcv.org/index.php?title=Example_Thresholding) (unsigned byte)

To run the benchmark just type the command below. The first time you run it there will be a lot of downloads. If you
don't have JDK 16 installed it will download it for you automatically. Once it starts running the actual benchmark 
that will take about 12 minutes to complete.
```bash
./gradlew runtimeBenchmark
```

If you load this up in your favorite IDE (in my case IntelliJ) you're highly likely to experience issues. This
is using bleeding edge version of Gradle with a bleeding edge JDK, and a new API.

# Results

Setup
* OpenJDK 64-Bit Server VM AdoptOpenJDK (build 16+36, mixed mode, sharing)
* Intel(R) Core(TM) i7-6700 CPU @ 3.40GHz
* Description:	Ubuntu 18.04.5 LTS

Summary
* Matrix Multiplication IKJ Order, Large [1]: 1.71x faster
* Matrix Multiplication IKJ Order, Small [2]: 0.86x slower
* Horizontal convolution with a large kernel: 1.73x faster
* Horizontal convolution with a small kernel: 1.83x faster
* Horizontal convolution with a small kernel [3]: 0.32x slower than unrolling
* Image Threshold [4]: 0.59x slower

```
Benchmark                                       (kernelSize)  (size)  Mode  Cnt          Score           Error  Units
BenchmarkOperations.convolve_horizontal                    5     N/A  avgt    3    7962972.476 ±   5023632.829  ns/op
BenchmarkOperations.convolve_horizontal                   31     N/A  avgt    3   24922124.677 ±   1359366.766  ns/op
BenchmarkOperations.convolve_horizontal_boofcv             5     N/A  avgt    3    1400447.965 ±    421177.616  ns/op
BenchmarkOperations.convolve_horizontal_boofcv            31     N/A  avgt    3   25921277.348 ±  18099895.988  ns/op
BenchmarkOperations.convolve_horizontal_vector             5     N/A  avgt    3    4347001.636 ±    310322.800  ns/op
BenchmarkOperations.convolve_horizontal_vector            31     N/A  avgt    3   14414280.946 ±   5117210.341  ns/op
BenchmarkOperations.image_threshold                      N/A     N/A  avgt    3     342890.999 ±     50719.814  ns/op
BenchmarkOperations.image_threshold_vector               N/A     N/A  avgt    3     580848.853 ±     15860.960  ns/op
BenchmarkOperations.matrix_mult_ejml                     N/A      10  avgt    3        998.899 ±        39.813  ns/op
BenchmarkOperations.matrix_mult_ejml                     N/A    1000  avgt    3  621539857.376 ± 404986392.863  ns/op
BenchmarkOperations.matrix_mult_vectors                  N/A      10  avgt    3       1157.076 ±       390.079  ns/op
BenchmarkOperations.matrix_mult_vectors                  N/A    1000  avgt    3  362752043.709 ± 315620849.774  ns/op
```

[1] I would expect a well writen C++ port of that same function to run about 2.5x faster than pure Java on large
matrices. That's about the performance different you get when you compare the top performing pure Java 
libraries against Eigen or LAPACK.

[2] This result isn't surprising. Optimizing for small matrices requires very different approaches than large ones.
One potential improvement for Vector API would be to allow recycling of memory. More hand optimization of the
loops could reduce the gap. While the current API is easy to use it's clobbering the innermost loop with calls to new.
That's a big no in writing high performance code. I could be  wrong, maybe there's some specialized code that 
recognizes what's going on and recycles memory. Small matrix perform is critical in computer vision and signal 
processing.

[3] BoofCV includes code where if the kernel is small, it will invoke code which is unrolled. This typically
results in massive speed up. I wish the JVM was better is at recognizing when to unroll a loop, so I don't
need to write all this auto generated code.

[4] The large speed hit is mostly caused by Vector outputting a boolean[] which then needs to be converted into a
byte[] array. There are good reasons why BoofCV doesn't use a boolean[] for binary images. Basically it's easier
to write faster code and just easier to write image processing code. The other major issue is that Vector doesn't
support unsigned bytes yet and the Vector implementation fails the unit test. Based on comments in the JDK looks
like that is planned.

Author:
Peter Abeles
https://twitter.com/NotSoOptimal