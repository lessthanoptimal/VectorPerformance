Proof of concept test using new Vector API ([JEP 338](https://openjdk.java.net/jeps/338)). Vectorized code is compared against already optimized code from 
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
* Ubuntu 18.04.5 LTS

Summary
```
Operation                     | Data |     Size     |     |   Relative  |
                              | Type |              |     | Performance |
-----------------------------------------------------------------------------------------
Matrix Mult IKJ               |   D  | Large Matrix | [1] |     1.84    |
Matrix Mult IKJ               |   D  | Small Matrix [ [2] |      .86    |
Matrix Mult IKJ Complex       |   D  | Large Matrix |     |             | Vector code needed
Matrix Mult IKJ Complex       |   D  | Small Matrix [     |             | Vector code needed
Image 1D Conv                 |   F  | Large kernel |     |     1.82    | 
Image 1D Conv                 |   F  | Small kernel |     |     1.86    |
Image 1D Conv  BoofCV         |   F  | Small kernel | [3] |      .41    | Compared to unrolled
Image 1D Mean                 |   F  |              |     |             | Vector code needed
Image Threshold               |  U8  |              | [4] |     6.78    |
Image Histogram               |  U16 |              |     |             | Vector code needed
```

Unless otherwise stated, all performance is baseline code over vectorized code. Values > 1 mean vectorized code was
faster and values < 1 mean vectorized was slower. In some cases unrolled code from EJML and BoofCV have been
included to provide a point of comparison.

```
Benchmark                                       (kernelSize)  (size)  Mode  Cnt           Score           Error  Units
BenchmarkOperations.convolve_horizontal                    5     N/A  avgt    5     8080015.121 ±    169251.559  ns/op
BenchmarkOperations.convolve_horizontal                   31     N/A  avgt    5    24767084.561 ±    462767.053  ns/op
BenchmarkOperations.convolve_horizontal_boofcv             5     N/A  avgt    5     1775128.816 ±     13315.269  ns/op
BenchmarkOperations.convolve_horizontal_boofcv            31     N/A  avgt    5    24833110.727 ±    265814.061  ns/op
BenchmarkOperations.convolve_horizontal_vector             5     N/A  avgt    5     4351633.285 ±     29120.843  ns/op
BenchmarkOperations.convolve_horizontal_vector            31     N/A  avgt    5    13615354.944 ±    263422.696  ns/op
BenchmarkOperations.image_threshold                      N/A     N/A  avgt    5      345424.878 ±      6195.410  ns/op
BenchmarkOperations.image_threshold_vector_v1            N/A     N/A  avgt    5      580158.660 ±      8812.190  ns/op
BenchmarkOperations.image_threshold_vector_v2            N/A     N/A  avgt    5       50925.242 ±      2032.203  ns/op
BenchmarkOperations.matrix_mult                          N/A       4  avgt    5         104.410 ±         5.414  ns/op
BenchmarkOperations.matrix_mult                          N/A    1000  avgt    5   606881005.900 ±   3875032.130  ns/op
BenchmarkOperations.matrix_mult_complex                  N/A       4  avgt    5         202.456 ±         1.172  ns/op
BenchmarkOperations.matrix_mult_complex                  N/A    1000  avgt    5  1616543112.600 ± 500480900.857  ns/op
BenchmarkOperations.matrix_mult_ejml                     N/A       4  avgt    5          84.286 ±         2.964  ns/op
BenchmarkOperations.matrix_mult_ejml                     N/A    1000  avgt    5   611016316.100 ±  13669452.444  ns/op
BenchmarkOperations.matrix_mult_vectors                  N/A       4  avgt    5         121.820 ±         1.471  ns/op
BenchmarkOperations.matrix_mult_vectors                  N/A    1000  avgt    5   329232594.200 ±  10054084.639  ns/op
BenchmarkOperations.mean_horizontal                      N/A     N/A  avgt    5     2188929.877 ±     19603.001  ns/op
```

[1] I would expect a well writen C++ port of that same function to run about 2.5x faster than pure Java on large
matrices. That's about the performance different you get when you compare the top performing pure Java 
libraries against Eigen or LAPACK. The code used is designed for medium sized matrices.

[2] This result isn't surprising. Optimizing for small matrices requires very different approaches than large ones.
One potential improvement for Vector API would be to allow recycling of memory. More hand optimization of the
loops could reduce the gap. While the current API is easy to use it's clobbering the innermost loop with calls to new.
That's a big no in writing high performance code. I could be  wrong, maybe there's some specialized code that 
recognizes what's going on and recycles memory. Small matrix perform is critical in computer vision and signal 
processing.

[3] BoofCV includes code where if the kernel is small, it will invoke code which is unrolled. This typically
results in massive speed up. I wish the JVM was better is at recognizing when to unroll a loop, so I don't
need to write all this auto generated code.

[4] Vector doesn't support unsigned bytes yet and the Vector implementation fails the unit test. Based on comments 
in the JDK looks like that is will be added.

Author: Peter Abeles

https://twitter.com/NotSoOptimal