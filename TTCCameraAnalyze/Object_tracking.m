clear;
close;

pictures = [0:17];

ttcLidar = [12.9722, 
            12.264, 
            13.9161, 
            14.8865, 
            7.41552,
            12.4213,
            34.3404,
            9.34376,
            18.1318,
            18.0318,
            14.9877,
            10.1,
            9.22037,
            10.9678,
            8.09422,
            8.81392,
            10.2926,
            8.52001];

ttcCamera_ShiTomasi_Brisk = [13.2775,
                             22.3972,
                             14.3811,
                             16.1216,
                             26.0949,
                             17.6191,
                             19.6355,
                             16.5123,
                             15.4439,
                             13.9839,
                             12.0678,
                             11.5788,
                             12.3592,
                             12.1961,
                             12.5792,
                             11.4024
                             9.54524,
                             11.7254];

ttcCamera_ShiTomasi_Orb = [19.4626,
                           10.7511,
                           16.6755,
                           21.791,
                           39.391,
                           18.4549,
                           173.32,
                           10.9033,
                           1000,
                           34.8393,
                           8.94212,
                           47.5656,
                           9.46076,
                           11.4024,
                           13.6463,
                           10.5063,
                           17.9984,
                           31.3538];

ttcCamera_ShiTomasi_Akaze = [12.3168,
                             14.3973,
                             13.3458,
                             14.409,
                             15.8694,
                             13.8408,
                             15.2979
                             14.3975,
                             14.7335,
                             11.5619,
                             12.1969,
                             11.1809,
                             10.789,
                             10.4216,
                             10.5751,
                             10.1631,
                             9.46152,
                             9.43762];

ttcCamera_Fast_Brisk = [13.2775,
                             22.3972,
                             14.3811,
                             16.1216,
                             26.0949,
                             17.6191,
                             19.6355,
                             16.5123,
                             15.4439,
                             13.9839,
                             12.0678,
                             11.5788,
                             12.3592,
                             12.1961,
                             12.5792,
                             11.4024,
                             9.54524,
                             11.7254];

ttcCamera_Sift_Sift = [11.8355,
                       12.8856,
                       13.2767,
                       18.4705,
                       13.4614,
                       11.5827,
                       13.9362,
                       15.293,
                       13.2573,
                       11.5166,
                       12.3914,
                       11.2922,
                       9.97088,
                       11.3392,
                       9.77498,
                       9.0902,
                       8.79534,
                       8.92371];

% Plot
figure(1)
plot(pictures, ttcLidar);
title("TTC Lidar")
ylabel ("time")
xlabel ("picture")


figure(2)
plot(pictures,ttcCamera_Sift_Sift, pictures, ttcCamera_ShiTomasi_Akaze, pictures, ttcCamera_ShiTomasi_Brisk, pictures, ttcCamera_ShiTomasi_Orb);
title("TTC Camera")
ylabel("time" )
xlabel("picture")
legend("Sift-Sift", "ShiTomasi-Akaze", "ShiTomasi-Brisk", "ShiTomasi-Orb")



