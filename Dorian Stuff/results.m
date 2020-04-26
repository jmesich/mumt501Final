% Script for comparing sound file outputs

[y, Fs] = audioread("Test_Samples/sampling_101.wav");
%[y1, Fs1] = audioread("Test_Results/restored_sampling101_cholesky.wav");
[y2, Fs2] = audioread("Test_Results/new_cholesky.wav");

figure;
plot(y(:, 1)/norm(y(:, 1)));
%hold on;
%plot(y1(:, 1)/norm(y1(:, 1)));
hold on;
plot(y2(:, 1)/norm(y2(:, 1)));