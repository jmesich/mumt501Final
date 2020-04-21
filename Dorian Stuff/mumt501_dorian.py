import scipy
import wave
import numpy as np
import math
from scipy.io import wavfile
#import matplotlib.pyplot as plt
import sys
import wavio


def window(M):
    """
    Function used to return a Hamming window of leangth M.
    """
    w = np.zeros(M)

    for x in range(M):
        w[x] = (1 / (4 * 0.54)) * (0.54 - 0.46 * np.cos(2 * math.pi * (x / M)))

    return w


def padding(channel, Nw, Nh, N):
    """
    Functions used to pad whole signal.
    """
    z = np.zeros(Nw)

    # First pad (beginning and end)
    padded_signal = np.concatenate((z, channel, z))

    # Second pad: only end
    end_pad = math.ceil((N + Nw) / Nh)
    end_pad = end_pad * Nh - Nw
    end_pad = end_pad - N
    end = np.zeros(int(end_pad))

    # Whole padded signal
    signal = np.concatenate((padded_signal, end))

    # Get signal length
    l = len(signal)

    return signal, l, end_pad


def framing(signal, Nh, Nw):
    """
    Function used to split signal into frames.
    """
    index = 0
    stop = signal.shape[0]
    frames = []

    while (index < stop - 3 * Nh):
        temp = signal[index:index + Nw]
        frames.append(temp)
        index = index + Nh
    
    frames = np.asarray(frames)

    return frames


def dot(x, i_minx, i_maxx, y, i_miny, i_maxy):
    """
    Note: this version of the dot product was adapted from the authors' methodology.

    It does not follow standard dot product formulation, but is what they use to calculate
    the correlation vector for AR parameter estimation.
    """
    value = 0
    N = i_maxx - i_minx + 1

    for i in range(0, N, 1):
        value += x[i + i_minx] * y[i + i_miny]

    return value


def levinson_devin(signal, p, Nw):
    """
    Levinson-Durbin algorithm for estimating AR parameters.
    """

    # Calculate correlation vector first
    R = np.zeros(p + 1)
    for i in range(0, p + 1, 1):
        R[i] = dot(signal, i, Nw - 1, signal, 0, Nw - i - 1) / Nw

    # Allocate vectors
    a = np.zeros(p + 1)
    a_old = np.zeros(p)

    # Initialization

    # Avoid NaN's
    if (math.isclose(R[0], 0.0, rel_tol=1e-09, abs_tol=0.0)):
        a_old[:] = float('nan')
        a[:] = float('nan')
        var = 0.
        return a, a_old, var

    a_old[0] = -(R[1] / R[0])
    a[0] = a_old[0]
    var = (1 - a[0] * a[0]) * R[0]

    # Recursion
    for l in range(1, p, 1):

        s = 0
        for j in range(0, l, 1):
            s = s + a_old[j] * R[l - j]

        # Check for NaN's, if a NaN is present, AR parameters are not valid!
        if (math.isclose(var, 0.0, rel_tol=1e-09, abs_tol=0.0)):
            a_old[:] = float('nan')
            a[:] = float('nan')
            var = 0.
            return a, a_old, var
        
        k = (R[l + 1] + s) / var
        
        a[l] = -k

        var = (1 - k * k) * var

        for j in range(0, l, 1):
            a[j] = a_old[j] - k * a_old[l - j - 1]

        for j in range(0, l + 1, 1):
            a_old[j] = a[j]
        
    # Add parameter 1 at start of AR params and add rest
    a[0] = 1.0
    for j in range(0, p, 1):
        a[j + 1] = a_old[j]

    var = math.sqrt(var)

    return a, a_old, var


def AR_parameters(frames, p, Nw):
    """
    Function used to estimate AR parameters for each frame.
    """
    dim = frames.shape[0]
    a_hat = np.zeros([dim, p + 1])
    var_hat = np.zeros(dim)

    modulus = math.floor(dim / 10)
    percent = 0
    for fr in range(dim):
        if ((fr % modulus) == 0):
            print("--> %i%%" %(percent))
            percent = percent + 10

        frame = frames[fr]
        a, a_old, var = levinson_devin(frame, p, Nw) 
        a_hat[fr, :] = a
        var_hat[fr] = var

    return a_hat, var_hat


def criterion(frames, Nw, p, a_hat):
    """
    Function for calculating the criterion for each frame.
    """
    dim = frames.shape[0]

    # We only want the criterion defined s.t. x_(t-k) can be calculated!
    t_interval = np.linspace(p, Nw - 1, Nw - p - 1, dtype=int)

    dt = np.zeros([dim, Nw], dtype=float)

    modulus = math.floor(dim / 10)
    percent = 0

    for fr in range(dim):
        if ((fr % modulus) == 0):
            print("--> %i%%" %(percent))
            percent = percent + 10

        # a vectors with a NaN should be ignored -- since their variance = 0, 
        # they will not be processed
        if (float('nan') in a_hat[fr, :]):
            dt[fr, :] = 0.

        else:
            # Criterion calculated here.
            for t in t_interval:
                s = 0
                for k in range(0, p, 1):
                    s = s + a_hat[fr, k] * frames[fr, t - k]
                # Use absolute value as that will be used for comparison
                temp = abs(s)
                dt[fr, t] = temp

    return dt


def time_indices(dt, var_hat, K, b):
    """
    Function used to detect burst detections.
    """
    dim = dt.shape[0]

    times = []

    for fr in range(dim):
        dt_temp = dt[fr]
        var = var_hat[fr]

        t = []

        x = 0
        while (x < len(dt_temp)):
            if (dt_temp[x] > K * var):
                t_f = True
                y = x

                while(t_f == True):
                    y = y + 1
                    if (y >= len(dt_temp)):
                        t_f = False
                    elif (dt_temp[y] > K * var):
                        continue
                    else:
                        t_f = False

                if (y - x >= b):
                    if (t == []):
                        t.append((x, y))
                    else:
                        (x_prev, y_prev) = t[-1]

                        # Group bursts if close
                        if (abs(x - y_prev) < b):
                            t.pop()
                            t.append((x_prev, y))
                            print("%i %i %i %i" %(x_prev, y_prev, x, y))
                            print(t[-1])
                        
                        # Just append if not.
                        else:
                            t.append((x, y))

                x = y

            else:
                x = x + 1

        times.append(t)
    
    return times


def cholesky(A, x_mat, N):
    """
    Cholesky decomposition as is explained in the paper.
    """
    L = np.zeros([N, N])
    d = np.zeros(N)
    v = np.zeros(N)

    for j in range(0, N, 1):
        if (j > 0):
            for i in range(0, j, 1):
                v[i] = L[j, i] * d[i]
            v[j] = A[j, j] - dot(L[j, :], 0, j - 1, v, 0, j - 1)

        else:
            v[j] = A[j, j]

        d[j] = v[j]

        if (v[j] == 0):
            sys.exit("Singular Matrix!!!")

        if (j < N - 1):
            for i in range(j + 1, N, 1):
                L[i, j] = (A[i, j] - dot(L[i, :], 0, j - 1, v, 0, j - 1)) / v[j]
        
        L[j, j] = 1

    return x_mat


def cholesky_reconstruct(frames, p, Nw, a_hat, times):
    """
    Reconstruct missing samples!
    """
    index = 0
    for t in times:
        print(index)

        if (t == []):
            index = index + 1
            continue

        else:
            temp_frame = frames[index]
            temp_a = a_hat[index]

            # if a parameters are NaN,s, skip
            if (float('nan') in temp_a[:]):
                index = index + 1
                continue

            values = np.zeros(len(temp_frame), dtype=int)

            for tup in t:
                for x in range(tup[0], tup[1], 1):
                    values[x] = 1

            values[0:p] = 0
            values[Nw - p:Nw] = 0

            # Number of samples to reconstruct.
            l = int(np.sum(values))

            if (l > 0):

                b = np.zeros(p + 1)
                B = np.zeros([l, l])
                d = np.zeros(l)
                t = np.zeros(l, dtype=int)

                # Indices of all missing samples.
                temp_index = 0
                for x in range(len(values)):
                    if (values[x] == 1):
                        t[temp_index] = x
                        temp_index = temp_index + 1

                # Construct b vector for B
                for i in range(0, p + 1, 1):
                    b[i] = 0.0
                    for j in range(i, p + 1, 1):
                        b[i] = b[i] + temp_a[j] * temp_a[j - i]

                # Construct B vector.
                for i in range(0, l, 1):
                    for j in range(i, l, 1):
                        if (abs(t[i] - t[j]) < p + 1):
                            B[i, j] = b[abs(t[i] - t[j])]
                            B[j, i] = b[abs(t[i] - t[j])]

                # Construct -d vector.
                for i in range(0, l, 1):
                    d[i] = 0
                    for j in range(-p, p + 1, 1):
                        if ((t[i] - j) in t):
                            continue
                        else:
                            d[i] = d[i] - b[abs(j)] * temp_frame[t[i] - j]

                #x_mat = np.zeros(l)

                s_t = cholesky(B, d, l)

                #if (x_mat_new == False):
                #index = index + 1
                #continue
                """
                L, D, perm = scipy.linalg.ldl(B, lower=True, hermitian=False, overwrite_a=True, check_finite=True)
                #x_mat = scipy.linalg.ldl(B, lower=True, hermitian=True, overwrite_a=False, check_finite=True)

                x_mat = np.linalg.solve(L, d)

                L_t = np.transpose(L)
                D_inv = np.linalg.inv(D)
                D_x = np.matmul(D_inv, x_mat)

                s_t = np.linalg.solve(L_t, D_x)
                """

                print(s_t.shape)

                #print(x_mat_new)

                for c in range(l):
                    temp_frame[t[c]] = s_t[c]

                frames[index, :] = temp_frame

            index = index + 1

    return frames


def remove_noise(sound_file, K, b, p, Nw, Niter, overlap):
    """
    Parameters:
    -- sound_file:
    -- K:
    -- b:
    -- p:
    -- Nw: 
    -- Niter:
    -- overlap: 
    """

    # Read wav file
    wv = wavio.read(sound_file)
    rate = wv.rate
    sampwidth = wv.sampwidth
    #arr = wv.data
    #rate, sampwidth, arr = wavio.read(sound_file)

    wv_shape = wv.data.shape
    arr = np.zeros(wv_shape)

    for i in range(0, wv_shape[0], 1):
        for j in range(0, wv_shape[1], 1):
            arr[i, j] = wv.data[i, j]

    #arr = arr[20000:40000, :]

    new_arr = np.zeros(arr.shape, dtype=float)

    # Number of channels and samples
    N = arr.shape[0]
    num_ch = arr.shape[1]

    # Compute Nh using overlap
    Nh = int(Nw - Nw * overlap)

    for ch in range(num_ch):
        """
        Read each channel separately.
        """
        print("Channel: %i" %(ch))

        channel = arr[:, ch]
        print(channel.shape)

        print("--> Starting Algorithm.")

        for x in range(Niter):
            """
            Algorithm starts here.
            """
            print("--> Iteration %i" %(x))

            # Step 1: Pad the signal with zeroes
            print("--> Padding.")

            signal, p_l, upper = padding(channel, Nw, Nh, N)
            print(p_l)
            print(upper)

            # Step 2: Divide Signal into overlapping frames
            print("--> Dividing signal into overlapping frames.")

            frames = framing(signal, Nh, Nw)

            # Step 3: Estimate the AR parameters
            print("--> Estimating AR parameters.")

            a_hat, var_hat = AR_parameters(frames, p, Nw)

            # Step 4: Calculate the d_t on each frame
            print("--> Calculating the criterion for each frame.")

            dt = criterion(frames, Nw, p, a_hat)

            # Step 4.5: detect corrupt signal time indices
            print("--> Detecting time indices for corrupt signals.")

            times = time_indices(dt, var_hat, K, b)

            # Step 5: reconstruct signal
            print("--> Reconstructing signal.")

            frames = cholesky_reconstruct(frames, p, Nw, a_hat, times)

            # Step 6: window every frame
            print("--> Windowing.")

            w = window(Nw)

            for fr in range(frames.shape[0]):
                frames[fr, :] = np.multiply(frames[fr, :], w)

            # Step 7: Add frames up again for new signal!
            print("--> Adding frames for reconstruction.")

            new_signal = np.zeros(p_l)
            index = 0

            for fr in range(frames.shape[0]):
                new_signal[index:index + Nw] = new_signal[index:index + Nw] + frames[fr, :]
                index = index + Nh

            print(new_signal.shape)
            print(channel.shape)

            channel = new_signal[Nw:p_l - upper - Nw]

            print(channel.shape)

            print("Done!")

        arr[:, ch] = channel[:]

    return arr, rate, sampwidth


def main():
    data, rate, sampwidth = remove_noise('Test_Samples/sampling_101.wav', 2, 20, 302, 2416, 2, 0.75)

    wavio.write('restored_sampling_101.wav', data, rate, scale=None, sampwidth=sampwidth)


main()