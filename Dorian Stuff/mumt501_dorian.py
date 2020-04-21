import scipy
import wave
import numpy as np
import math
from scipy.io import wavfile
#import matplotlib.pyplot as plt
import sys

"""
Description.
"""

def _wav2array(nchannels, sampwidth, data):
    """
    First function for reading a wav file.

    All credits go to this link: https://gist.github.com/WarrenWeckesser/7461781
    """

    """data must be the string containing the bytes from the wav file."""

    num_samples, remainder = divmod(len(data), sampwidth * nchannels)
    if remainder > 0:
        raise ValueError('The length of data is not a multiple of '
                         'sampwidth * num_channels.')
    if sampwidth > 4:
        raise ValueError("sampwidth must not be greater than 4.")

    if sampwidth == 3:
        a = np.empty((num_samples, nchannels, 4), dtype=np.uint8)
        raw_bytes = np.frombuffer(data, dtype=np.uint8)
        a[:, :, :sampwidth] = raw_bytes.reshape(-1, nchannels, sampwidth)
        a[:, :, sampwidth:] = (a[:, :, sampwidth - 1:sampwidth] >> 7) * 255
        result = a.view('<i4').reshape(a.shape[:-1])
    else:
        # 8 bit samples are stored as unsigned ints; others as signed ints.
        dt_char = 'u' if sampwidth == 1 else 'i'
        a = np.frombuffer(data, dtype='<%s%d' % (dt_char, sampwidth))
        result = a.reshape(-1, nchannels)

    return result


def readwav(file):
    """
    Second function for reading a wav file.

    All credits go to this link: https://gist.github.com/WarrenWeckesser/7461781
    """

    """
    Read a wav file.
    Returns the frame rate, sample width (in bytes) and a numpy array
    containing the data.
    This function does not read compressed wav files.
    """
    wav = wave.open(file)
    rate = wav.getframerate()
    nchannels = wav.getnchannels()
    sampwidth = wav.getsampwidth()
    nframes = wav.getnframes()
    data = wav.readframes(nframes)
    wav.close()
    array = _wav2array(nchannels, sampwidth, data)

    return rate, sampwidth, array


def writewav24(filename, rate, data):
    """Create a 24 bit wav file.
    data must be "array-like", either 1- or 2-dimensional.  If it is 2-d,
    the rows are the frames (i.e. samples) and the columns are the channels.
    The data is assumed to be signed, and the values are assumed to be
    within the range of a 24 bit integer.  Floating point values are
    converted to integers.  The data is not rescaled or normalized before
    writing it to the file.
    Example: Create a 3 second 440 Hz sine wave.
    >>> rate = 22050  # samples per second
    >>> T = 3         # sample duration (seconds)
    >>> f = 440.0     # sound frequency (Hz)
    >>> t = np.linspace(0, T, T*rate, endpoint=False)
    >>> x = (2**23 - 1) * np.sin(2 * np.pi * f * t)
    >>> writewav24("sine24.wav", rate, x)
    """
    a32 = np.asarray(data, dtype=np.int32)
    if a32.ndim == 1:
        # Convert to a 2D array with a single column.
        a32.shape = a32.shape + (1,)
    # By shifting first 0 bits, then 8, then 16, the resulting output
    # is 24 bit little-endian.
    a8 = (a32.reshape(a32.shape + (1,)) >> np.array([0, 8, 16])) & 255
    wavdata = a8.astype(np.uint8).tostring()

    w = wave.open(filename, 'wb')
    w.setnchannels(a32.shape[1])
    w.setsampwidth(3)
    w.setframerate(rate)
    w.writeframes(wavdata)
    w.close()


# Return a Hamming window of length M
def window(M):
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


#Return the dot product between vectors x and y
def dot(x, i_minx, i_maxx, y, i_miny, i_maxy):
    N = 0
    value = 0
    i = 0
    N = i_maxx - i_minx + 1

    for i in range(0, N, 1):
        value += x[i + i_minx] * y[i + i_miny]

    return value


def levinson_devin(signal, p, Nw):
    """
    Levinson-Durbin algorithm for estimating AR parameters.
    """

    # Calculate input first
    R = np.zeros(p + 1)
    for i in range(0, p + 1, 1):
        """
        v = 0.0
        for j in range(i, Nw, 1):
        v = v + signal[j] * signal[j - 1]
        R[i] = v / Nw
        """
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

        if (math.isclose(var, 0.0, rel_tol=1e-09, abs_tol=0.0)):
            a_old[:] = float('nan')
            a[:] = float('nan')
            var = 0.
            return a, a_old, var
        
        #else:
        k = (R[l + 1] + s) / var
        
        a[l] = -k

        var = (1 - a[l] * a[l]) * var

        for j in range(l - 1, -1, -1):
            a[j] = a_old[j] + a[l] * a_old[l - j - 1]

        for j in range(0, l + 1, 1):
            a_old[j] = a[j]
        
    a[0] = 1.0
    for j in range(0, p, 1):
        a[j + 1] = a_old[j]

    var = math.sqrt(var)

    return a, a_old, var


def AR_parameters(frames, p, Nw):
    """
    Function used to estimate AR parameters.
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
    t_interval = np.linspace(p, Nw - p - 1, Nw - p - p - 1, dtype=int)

    dt = np.zeros([dim, Nw], dtype=float)

    modulus = math.floor(dim / 10)
    percent = 0

    for fr in range(dim):
        if ((fr % modulus) == 0):
            print("--> %i%%" %(percent))
            percent = percent + 10

        if (float('nan') in a_hat[fr, :]):
            dt[fr, :] = 0

        else:
            for t in t_interval:
                s = 0
                for k in range(0, p, 1):
                    s = s + a_hat[fr, k] * frames[fr, t - k]
                temp = abs(frames[fr, t] + s)
                dt[fr, t] = temp

    return dt


def time_indices(dt, var_hat, K, b):
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
                    if (dt_temp[y] > K * var):
                        continue
                    else:
                        t_f = False

                if (y - x > b):
                    t.append((x, y))

                x = y

            else:
                x = x + 1

        times.append(t)
    
    return times


def cholesky(A, x_mat, b, N):
    L = np.zeros([N, N])
    d = np.zeros(N)
    y = np.zeros(N)

    v = 0

    for j in range(0, N, 1):
        d[j] = A[j, j]
        for i in range(0, j, 1):
            d[j] -= L[j, i] * L[j, i] * d[i]
        
        if (math.isclose(d[j], 0.0, rel_tol=1e-09, abs_tol=0.0)):
            print('Singular Matrix Error!')
            return False

        for i in range(j + 1, N, 1):
            v = A[i, j]
            for c in range(0, j, 1):
                v = v - L[i, c] * L[j, c] * d[c]
                L[i, j] = v / d[j]
                L[j, i] = v / d[j]

    for i in range(0, N, 1):
        y[i] = b[i]
        for j in range(0, i + 1, 1):
            y[i] = y[i] - L[i, j] * y[j]

    for i in range(N - 1, -1, -1):
        x_mat[i] = y[i] / d[i]
        for j in range(i, N, 1):
            x_mat[i] = x_mat[i] - L[i, j] * x_mat[j]

    return x_mat


def cholesky_reconstruct(frames, p, Nw, a_hat, times):
    dim = frames.shape[0]

    modulus = math.floor(dim / 10)
    percent = 0

    index = 0
    for t in times:
        """if ((index % modulus) == 0):
        print("--> %i%%" %(percent))
        percent = percent + 10"""
        print(index)

        if (t == []):
            index = index + 1
            continue

        else:
            ex = False

            temp_frame = frames[index]
            temp_a = a_hat[index]

            if (float('nan') in temp_a[:]):
                index = index + 1
                continue

            values = np.zeros(len(temp_frame), dtype=int)

            for tup in t:
                for x in range(tup[0], tup[1], 1):
                    values[x] = 1

            values[0:p] = 0
            values[Nw - p:Nw] = 0

            l = int(np.sum(values))

            if (l > 0):

                b = np.zeros(p + 1)
                B = np.zeros([l, l])
                d = np.zeros(l)
                t = np.zeros(l, dtype=int)

                temp_index = 0
                for x in range(len(values)):
                    if (values[x] == 1):
                        t[temp_index] = x
                        temp_index = temp_index + 1

                for i in range(0, p + 1, 1):
                    b[i] = 0.0
                    for j in range(i, p + 1, 1):
                        b[i] = b[i] + temp_a[j] * temp_a[j - i]

                for i in range(0, l, 1):
                    for j in range(i, l, 1):
                        if (abs(t[i] - t[j]) < p + 1):
                            B[i, j] = b[abs(t[i] - t[j])]
                            B[j, i] = b[abs(t[i] - t[j])]

                for i in range(0, l, 1):
                    d[i] = 0
                    for j in range(-p, p + 1, 1):
                        if ((t[i] - j) in t):
                            continue
                        else:
                            d[i] = d[i] - b[abs(j)] * temp_frame[t[i] - j]

                x_mat = np.zeros(l)

                #x_mat = cholesky(B, x_mat, d, l)

                #if (x_mat_new == False):
                #index = index + 1
                #continue

                L, D, perm = scipy.linalg.ldl(B, lower=True, hermitian=False, overwrite_a=True, check_finite=True)
                #x_mat = scipy.linalg.ldl(B, lower=True, hermitian=True, overwrite_a=False, check_finite=True)

                x_mat = np.linalg.solve(L, d)

                L_t = np.transpose(L)
                D_inv = np.linalg.inv(D)
                D_x = np.matmul(D_inv, x_mat)

                s_t = np.linalg.solve(L_t, D_x)

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
    rate, sampwidth, arr = readwav(sound_file)

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

        new_arr[:, ch] = channel[:]

    return rate, new_arr


def main():
    rate, data = remove_noise('vinyl-crackle_123bpm_B_minor.wav', 1.8, 20, 300, 2400, 1, 0.75)

    writewav24('restored_crackle.wav', rate, data)


main()