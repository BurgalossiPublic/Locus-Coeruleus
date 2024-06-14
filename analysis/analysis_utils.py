import pandas as pd
import numpy as np
from scipy.signal import find_peaks

#install dependencies
#!pip install openpyxl statsmodels statannotations pandas numpy matplotlib seaborn scikit-learn 

def load_json_restore_arrays(file_path):
    """
    Load a JSON file into a DataFrame and restore arrays from lists in specified columns.

    Parameters:
    - file_path: str, the path to the JSON file.

    Returns:
    - DataFrame with columns containing numpy arrays where applicable.
    
    ### FIX: UPDATE THIS FUNCTION TO USE NWB FILE INSTEAD OF JSON FILE
    """
    # Load the JSON file into a DataFrame
    data = pd.read_json(file_path)

    # Columns expected to contain numpy arrays
    array_columns = ['spikes', 'recording_dur', 'ISI', 'ISI_tv', 'ACG_narrow',
                     'ACG_narrow_tv', 'ACG_wide', 'ACG_wide_tv', 'FS_response',
                     'FS_tv', 'waveshape', 'waveshape_tv']

    # Convert lists in specified columns back to numpy arrays
    for col in array_columns:
        if col in data.columns:
            data[col] = data[col].apply(lambda x: np.array(x) if isinstance(x, list) else x)

    return data

def compute_isi(spkt, isi_binw=0.02, isi_max=1, standardize=True):
    """
    Retruns the interspike interval distribution.

    Parameters
    ----------
    spkt: np.array
        Spike times.
    isi_binw: float
        Bin width for the ISI distribution.
    isi_max: float
        Maximum ISI to be considered.
    standardize: bool
        If True, the ISI distribution will be standardized.

    Returns
    -------
    isi_distr: np.array
        ISI distribution.
    bin_centers: np.array
        Time vector.
    """
    isi = np.diff(spkt)
    bins = np.arange(0, isi_max + isi_binw, isi_binw)
    bin_centers = np.arange(isi_binw / 2, isi_max, isi_binw)
    isi_distr, isi_distr_bins = np.histogram(isi, bins=bins, density=True)
    if np.isnan(isi_distr).any():
        isi_distr = np.zeros(isi_distr.shape)
    if (standardize==True) & (sum(isi_distr)!=0):
        isi_distr = standardize_data(isi_distr)
    return isi_distr, bin_centers

def coefficient_of_variation(spkt):
    """
    Returns the coefficient of variation.

    Parameters
    ----------
    spkt: np.array
        Spike times.

    Returns
    -------
    cv2: float
        Coefficient of variation after Softky and Koch (1993)
    """
    isi = np.diff(spkt)
    isi_mean = np.mean(isi)
    isi_sd = np.std(isi)
    cv2 = np.round(isi_sd / isi_mean, 4)
    return cv2

def compute_acg(spkt, binsize, maxlag, stdz_per_neuron):
    """
    Calculate the autocorrelogram.

    Parameters
    ----------
    spkt: list or np.array
        Spike times.
    binsize: float
        Bin size for the autocorrelogram.
    maxlag: float
        Maximal lag.
    stdz_per_neuron: bool
        If True, the autocorrelogram will be standardized.

    Returns
    -------
    acg: np.array
        Computed autocorrelogram.
    bins: np.array
        Bin times relative to center
    """
    if type(spkt) == list:
        spkt = np.array(spkt)
    # Create time lag vector
    bins = np.arange(-maxlag, maxlag + binsize, binsize)
    if sum(spkt) != 0:
        nbins = int(np.round(maxlag / binsize))
        acg = np.zeros(2 * nbins + 1, )
        N = spkt.size
        j = 0
        # Loop over spike time
        for i in range(N):
            # Loop over reference spike time
            while j > 0 and spkt[j] >= spkt[i] - maxlag:
                j -= 1
            while j < N - 1 and spkt[j + 1] <= spkt[i] + maxlag:
                j += 1
                # Ignore the zero time lag
                if i != j:
                    off = (np.round((spkt[i] - spkt[j]) / binsize))
                    acg[int(nbins + off)] += 1
        # Correct for rounding artifact in first and last bin
        acg[0] = acg[0] * 2
        acg[-1] = acg[-1] * 2
    else:
        print('Empty spike times array.')
        acg = np.zeros((bins.shape))

    # standardize
    if stdz_per_neuron:
        if sum(acg) == 0:
            acg = np.zeros(acg.shape)
        elif stdz_per_neuron:
            acg = standardize_data(acg)

    return (acg, bins)

def get_waveshape_peak_to_trough(waveshape, waveshape_tv):
    """ Get peak to trough time in seconds

    Parameter
    ---------
    waveshape: np.array
        Waveshape recording.
    waveshape_tv: np.array
        Time vector for waveshape recording.

    Return
    ------
    peakToTrough: float
        Peak to trough time in seconds.

    """
    # Use max of waveshape as peak
    peak = waveshape.max()
    idx_peak = np.where(waveshape == peak)[0][0]
    time_peak = waveshape_tv[idx_peak]

    # Catch artifact of detecting peak at the end of waveshape recording
    if waveshape_tv[np.where(waveshape == waveshape[idx_peak:].min())[0][0]] > 0.0075:
        idxs_troughs, _ = find_peaks(waveshape[idx_peak:] * -1, distance=15)
        idx_trough = idxs_troughs[0] + idx_peak
    else:
        trough = waveshape[idx_peak:].min()
        idx_trough = np.where(waveshape == trough)[0][0]
    time_trough = waveshape_tv[idx_trough]

    peakToTrough = time_trough - time_peak
    peakToTrough = peakToTrough * 1000  # Convert to ms
    peakToTrough = np.round(peakToTrough, 5) # Round to 5 decimal places

    return peakToTrough

def get_waveshape_FWHM(signal, time_vector):
    """
    Calculate the Full Width at Half Maximum (FWHM) of a given signal.

    Parameters:
    - signal: numpy array, the signal waveform.
    - time_vector: numpy array, the time vector corresponding to the signal.

    Returns:
    - fwhm: float, the full width at half maximum of the signal. Returns None if FWHM cannot be calculated.
    """
    max_val = np.max(signal) # Find max amplidute of signal
    half_max = max_val / 2 # Calculate half max amplidute
    difference = signal - half_max # Subtract half max from signal to find crossings
    sign_difference = np.sign(difference) # Make an array of sign values
    diff_sign = np.diff(sign_difference) # Difference between consecutive elements in the array of sign values
    crossings = np.where(diff_sign)[0] # Find the indices of the crossings

    if len(crossings) >= 2:
        fwhm = time_vector[crossings[-1]] - time_vector[crossings[0]]
        fwhm = fwhm * 1000  # Convert to ms
        fwhm = np.round(fwhm, 3) # Round to 3 decimals
    else:
        # Unable to calculate FWHM if there are less than 2 crossing points
        fwhm = None
        
    return fwhm

def normalize_data(data, range=[0, 1]):
    """
    Normalize data to [0,1] range.
    Parameters
    ----------
    data: np.array
        Data to be normalized.

    Returns
    -------
    data_norm: np.array
        Normalized data.

    """
    a = range[0]
    b = range[1]
    data_norm = (b - (a)) * ((data - np.min(data)) / (np.max(data) - np.min(data))) + (a)
    return data_norm

def standardize_data(data):
    """
    Standardize data to zero mean and unit variance.
    Parameters
    ----------
    data: np.array
        Data to be standardized.

    Returns
    -------
    data_std: np.array
        Standardized data.

    """
    data_std = (data - np.mean(data)) / np.std(data)
    return data_std

def normalize_waveshape(waveshape):
    """
    Normalize the waveshape array by adjusting its values based on the first element and then scale start-to-peak to the range [0, 1].

    Parameters:
    - waveshape: numpy array, the waveshape array to be normalized.

    Returns:
    - numpy array, the normalized waveshape.
    """
    # Subtract DC offset
    waveshape_adjusted = waveshape - waveshape[0]
    
    # Find max after adjusting
    max_val = np.max(waveshape_adjusted)
    
    # Normalize waveshape
    waveshape_normalized = (waveshape_adjusted - waveshape_adjusted[0]) / (max_val - waveshape_adjusted[0])
    return waveshape_normalized

def perform_sparse_pca(X, tv, sparsity):
    """ Perform sparse PCA on the input data.

    Parameters
    ----------
    X: np.array
        Input data.
    tv: np.array
        Time vector.
    sparsity: float
        Sparsity parameter for SparsePCA.

    Returns
    -------
    sparse_pca_components: np.array
        Sparse PCA components.
    X_pca: np.array
        PCA transformed data.
    """
    from sklearn.decomposition import PCA, SparsePCA
    # Standardize data before PCA
    X_stdz = np.zeros((X.shape[0], X.shape[1]))
    for i in range(X.shape[0]):
        if sum(X[i, :]) == 0:
            X_stdz[i, :] = np.zeros([X.shape[1], ])
        else:
            X_stdz[i, :] = standardize_data(X[i, :])
    X = X_stdz

    # PCA
    sparse_pca = SparsePCA(n_components=4, alpha=sparsity)
    sparse_pca.fit(X)
    X_pca = sparse_pca.transform(X)

    # Sort components in temporal order
    sparse_pca_components = sparse_pca.components_
    # restrict to positive time
    pcs_time_max = tv[np.where(tv >= 0)[0]][
        np.argmax(abs(sparse_pca_components[:, np.where(tv >= 0)[0]]), axis=1)]
    pcs_order = np.argsort(pcs_time_max)
    sparse_pca_components_sorted = []
    X_pca_sorted = []
    for i in range(4):
        sparse_pca_components_sorted.append(sparse_pca_components[pcs_order[i], :])
        X_pca_sorted.append(X_pca[:, pcs_order == i])
    sparse_pca_components = np.vstack(sparse_pca_components_sorted)
    X_pca = np.hstack(X_pca_sorted)
    return sparse_pca_components, X_pca

def get_spikes_in_intertrial_interval(FS_spkt, trange):
    spikes_in_intertrial_interval = np.zeros((len(FS_spkt)))
    for i in np.arange(len(FS_spkt)):
        spikes_in_intertrial_interval[i] = len(FS_spkt[i])
    mean_spikes_in_intertrial_interval = np.mean(spikes_in_intertrial_interval)
    iti_duration = trange[1] - trange[0]
    spikes_per_sec = mean_spikes_in_intertrial_interval / iti_duration
    return spikes_per_sec

def get_single_cell_fsmi(psth, psth_t, stim_tranges=[0, 0.5], ctype='modi', fr_crit=1):
    """ Get foot-shock modulation index (FSMI) for a single cell.

    Parateters
    ----------
    psth: np.array
        The FS response as PSTH.
        Note: to get a meaningful 'modi' the PSTH should be non-normalized/non-standardized
        and not set to baseline (i.e. use the raw trace).
    psth_t: np.array
        The time vector that corresponds to the PSTH.
    stim_tranges: list
        List containing [start, stop] of the stimulus in sec.
    ctype: str
        Can be either 'modi' or 'FRdiff' for modulation index or simple FR difference,
        respectively.
    fr_crit: int
        Critical firing rate in Hz for the unit to be included. If firing rate < fr_crit NaN is
        returned.

    Return
    ------
    fs_modi: float
        The foot-shock modulation index.
    """
    if (np.isnan(psth).any()) or (np.mean(psth) < fr_crit):
        fs_modi = np.nan
    else:
        # Get FR of the baseline
        ctrl_idxs = np.where(psth_t < 0)
        ctrl_psth = psth[ctrl_idxs]
        ctrl_FR = np.mean(ctrl_psth)

        # Get FR during stimulus
        stim_idxs = np.where((psth_t >= stim_tranges[0]) & (psth_t < stim_tranges[1]))
        stim_psth = psth[stim_idxs]
        stim_FR = np.mean(stim_psth)

        # Calculate the FS modulation index
        if ctype == 'modi':
            fs_modi = (stim_FR - ctrl_FR) / (stim_FR + ctrl_FR)
        elif ctype == 'FRdiff':
            fs_modi = stim_FR - ctrl_FR

    return fs_modi

def get_fs_response(spkt_fs, fs_onsets, binw=0.03, trange=[-1, 2]):
    """ Returns PSTH triggered on FS onsets.

     Paramters
    ---------
    spkt_fs: np.array
        Spike times in response to FS stimulus.
    fs_onsets: list
        FS stimulus onsets.
    binw: float
        Bin width for the PSTH.
    trange: list
        List contains the time range around the FS response that should be shown.
        Default is 1 s before and 2 s after stimulus onset in seconds.
        trange[0] should be negative number to show time before stimulus onset.
    ev_trange: list
        List contains the event/stimulus time ranges [stimulus onset, stimulus offset]
        in seconds.

    Return
    ------
    psth: np.array
        FS response curve.
    tv: np.array
        Time vector.

    """

    # Loop through trials and get spkt
    FS_spkt = []
    t_prev_fs = trange[0]
    t_aft_fs = trange[1]
    for i, fs_onset in enumerate(fs_onsets):
        # Save spike times for the respective trial
        t_start = fs_onset + t_prev_fs
        t_stop = fs_onset + t_aft_fs
        spkt_fs_raw = spkt_fs[((spkt_fs >= t_start) & (spkt_fs <= t_stop))]
        FS_spkt.append(spkt_fs_raw - fs_onset)
    bin_edges = np.arange(trange[0], trange[1] + binw, binw)
    psth, bins = np.histogram(np.sort(np.hstack(FS_spkt)), bins=bin_edges)
    bincenters = bins[:-1] + binw / 2

    return bincenters, psth

def get_fs_spkt(df, idx, trange=[-1, 2], split_by_trial=False):
    """ Get spike times for a single cell w.r.t. stimulus onset.

     Paramters
    ---------
    df: pd.DataFrame
        Pandas data frame with LHb data.
    cell_id: str or int
        Can be either the 'cellID' (str) or the index in Data Frame (int).
    split_by_trial: bool
        If true, spike times for each trial are returned seperately.

    Return
    -------
    FS_spkt: list
        Spike times in response to stimulus onset at 0 s.
    """
    cellID = df.at[idx, 'Cell_Num']
    
    # Catch cells for which no FS response was recorded
    if (np.isnan(df.at[idx, 'FS_response'])).any():
        print('No FS response for cell {:s}.\nReturning None.'.format(cellID))
        FS_spkt = None
    else:
        # Get spike times and stimulus onsets
        spkt = np.array(df.at[idx, 'FS_spikes'])
        fs_onsets = df.at[idx, 'FS_onset']
        # Loop through trials and get spkt
        FS_spkt = []
        t_prev_fs = trange[0]
        t_aft_fs = trange[1]
        for i, fs_onset in enumerate(fs_onsets):
            # Save spike times for the respective trial
            t_start = fs_onset + t_prev_fs
            t_stop = fs_onset + t_aft_fs
            spkt_fs_raw = spkt[((spkt >= t_start) & (spkt <= t_stop))]
            FS_spkt.append(spkt_fs_raw - fs_onset)
        if split_by_trial is False:
            FS_spkt = list(np.sort(np.hstack(FS_spkt)))

    return FS_spkt

def get_fs_psth_mat(df, exclude_nans=False, norm_per_cell=False, standardize_per_cell=False,
                    set_baseline=False, verbal=True, binw=0.03, trange=[-1, 2]):
    """ Get foot-shock (FS) response functions for all cells as matrix.
    Dimensions of the returned matrix are (n_cells x psth_times).

    Parameters
    ----------
    df : pd.DataFrame
        Data frame must have 'PSTH_time' and 'FS_response' as columns.
    exclude_nans : bool
        If True it will only return PSTHs of cell for which FS-response was recorded.
        If False it will return arrays of NaNs for cells that do not have FS-response recorded.
    norm_per_cell : bool
        If True normalize each cells PSTH between 0 and 1.
    set_baseline : bool
        If True it will set the baseline to zero.
    standardize_per_cell : bool
        If true PSTH will be standardized to have mu=0 and sigma=1.
    verbal : bool
        If True it will print how many cells and how many cells w/o FS-response.

    Returns
    -------
    psth_mat : np.array
        Contains FS-responses for cells as rows.
    cell_ids : list
        List of int containing the corresponding cell indices.
        Should have the same dimensions as psth_mat.shape[0]
    psth_t : np.array
        Time vector for PSTHs.
    """
    assert ~((standardize_per_cell is True) & (norm_per_cell is True)), (
        "Only norm_per_cell or standardize_per_cell can be True.")

    psth_temp = []
    cell_ids = []
    counter_nan = 0
    bin_edges = np.arange(trange[0], trange[1] + binw, binw)

    for cell_idx in df.index:
        spkt = np.array(df.at[cell_idx, 'FS_spikes'])

        # If no FS recording
        if (np.isnan(spkt) == True).any():
            counter_nan += 1
            if not exclude_nans:
                psth_temp.append(np.full((bin_edges.shape[0] - 1,), np.nan))
                cell_ids.append(cell_idx)
        else:
            fs_onsets = df.at[cell_idx, 'FS_onset']
            FS_spkt = []
            for fs_onset in fs_onsets:
                t_start = fs_onset + trange[0] # Start of the PSTH
                t_stop = fs_onset + trange[1] # End of the PSTH
                spkt_fs_raw = spkt[((spkt >= t_start) & (spkt <= t_stop))] # Spike times in PSTH
                FS_spkt.append(spkt_fs_raw - fs_onset) # Spike times relative to FS onset

            psth, bins = np.histogram(np.sort(np.hstack(FS_spkt)), bins=bin_edges) # PSTH
            psth = (psth/binw).astype('float64') # Hz
            psth_t = bins[:-1] + binw / 2 # bincenters

            # Normalize, standardize, set baseline
            if norm_per_cell:
                psth = (psth - np.min(psth)) / (np.max(psth) - np.min(psth))
            if standardize_per_cell:
                stim_onset = len([1 for i in psth_t if i < 0])
                psth -= np.mean(psth[:stim_onset - 1])
                psth /= np.std(psth[:stim_onset - 1])
            if set_baseline:
                baseline_psth = psth[np.where(psth_t <= 0)]
                baseline_mean = np.mean(baseline_psth)
                psth -= baseline_mean
            psth_temp.append(psth)
            cell_ids.append(cell_idx)

    psth_mat = np.vstack(psth_temp)
    
    if verbal:
        print('{:d}/{:d} cells without FS-response.'.format(counter_nan, df.shape[0]))

    return psth_mat, cell_ids, psth_t


def calculate_latencies(row):
    
    response_window = 0.1  # Set response window here

    FS_spikes = row['FS_spikes']
    FS_onset = row['FS_onset']
    
    # Calculate the indices of the immediately bigger FS_spike value for each FS_onset value
    indices = np.searchsorted(FS_spikes, FS_onset, side='right')
    
    # Initialize latencies array
    latencies = np.empty(len(FS_onset))
    latencies[:] = np.nan
    
    # Only calculate latencies where the index is within bounds
    valid_indices = indices < len(FS_spikes)
    latencies[valid_indices] = [FS_spikes[i] - FS_onset[j] for i, j in zip(indices[valid_indices], np.where(valid_indices)[0])]
    
    # Set latencies larger than response_window to NaN
    latencies[latencies > response_window] = np.nan

    latencies = latencies * 1000 # Convert to ms

    return latencies