from multiprocessing.sharedctypes import RawArray
import yaml
from typing import List, Tuple
import pandas as pd
import numpy as np
import os
import scipy.io as sio
import mne
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import TwoSlopeNorm
from scipy.spatial.distance import squareform, pdist
from copy import deepcopy
from scipy.signal import butter, filtfilt
import matplotlib
import matplotlib.pyplot as plt
import category_encoders as ce
import nltk
from nltk.probability import FreqDist, ConditionalFreqDist
import string
plt.rcParams['figure.figsize'] = [20, 8]


def load_config(config_path: str) -> dict:
    """Method for loading a configuration file

    :param config_path: path of configuration file
    :return: dictionary of configuration file
    """
    with open(config_path) as file:
        config=yaml.safe_load(file)
    return config


def save_config(config: dict, config_path: str) -> None:
    """Method for saving a configuration file
    
    :param config: dictionary of configuration file
    :param config_path: path of configuration file
    """
    with open(config_path, 'w') as file:
        yaml.safe_dump(config, file)


def get_frequency_bands() -> dict:
    """Returns frequency bands
    
    :return: dictionary with frequency bands
    """
    return {
        'theta': (4, 8),
        'low alpha': (8, 10),
        'high alpha': (10, 12),
        'alpha': (8, 12),
        'SMR': (12, 15),
        'low beta': (15, 23),
        'high beta': (23, 30),
        'beta': (15, 30)
    }


def get_timings() -> dict:
    """Returns timing in seconds of particular recording related to the name of the task
    
    :return: dictionary with timings in seconds
    """
    return {
        'Squats': 27,
        'Successful_Competition': 38,
        'Fitness_Activity': 37,
        'Slow_Start': 47,
        'Start_high_level_championship': 33,
        'Training_Session': 37,
        'Your_Home_Venue': 40
    }


def get_coordinates(config: dict, channels: List[str]) -> pd.DataFrame:
    """Return coordinates (x,y,z) for each channel with g.GAMMAcap2 schema

    :return: Data Frame with coordinates for each channel
    """
    coords = pd.read_csv(config.get('environment').get('electrodes_coordinates'))
    lower_channels = [ch.lower() for ch in channels]
    filtered_coords = coords[coords['Electrode name'].str.lower().isin(lower_channels)].reset_index(drop=True)
    lower_channels_indexes = [filtered_coords[filtered_coords['Electrode name'].str.lower()==l_ch].index[0] for l_ch in lower_channels]

    for i, name in zip(lower_channels_indexes, channels):
        filtered_coords.loc[i, 'Electrode name']=name 

    filtered_coords.columns = ['electrode_name', 'x', 'y', 'z']
    for coord_col in ['x', 'y', 'z']:
        filtered_coords[coord_col] = filtered_coords[coord_col].apply(lambda x: float(x.replace(',','.')))

    return filtered_coords


def get_distances(df_coordinates: pd.DataFrame) -> pd.DataFrame:
    df_coordinates_copy = df_coordinates.copy()
    df_coordinates_copy.set_index('electrode_name', inplace=True)
    df_distances = pd.DataFrame(squareform(pdist(df_coordinates_copy)),
                                columns = df_coordinates_copy.index,
                                index= df_coordinates_copy.index)
    return df_distances


def get_neighbours(df_distances: pd.DataFrame, channel_name: str, lower_distance=0, upper_distance=0.7) -> List[str]:
    electrode_neighbours = df_distances[(df_distances>lower_distance)&(df_distances<upper_distance)][channel_name].dropna().index.values
    return electrode_neighbours


def load_eeg_data(config: dict, participant_id: int = None) -> Tuple[List[str], pd.DataFrame]:
    """Loading raw eeg signal data for single or all participants

    :return: The tuple consisted of list of electrodes' channels and data frame with signals
    """
    
    # reading channels
    channels = pd.read_csv(config.get('environment').get('electrodes_legend'), header=None).iloc[:,1].tolist()
    # reading paths of signals from given in config path 
    signal_paths = os.listdir(config.get('environment').get('eeg_raw_signals'))
    # creating dataframe consisted of paths for each participant id
    id_list = [int(path.split('_')[0][4:]) for path in signal_paths]
    df_data_raw = pd.DataFrame(list(zip(id_list, signal_paths)),
            columns=['participant_id', 'signal_path'])

    # loading raw signals from files for individual or all
    if not participant_id is None:
        df_data_raw = df_data_raw.loc[df_data_raw['participant_id']==participant_id]

    df_data_raw['raw_signal'] = [sio.loadmat(os.path.join(config.get('environment').get('eeg_raw_signals'),
                path))['eeg'] for path in df_data_raw['signal_path'].values]

    return channels, df_data_raw


def prepare_mne_array(config: dict, channels: List[str], raw_eeg_signal: List[List[float]], annotation_mask, ordered_annotation_mask) -> mne.io.RawArray:
    """Preparing men.io.RawArray instance from raw eeg signal data

    :return: The eeg signal as mne.io.RawArray
    """
    # creates a basic Info instance in package mne
    info = mne.create_info(ch_names=channels,
                            ch_types=['eeg']*len(channels),
                            sfreq=config.get('experiment').get('frequency'))
    # creating RawArray                            
    simulated_raw = EegArray(eeg_signal=raw_eeg_signal, info=info, annotation_mask=annotation_mask, ordered_annotation_mask=ordered_annotation_mask)

    return simulated_raw


def load_tasks(config: dict) -> pd.DataFrame:
    """Loading ordered tasks for each participant

    :return: Data Frame with tasks in order for each participant
    """
    tasks_mapping = {'Przysiady': 'Squats',
                'WygraneZawody': 'Successful_Competition',
                'AktywnoscRuchowa': 'Fitness_Activity',
                'KiepskiPoczatek': 'Slow_Start',
                'StartZawody': 'Start_high_level_championship',
                'Trening': 'Training_Session',
                'MiejsceTreningu': 'Your_Home_Venue'}
    
    df_tasks = pd.read_csv(config.get('environment').get('sports_order'))
    df_tasks.columns = ['task', 'participant_id', 'task_order']
    # removing file extension
    df_tasks.task = df_tasks.task.apply(lambda x: x.split('.')[0])
    # mapping polish names to english
    df_tasks.task = df_tasks.task.apply(lambda x: tasks_mapping[x] if x in list(tasks_mapping.keys()) else x)
    # represent participant_id as int
    df_tasks.participant_id = df_tasks.participant_id.apply(lambda x: int(x[1:]))

    return df_tasks


def load_markers(config: dict) -> pd.DataFrame:
    """Loading time markers of tasks for each participant

    :return: Data Frame with time markers in order for each participant
    """
    marker_paths = os.listdir(config.get('environment').get('eeg_markers'))
    # creating dataframe consisted of paths for each participant id
    id_list = [int(path.split('_')[0][4:]) for path in marker_paths]
    df_marker_raw = pd.DataFrame(list(zip(id_list, marker_paths)),
                                columns=['participant_id', 'marker_path'])

    # loading time markers for each participant and alligning them in order to participant_id
    df_marker_ordered = pd.DataFrame({
                        row['participant_id']:
                        np.sort(sio.loadmat(
                        os.path.join(config.get('environment').get('eeg_markers'),row['marker_path'])
                        )['marker'].reshape(-1).astype(int))
                        for _, row in df_marker_raw.iterrows()}
                    ).stack().reset_index()
    # columns naming
    df_marker_ordered.columns = ['task_order', 'participant_id', 'time_marker']
    df_marker_ordered.task_order = df_marker_ordered.task_order+1

    return df_marker_ordered


def extend_markers(config: dict, df_marker: pd.DataFrame) -> pd.DataFrame:
    """Extending data frame with a recording and imagine markers and duration of each part
    
    :return: Extended data frame with division of markers per recording and imagine parts
    """
    bias=config.get('experiment').get('bias')*config.get('experiment').get('frequency')
    timings = get_timings()

    df_marker['auditory_marker'] = df_marker['time_marker'].copy()
    df_marker['auditory'] = df_marker.task.apply(lambda x: timings.get(x))
    df_marker['imagine_marker'] = df_marker['auditory_marker']+(df_marker['auditory']*config.get('experiment').get('frequency'))+bias
    df_marker['imagine'] = config.get('experiment').get('imagine')
    if not config.get('experiment').get('imagine_long') is None:
        df_marker['imagine_long_marker'] = df_marker['imagine_marker'].copy()
        df_marker['imagine_long'] = config.get('experiment').get('imagine_long')

    return df_marker


def prepare_annotations(config: dict, df_marker_task: pd.DataFrame) -> pd.DataFrame:
    """Prepare data frame for annotations

    :return: Annotations data frame
    """
    annotation_columns = [column for column in df_marker_task.columns
                        if column.split('_')[0] in ['auditory', 'imagine']]
    marker_columns = [column for column in annotation_columns if '_marker' in column]
    duration_columns = list(set(annotation_columns).difference(marker_columns))

    df_annotation_marker = df_marker_task.set_index(['task', 'participant_id'])[marker_columns].stack().reset_index()
    df_annotation_duration = df_marker_task.set_index(['task', 'participant_id'])[duration_columns].stack().reset_index()

    df_annotation_marker.columns = ['task', 'participant_id', 'phase', 'timestamp']
    df_annotation_duration.columns = ['task', 'participant_id', 'phase', 'duration']

    df_annotation_marker['timestamp'] = df_annotation_marker['timestamp']/config.get('experiment').get('frequency')
    df_annotation_marker['phase'] = df_annotation_marker['phase'].apply(lambda x: '_'.join(x.split('_')[:-1]))
    df_annotation = df_annotation_marker.merge(df_annotation_duration, on=['task', 'participant_id', 'phase'])
    df_annotation['description'] = df_annotation.apply(lambda x: '_'.join([x['task'], x['phase']]), axis=1)

    return df_annotation


def plot_channel(raw_signal: mne.io.array.array.RawArray, annotations: mne.annotations.Annotations, channel: str, annotate: bool = True, use_mask: bool = True) -> None:
    """Plot channel for signal with annotations

    """
    single_signal = raw_signal[channel]

    fig, ax = plt.subplots(figsize=(20,4))
    try:
        single_signal_y = single_signal[0][0].copy()
        single_signal_x = single_signal[1]
    except:
        single_signal_y = single_signal
        single_signal_x = np.arange(len(single_signal))
        
    if use_mask:
        annotation_mask = np.array(raw_signal.ordered_annotation_mask, dtype=bool)
        reverse_mask = [not elem for elem in annotation_mask]
        single_signal_y[reverse_mask] = 0
        y_min = np.min(single_signal_y[annotation_mask])*1.3
        y_max = np.max(single_signal_y[annotation_mask])*1.3
    else:
        y_min = np.quantile(single_signal_y, 0.05)
        y_max = np.quantile(single_signal_y, 0.95)
    ax.plot(single_signal_x, single_signal_y, '-k', linewidth=1)
    ax.set_xlabel("time [s]", fontsize=20)
    ax.set_ylabel("potential [$\mu$V]", fontsize=20)
    fig.tight_layout()
    plt.title(channel, fontsize = 30)
    ax.set_ylim((y_min, y_max))

    if annotate:
        current_annotation = ""
        for annotation in annotations:
            annotation = dict(annotation)
            mark_area = single_signal[1][np.where(np.logical_and(single_signal[1]>=annotation['onset'], single_signal[1]<=annotation['onset']+annotation['duration']))]
            if current_annotation != "_".join(annotation['description'].split('_')[:-1]):
                current_annotation = "_".join(annotation['description'].split('_')[:-1])
                label_text_beginning = single_signal[1][np.where(single_signal[1]>annotation['onset'])][0]
                ax.text(label_text_beginning, y_max-((y_max-y_min)/10), current_annotation,
                            bbox={'facecolor': 'pink', 'alpha': 1.0, 'pad': 10})
            if annotation['description'].split('_')[-1]=='auditory':
                color = 'orange'
            elif annotation['description'].split('_')[-1]=='imagine':
                color = 'green'
            elif annotation['description'].split('_')[-1]=='imagine_long':
                color = 'blue'
            else:
                color = 'red'
            ax.fill_between(mark_area, y_max, y_min, color=color, alpha=0.3, interpolate=True, label=current_annotation)


def fig2data (fig: matplotlib.figure.Figure) -> np.array:
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf


def prepare_weights(config: dict) -> np.array:
    size = config.get('constants').get('weights_size',100)
    space = np.linspace(0,1,int(size/2)+2)
    space = np.log(space+1)*3
    half_w = []
    for i in range(int(size/2))[::-1]:
        half_w.append(np.concatenate([
        np.zeros(i),
        space[:-(i+1)],
        space[:-(i+2)][::-1],
        np.zeros(i)
        ]))
    weights = np.concatenate([
                    half_w,
                    [np.concatenate([space[1:], space[1:][::-1][1:]])], 
                    half_w[::-1]
                ])

    return weights


def create_moment_topography(config: dict, moment_signal: pd.Series, df_coordinates: pd.DataFrame) -> np.array:
    weights = prepare_weights(config=config)
    shape = config.get('constants').get('mesh_shape')
    scale = config.get('constants').get('mesh_scale')
    mesh = np.zeros((shape,shape))

    coords = df_coordinates[['x','y']]
    y = (((coords['x']*100)+(100*scale))/(2*(100*scale)/shape)).astype(int)
    x = (((coords['y']*100)+(100*scale))/(2*(100*scale)/shape)).astype(int)
    row_indexes = x - int(weights.shape[0]/2)
    column_indexes = y - int(weights.shape[0]/2)

    electrodes_weights = {}
    for electrode in df_coordinates['electrode_name']:
        electrodes_weights.update({electrode: weights*moment_signal[electrode]})

    for electrode, row_idx, column_idx in zip(df_coordinates['electrode_name'], row_indexes, column_indexes):
        electrode_weights = electrodes_weights[electrode]
        mesh_crop = mesh[row_idx:row_idx+electrode_weights.shape[0], column_idx:column_idx+electrode_weights.shape[0]]
        mesh[row_idx:row_idx+electrode_weights.shape[0], column_idx:column_idx+electrode_weights.shape[0]] = np.where(
            mesh_crop!=0,
            np.where(electrode_weights!=0, (mesh_crop+electrode_weights), mesh_crop+electrode_weights),
            mesh_crop+electrode_weights)

    return mesh


def plot_topography(config: dict, mesh: np.array, head_figure: matplotlib.figure.Figure, return_figure: bool = False):
    head_im = fig2data(head_figure)
    shape = config.get('constants').get('mesh_shape')
    scale = config.get('constants').get('mesh_scale')
    x, y = np.meshgrid(np.linspace(-scale,scale,shape), np.linspace(-scale,scale,shape))

    fig, ax = plt.subplots()
    ax.imshow(head_im, extent=[x.min(), x.max(), y.min(), y.max()])
    max_ = np.max([np.abs(mesh.min()), mesh.max()])
    cmesh = ax.pcolormesh(x, y, mesh, cmap='RdBu', vmin=-1*max_, vmax=max_, alpha=0.9) #norm=TwoSlopeNorm(vmin=mesh.min(), vmax=mesh.max(), vcenter=0)
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(cmesh, ax=ax)
    
    if return_figure:
        return fig


def distance_laplace(raw, df_distances):
    raw_laplace = deepcopy(raw)
    for channel in raw.ch_names:
        neighbours = get_neighbours(df_distances, channel)
        sum_neighbours = np.zeros(len(raw[channel][0][0]))
        for neighbour in neighbours:
            sum_neighbours+=raw[neighbour][0][0]
        new_raw = raw[channel][0][0] - ((1/len(neighbours))*sum_neighbours)
        raw_laplace[channel][0][0] = new_raw
    return raw_laplace


def normalize_raw(raw, annotation_mask):
    raw_normalize = deepcopy(raw)
    for channel in raw.ch_names:
        new_raw = raw[channel][0][0].copy()
        short_raw = raw[channel][0][0][annotation_mask]
        short_raw = (short_raw - np.mean(short_raw)) / np.std(short_raw)    
        new_raw[annotation_mask] = short_raw
        raw_normalize[channel][0][0] = new_raw
    return raw_normalize


def butter_filter(raw, annotation_mask, minimal_frequency, maximal_frequency, sampling_frequency, order):
    raw_filter = deepcopy(raw)
    b, a = butter(N=order, Wn=[minimal_frequency, maximal_frequency], fs=sampling_frequency, btype='band')

    for channel in raw.ch_names:
        new_raw = raw[channel][0][0].copy()
        short_raw = raw[channel][0][0][annotation_mask]
        short_raw = filtfilt(b, a, short_raw)
        new_raw[annotation_mask] = short_raw
        raw_filter[channel][0][0] = new_raw

    return raw_filter
    

def iqr_filter(raw, annotation_mask, k):
    raw_iqr = deepcopy(raw)
    for channel in raw.ch_names:
        new_raw = raw[channel][0][0].copy()
        short_raw = raw[channel][0][0][annotation_mask]
        q3, q1 = np.percentile(short_raw, [75 ,25])
        iqr = q3 - q1
        lower_bound = q1 - k * iqr
        upper_bound = q3 + k * iqr
        upper_raw = np.where(short_raw > upper_bound, np.mean(short_raw), short_raw)
        lower_raw = np.where(upper_raw < lower_bound, np.mean(short_raw), upper_raw)
        new_raw[annotation_mask] = lower_raw
        raw_iqr[channel][0][0] = new_raw
    return raw_iqr


def plot_labels(df, limit=200):
    df['value'] = 1
    colors = {0:'r', 1:'g', 2:'b', 3:'yellow'}
    colored_label = df['label'].map(lambda x: colors[x])
    df[:limit].reset_index().plot('index', 'value', color=colored_label, kind='bar', width=1, figsize=(40,5))
    plt.show()


def compare_labels(df1, df2):
    colors = {0:'orange', 1:'g', 2:'b', 3:'purple'}
    colored_label = df1['label'].map(lambda x: colors[x])
    colored_label2 = df2['label'].map(lambda x: colors[x])
    fig, ax = plt.subplots(nrows=2, figsize=(20, 6), gridspec_kw={'height_ratios': [5, 1]})
    difference = np.abs(df1.label-df2.label)
    ax[0].set_ylim((-4,4))
    ax[0].bar(df1.index, df1.label+1, width=1, color=colored_label)
    ax[0].bar(df2.index, -1*(df2.label+1), width=1, color=colored_label2)
    ax[0].set_title(f"Adnotacja: {df1.annotation.unique()[0]}", fontsize=16)
    for idx in difference[difference!=0].index:
        ax[0].axvline(x=idx, color='red', linewidth=2.4, linestyle='--')
    fig.tight_layout()
    ax[1].bar(df1.index, difference, width=1)
    ax[1].set_ylim((0,1))
    ax[1].set_title(f"Procent podobieństwa: {np.round(len(difference[difference==0])*100/len(difference), 2)}%", fontsize=16)
    plt.show()


def zipf(text, new_figure=False, log=True, freq_rank=True, label=None):
    if isinstance(text, nltk.probability.FreqDist):
        fdis = text
    else:
        fdis = dict(FreqDist(text))
    freq = [item[1] for item in sorted(fdis.items(), key=lambda kv: kv[1], reverse=True)]
    rank = [item+1 for item in range(len(sorted(fdis.items(), key=lambda kv: kv[1], reverse=True)))]
    
    if new_figure:
        plt.figure(figsize=(20,8))

    if freq_rank:    
        if label:
            plt.plot(rank, [freq*rank for freq, rank in zip(freq, rank)], label=label)
        else:
            plt.plot(rank, [freq*rank for freq, rank in zip(freq, rank)])
        plt.ylabel('Frequency * Rank')
        plt.title('Logorithmic Frequency * Rank vs Rank for Words in a Text')
    else:
        if label:
            plt.plot(rank, freq, label=label)
        else:
            plt.plot(rank, freq)
        plt.ylabel('Frequency')
        plt.title('Logorithmic Frequency vs Rank for Words in a Text')
    
    # change plot to log scale to visually confirm Zipf's Law
    if log:
        plt.xscale("log")
        plt.yscale("log")
    
    # add axis labels, title, and legend
    plt.xlabel('Rank')
    plt.legend(loc='upper right')


def encode_letters(letters):
    def map_letter(letter, dict_map):
        return dict_map[letter]

    encoder = ce.BinaryEncoder(cols=['letters'],return_df=True)
    unique_letters = list(set(letters))
    df_unique_letters = pd.DataFrame(unique_letters, columns=['letters'])
    data_map_letter = encoder.fit_transform(df_unique_letters) 
    data_map_letter['letter'] = df_unique_letters['letters']
    data_map_letter = data_map_letter.set_index('letter')
    dict_map = {letter: row.values  for letter, row in data_map_letter.iterrows()}

    mapped_letters = list(map(lambda x: map_letter(x, dict_map), letters))
    encoded_letters = [x for array in mapped_letters for x in array]
    
    return encoded_letters


def plot_ngrams_ax(n_grams, ax, title_matter="", log=True):
    for label, fdis in sorted(n_grams.items()):

        freq = [item[1] for item in sorted(fdis.items(), key=lambda kv: kv[1], reverse=True)]
        rank = [item+1 for item in range(len(sorted(fdis.items(), key=lambda kv: kv[1], reverse=True)))]

        ax.plot(rank, [freq*rank for freq, rank in zip(freq, rank)], label=label)
        ax.set_ylabel('Frequency * Rank')
        ax.set_title(f'Logorithmic Frequency * Rank vs Rank for {title_matter}')

        # change plot to log scale to visually confirm Zipf's Law
        if log:
            ax.set_xscale("log")
            ax.set_yscale("log")

        ax.set_xlabel('Rank')
        ax.legend(loc='upper right')
    return ax


def ngrams_freq_dist(n, list):
    dict_ngrams = {}
    for i in range(1, n):
        n_gram = nltk.ngrams(list, i)
        dict_ngrams[i] = nltk.FreqDist(n_gram)
    return dict_ngrams


def prepare_language_ngrams(language, word_n=5, letter_n=10, distinctive_features_n=12):
    language_ngrams = {}

    tokens = [word.lower() for word in language.words() if word.isalnum()]
    letters = [char for word in tokens for char in word]
    encoded_letters = encode_letters(letters[:1000000])

    cfdist = ConditionalFreqDist((len(word), word) for word in tokens)
    list_punctuation = list(string.punctuation)
    [cfdist[1].pop(punctuation, '') for punctuation in list_punctuation]
    language_ngrams['words_length'] = (cfdist, "words of specified lengths")

    word_ngrams = ngrams_freq_dist(word_n, tokens)
    language_ngrams['word_ngrams'] = (word_ngrams, "n-grams of words")

    letter_ngrams = ngrams_freq_dist(letter_n, letters)
    language_ngrams['letter_ngrams'] = (letter_ngrams, "n-grams of characters")

    distinctive_features_ngrams = ngrams_freq_dist(distinctive_features_n, encoded_letters)
    language_ngrams['letter_features_ngrams'] = (distinctive_features_ngrams, "n-grams of distinctive features of characters")

    return language_ngrams


def prepare_microstates_ngrams(microstates_string, microstates_n=6):
    microstates_n_grams = ngrams_freq_dist(microstates_n, microstates_string)
    
    return microstates_n_grams


def plot_microstates_to_language(microstates_ngrams, language_ngrams):
    nrows = len(language_ngrams.keys())

    fig, ax = plt.subplots(ncols=2, nrows=nrows)
    plt.subplots_adjust(hspace=0.3, top=1.8)

    for row, key in enumerate(list(language_ngrams.keys())):
        plot_ngrams_ax(microstates_ngrams, ax[row, 0], title_matter="n-grams of microstates")
        plot_ngrams_ax(language_ngrams[key][0], ax[row, 1], title_matter=language_ngrams[key][1])


class Fake_eeg:
    def __init__(self, raw_signal, channels):
        self.eeg_raw_channels = {channel_name: np.array([[raw_signal[i]]]) for i, channel_name in enumerate(channels)}
        self.ch_names = channels

    def __getitem__(self, channel):
         return self.eeg_raw_channels[channel]
    
    def apply_func(self, func):
        self.__init__(func(self).to_raw_signal(), self.ch_names)

    def to_raw_signal(self):
        return np.array([self.eeg_raw_channels[channel][0][0] for channel in self.ch_names])


class EegArray(mne.io.array.array.RawArray):
    def __init__(self, eeg_signal, info, annotation_mask, ordered_annotation_mask):
        mne.io.array.array.RawArray.__init__(self, eeg_signal, info)
        self.annotation_mask = annotation_mask
        self.ordered_annotation_mask = ordered_annotation_mask
        
    def generate_normalized_dataframe(self):
        self.raw_df = self.to_data_frame()
        new_df = self.raw_df.copy()
        new_df['min'] = new_df[self.ch_names].min(axis=1)
        new_df['max'] = new_df[self.ch_names].max(axis=1)
        for channel in self.ch_names:
            new_df[channel] = 2*((new_df[channel] - new_df['min']) / (new_df['max']-new_df['min'])) - 1
        self.normalized_df = new_df[['time']+self.ch_names]

    def calculated_grouped_mean(self, mean_frequency=60):
        self.grouped_mean_df = self.normalized_df.copy()
        self.grouped_mean_df['mean_group'] = np.floor(self.grouped_mean_df.index/mean_frequency).astype(int)
        self.grouped_mean_df = self.grouped_mean_df.groupby('mean_group').mean().reset_index().drop('mean_group', axis=1)