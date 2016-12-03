# python default
from __future__ import print_function
import time, threading, sys
import multiprocessing as mp

# external
import numpy as np
import OSC
from keypy.signal import band_pass
from emokit.emotiv import Emotiv
# mne if reading EDF data
# scipy if detrending while filtering



class eeg_to_osc(object):
    """
    Holds EEG data if offline, is connected to Emotiv Epoc if online.
    Offline data:
        *.txt - array as time x channels - need channel list with names!
        *.edf
    Handles data preprocessing and computation of parameters.
    Sends and recieves OSC messages via UDP on localhost.
    Preprocessing functions:
        average reference - _average_ref()
        filtering - filter_data(cutoff = [low, high], detrend = bool, overwrite = True)
        spectral centroid - spectral_centroid(channel = int or None, ts = if other time series, 
            N = window size or None), OSC keyword - 'spec_c'
        Hjorth activity - hjorth_activity(channel = int or None, ts = if other time series, 
            N = window size or None), OSC keyword - 'hj_act'
        Hjorth mobility - hjorth_mobility(channel = int or None, ts = if other time series, 
            N = window size or None), OSC keyword - 'hj_mob'
        Hjorth complexity - hjorth_complexity(channel = int or None, ts = if other time series, 
            N = window size or None), OSC keyword - 'hj_com'

    OSC messages are sent with address as '/ch[int]/[str]' where int is number of channel (default 4)
    so 1-4 and [str] is keyword for different data parameter (see above)
    """

    def __init__(self, source = "online", serial_number = None, input_file = None, sampling_rate = 128, channel_list = None, buffer_size = None, dump_online_file = False):
        """
        source:
            online - emotiv kit -- serial number
            offline - offline data -- input file
        """

        self.source = source
        
        if source not in ['online', 'offline']:
            raise Exception("Unknown source. Use 'online' or 'offline'.")
        
        if self.source == 'online':

            from emokit.emotiv import Emotiv

            if serial_number is None:
                raise Exception("For online - Emotiv headset - serial number must be provided.")
            self.serial_number = serial_number
            # self.headset = Emotiv(display_output = False, serial_number = self.serial_number, 
            #     write_values = False)

            self.sampling_rate = 128 # emotiv is 128Hz

            # division as 4 channels - frontal, temp right, temp left, parieto-occipital
            self.frontal = ['F3', 'F4', 'AF3', 'AF4']
            self.temporal_right = ['T8', 'FC6', 'F8']
            self.temporal_left = ['T7', 'FC5', 'F7']
            self.parietal_occipital = ['O1', 'O2', 'P7', 'P8']
            
            # division as 2 channels - left, right
            self.left_channels = ['AF3', 'F3', 'P7', 'O1'] + self.temporal_left
            self.right_channels = ['AF4', 'F4', 'O2', 'P8'] + self.temporal_right

            # full channel list
            self.channel_list = self.left_channels + self.right_channels

            if dump_online_file:
                self.dump_online_file = True
                self.online_data_to_write = dict(zip(self.channel_list, [[] for i in range(len(self.channel_list))]))
            else:
                self.dump_online_file = False


        if self.source == 'offline':
            if input_file is None:
                raise Exception("For offline data the input file name must be provided.")
            else:
                if input_file[-3:] == "txt":
                    self.raw_data = np.loadtxt(input_file) # time x channels
                    if channel_list is None:
                        raise Exception("For text file channel list must be included!")
                    self.channel_list = channel_list
                    self.sampling_rate = sampling_rate
                    self.frontal = ['Fp1', 'Fp2', 'F7', 'F8']
                    self.temporal_right = ['T4', 'T6', 'C4', 'F4']
                    self.parietal_occipital = ['P3', 'P4', 'O1', 'O2']
                    self.temporal_left = ['T3', 'T5', 'C3', 'F3']

                elif input_file[-3:] == 'edf':
                    import mne
                    raw = mne.io.read_raw_edf(input_file, preload = True)
                    self.raw_data = raw.to_data_frame().as_matrix()
                    self.channel_list = raw.ch_names
                    self.sampling_rate = int(raw.info['sfreq'])
                    self.frontal = ['FP1-F7', 'FP2-F8']
                    self.temporal_right = ['F8-T8', 'T8-P8']
                    self.temporal_left = ['F7-T7', 'T7-P7']
                    self.parietal_occipital = ['P7-O1', 'P8-O2']
                    self.right = ['FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2', 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2']
                    self.left = ['FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1', 'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1']
                    del raw

        self.osc_client_setup = False
        self.osc_server_setup = False

        self.send_data = None
        
        if buffer_size is None:
            self.buffer_size = self.sampling_rate
        else:
            self.buffer_size = buffer_size


    def prepare_ts_to_send(self):
        """
        Handles what to send as OSC message. 
        
        !!! Need to modify OSC server according to what is sent (e.g. Pd OSC server) !!!
        """

        self._data_to_channel_dict()
        
        self.send_data = {}
        
        ## AVERAGE OF FRONTAL, TEMPORAL AND PARIETO-OCCIPITAL CHANNELS
        # frontal - ch1
        # self.send_data['ch1'] = np.mean(np.array(list(self.dict_data[k] for k in self.frontal)), axis = 0)
        # left up to 20Hz
        to_filter = np.array(list(self.dict_data[k] for k in self.left)).T
        self.filter_data(cutoff = [1., 20.], overwrite = False, ts = to_filter)
        self.send_data['ch1'] = np.mean(self.filt_data, axis = 1)
        
        # temporal right - ch2
        # self.send_data['ch2'] = np.mean(np.array(list(self.dict_data[k] for k in self.temporal_right)), axis = 0)
        # right up to 20Hz
        to_filter = np.array(list(self.dict_data[k] for k in self.right)).T
        self.filter_data(cutoff = [1., 20.], overwrite = False, ts = to_filter)
        self.send_data['ch2'] = np.mean(self.filt_data, axis = 1)

        # parietal + occipital - ch3
        # self.send_data['ch3'] = np.mean(np.array(list(self.dict_data[k] for k in self.parietal_occipital)), axis = 0)
        # left 20-40Hz
        to_filter = np.array(list(self.dict_data[k] for k in self.left)).T
        self.filter_data(cutoff = [20., 40.], overwrite = False, ts = to_filter)
        self.send_data['ch3'] = np.mean(self.filt_data, axis = 1)


        # temporal left - ch4
        # self.send_data['ch4'] = np.mean(np.array(list(self.dict_data[k] for k in self.temporal_left)), axis = 0)
        # right 20-40Hz
        to_filter = np.array(list(self.dict_data[k] for k in self.right)).T
        self.filter_data(cutoff = [20., 40.], overwrite = False, ts = to_filter)
        self.send_data['ch4'] = np.mean(self.filt_data, axis = 1)


    def _average_ref(self):
        """
        Averages the data to common reference.
        """

        computed_mean = self.raw_data.mean(axis=1)
        self.raw_data = self.raw_data - computed_mean[:, np.newaxis]


    def filter_data(self, cutoff = [1., 40.], detrend = False, overwrite = True, ts = None):
        """
        Filters data with box filter.
        Detrend keyword whether to use linear detrending.

        TAKEN FROM KEYPY -- https://github.com/keyinst/keypy
        """

        if ts is None:
            x_all_channels = np.zeros(self.raw_data.shape, dtype = self.raw_data.dtype)
            time_frame = self.raw_data.shape[0]
        else:
            x_all_channels = np.zeros(ts.shape, dtype = ts.dtype)
            time_frame = ts.shape[0]
        frequency_bins = np.fft.fftfreq(time_frame, 1./self.sampling_rate)

        ch_num = self.raw_data.shape[1] if ts is None else ts.shape[1]

        for ch in range(ch_num):
            if ts is None:
                x = self.raw_data[:,ch]
            else:
                x = ts[:, ch]
    
            if detrend:
                import scipy.signal as ss
                x = ss.detrend(x)

            # loop across 2second epochs
            for i in range(len(x)/time_frame):
                epoch = x[i*time_frame:(i+1)*time_frame]
                epoch_fft = np.fft.fftpack.fft(epoch)
                selector = band_pass(frequency_bins, cutoff[0], cutoff[1])
                epoch_fft[selector] = 0
                epoch_reverse_filtered = np.fft.fftpack.ifft(epoch_fft)
                x_all_channels[i*time_frame:(i+1)*time_frame,ch] = np.real(epoch_reverse_filtered)
                            
        if overwrite:     
            self.raw_data = x_all_channels.copy()
        else:
            self.filt_data = x_all_channels.copy()


    def spectral_centroid(self, channel = None, ts = None, N = None):
        """
        Computes spectral centroid of the signal.
        channel is either index or None of average over channels.
        ts for external time series
        N is window length or if None the centroid is stationary
        """

        if channel is not None and channel > len(self.channel_list):
            raise Exception("Channel either valid index or None for average!")

        if channel is not None or ts is not None:
            if ts is None:
                x = self.raw_data[:, channel]
            else:
                x = ts.copy()
            if N is None:
                magnitudes = np.abs(np.fft.rfft(x)) # magnitudes of positive frequencies
                length = len(x)
                freqs = np.abs(np.fft.fftfreq(length, 1.0/self.sampling_rate)[:length//2+1]) # positive frequencies
                return np.sum(magnitudes*freqs) / np.sum(magnitudes)
            else:
                cent = []
                for w in range(x.shape[0]//N):
                    x_part = x[w*N : (w+1)*N]
                    magnitudes = np.abs(np.fft.rfft(x_part)) # magnitudes of positive frequencies
                    length = len(x_part)
                    freqs = np.abs(np.fft.fftfreq(length, 1.0/self.sampling_rate)[:length//2+1]) # positive frequencies
                    cent.append(np.sum(magnitudes*freqs) / np.sum(magnitudes))
                return np.array(cent)
        
        elif channel is None and ts is None:
            spc_centroids = []
            for ch in range(len(self.channel_list)):
                x = self.raw_data[:, ch]
                if N is None:
                    magnitudes = np.abs(np.fft.rfft(x)) # magnitudes of positive frequencies
                    length = len(x)
                    freqs = np.abs(np.fft.fftfreq(length, 1.0/self.sampling_rate)[:length//2+1]) # positive frequencies
                    spc_centroids.append(np.sum(magnitudes*freqs) / np.sum(magnitudes))
                else:
                    cent = []
                    for w in range(x.shape[0]//N):
                        x_part = x[w*N : (w+1)*N]
                        magnitudes = np.abs(np.fft.rfft(x_part)) # magnitudes of positive frequencies
                        length = len(x_part)
                        freqs = np.abs(np.fft.fftfreq(length, 1.0/self.sampling_rate)[:length//2+1]) # positive frequencies
                        cent.append(np.sum(magnitudes*freqs) / np.sum(magnitudes))
                    spc_centroids.append(np.array(cent))
            return np.mean(np.array(spc_centroids), axis = 0)


    def hjorth_activity(self, channel = None, ts = None, N = None):
        """
        Computes Hjorth activity of the signal.
        channel is either index or None of average over channels.
        ts for external time series
        N is window length or if None the mobility is stationary
        """

        if channel is not None and channel > len(self.channel_list):
            raise Exception("Channel either valid index or None for average!")

        if channel is not None or ts is not None:
            if ts is None:
                x = self.raw_data[:, channel]
            else:
                x = ts.copy()
            
            if N is None:
                return np.var(x, ddof = 1)
            
            else:
                act = []
                for w in range(x.shape[0]//N):
                    x_part = x[w*N : (w+1)*N]
                    act.append(np.var(x_part, ddof = 1))
                return np.array(act)
        
        elif channel is None and ts is None:
            hj_acts = []
            for ch in range(len(self.channel_list)):
                x = self.raw_data[:, ch]
                if N is None:
                    hj_acts.append(np.var(x, ddof = 1))
                
                else:
                    act = []
                    for w in range(x.shape[0]//N):
                        x_part = x[w*N : (w+1)*N]
                        act.append(np.var(x_part, ddof = 1))
                    hj_acts.append(np.array(act))
            
            return np.mean(np.array(hj_acts), axis = 0)


    def hjorth_mobility(self, channel = None, ts = None, N = None):
        """
        Computes Hjorth mobility of the signal.
        channel is either index or None of average over channels.
        ts for external time series
        N is window length or if None the mobility is stationary
        """

        if channel is not None and channel > len(self.channel_list):
            raise Exception("Channel either valid index or None for average!")

        if channel is not None or ts is not None:
            if ts is None:
                x = self.raw_data[:, channel]
            else:
                x = ts.copy()
            
            if N is None:
                deriv1 = np.diff(np.insert(x, 0, 0))
                return np.sqrt(np.var(deriv1, ddof = 1) / np.var(x, ddof = 1))
            
            else:
                mob = []
                for w in range(x.shape[0]//N):
                    x_part = x[w*N : (w+1)*N]
                    deriv1 = np.diff(np.insert(x_part, 0, 0))
                    mob.append(np.sqrt(np.var(deriv1, ddof = 1) / np.var(x_part, ddof = 1)))
                return np.array(mob)
        
        elif channel is None and ts is None:
            hj_mobs = []
            for ch in range(len(self.channel_list)):
                x = self.raw_data[:, ch]
                if N is None:
                    deriv1 = np.diff(np.insert(x, 0, 0))
                    hj_mobs.append(np.sqrt(np.var(deriv1, ddof = 1) / np.var(x, ddof = 1)))
                
                else:
                    mob = []
                    for w in range(x.shape[0]//N):
                        x_part = x[w*N : (w+1)*N]
                        deriv1 = np.diff(np.insert(x_part, 0, 0))
                        mob.append(np.sqrt(np.var(deriv1, ddof = 1) / np.var(x_part, ddof = 1)))
                    hj_mobs.append(np.array(mob))
            
            return np.mean(np.array(hj_mobs), axis = 0)


    def hjorth_complexity(self, channel = None, ts = None, N = None):
        """
        Computes Hjorth complexity of the signal.
        channel is either index or None of average over channels.
        ts for external time series
        N is window length or if None the mobility is stationary
        """

        if channel is not None and channel > len(self.channel_list):
            raise Exception("Channel either valid index or None for average!")

        if channel is not None or ts is not None:
            if ts is None:
                x = self.raw_data[:, channel]
            else:
                x = ts.copy()

            deriv1 = np.diff(np.insert(x, 0, 0))
            
            return self.hjorth_mobility(ts = deriv1, N = N) / self.hjorth_mobility(ts = x, N = N)
        
        elif channel is None and ts is None:
            hj_mobs = []
            for ch in range(len(self.channel_list)):
                x = self.raw_data[:, ch]
                deriv1 = np.diff(np.insert(x, 0, 0))
                
                hj_mobs.append(self.hjorth_mobility(ts = deriv1, N = N) / self.hjorth_mobility(ts = x, N = N))
            
            return np.mean(np.array(hj_mobs), axis = 0)


    def _data_to_channel_dict(self):
        """
        Converts data to dictionary with 1d array per channel.
        """

        if self.channel_list is None:
            raise Exception("Channel list must be defined.")

        self.dict_data = {}
        for ch in self.channel_list:
            self.dict_data[ch] = self.raw_data[:, self.channel_list.index(ch)]


    def setup_osc_client(self, port = 9001, address = 'localhost'):
        """
        Sets OSC client to send messages to Pd.
        default port is 9001 on localhost.
        """

        self.osc_client = OSC.OSCClient()
        if address == 'localhost':
            address = '127.0.0.1'
        self.osc_client.connect((address, port)) 
        self.osc_client_setup = True


    def setup_osc_server(self, port = 9002, address = 'localhost'):
        """
        Sets OSC server to listening for messages from Pd.
        default port is 9002 on localhost.
        """

        if address == 'localhost':
            address = '127.0.0.1'
        self.osc_server = OSC.OSCServer((address, port))
        self.osc_server.addDefaultHandlers()
        # TODO add handlers

        self.osc_thread = threading.Thread( target = self.osc_server.serve_forever )
        self.osc_thread.setDaemon(True)
        self.osc_server_setup = True


    def _osc_server_handler(self, addr, tags, data, source):
        """
        Handler for incoming OSC data.
        """

        pass


    def _osc_msg(self, address, msg):
        """
        Helper for sending single OSC message.
        """

        if not self.osc_client_setup:
            raise Exception("First setup OSC client with self.setup_osc_client().")
        oscmsg = OSC.OSCMessage()
        oscmsg.setAddress("%s" % address)
        oscmsg.append(msg)
        self.osc_client.send(oscmsg)


    def stop(self):
        """
        Stops OSC server.
        """

        if self.source == 'online' and self.dump_online_file:
            to_write = np.array([self.online_data_to_write[ch] for ch in self.channel_list]).T
            print(to_write.shape)
            np.savetxt("emotiv_raw_data_%s.txt" % str(time.time()), to_write, fmt = "%.6f")
        self.osc_server.close()
        print("Waiting for OSC server thread to finish...")
        self.osc_thread.join()
        if self.source == 'online':
            # self.headset.stop()
            self.eeg_reader_thread.join()
            self.eeg_process_thread.join()
        print("Done")


    def _thread_reading_online(self):
        """
        Helper function for reading online data.
        To be used as a thread, putting data to queue.
        """

        online_data = dict(zip(self.channel_list, [[] for i in range(len(self.channel_list))]))
        counter = 0
        
        headset = Emotiv(display_output = False, serial_number = self.serial_number, 
                write_values = False)
        # epoc_channels = [k[1] for k in enumerate(self.headset.sensors)]

        try:
            start_time = time.time()
            while True:
                packet = headset.dequeue()
                if packet is not None:
                    for ch in self.channel_list:
                        quality = packet.sensors[ch]['quality']
                        data = packet.sensors[ch]['value']
                        online_data[ch].append((data * 0.51) / 1000.) # in mV
                    counter += 1
                    if counter >= self.buffer_size:
                        ratio = self.sampling_rate // self.buffer_size
                        # assert ((time.time() - start_time) % (1./ratio)) < 0.075 # assert sampling rate
                        # print((time.time() - start_time) % (1./ratio))
                        # if not ((time.time() - start_time) % 1) < 0.05:
                            # print("WARNING: sampling rate is low!")
                        self.online_data_queue.put([online_data, headset.battery])
                        online_data = dict(zip(self.channel_list, [[] for i in range(len(self.channel_list))]))
                        counter = 0

        except KeyboardInterrupt:
            print("stopping...")
            self.stop()


    def _thread_process_online(self):
        """
        Helper function for processing online data.
        To be used as a thread, reading data from queue.
        """

        ticker_start = time.time()
        try:
            while True:
                online_data, battery = self.online_data_queue.get()
                if self.dump_online_file:
                    for ch in online_data:    
                        self.online_data_to_write[ch] += online_data[ch]

                # to 2d array for basic preprocessing
                self.raw_data = np.array([online_data[ch] for ch in self.channel_list]).T
                self._average_ref()
                self.filter_data()
                self.prepare_ts_to_send()
                for i in range(1,5):
                    ch = 'ch%d' % i
                    buffer_data = self.send_data[ch]
                    self._osc_msg(address = '/%s/spec_c' % (ch), 
                        msg = self.spectral_centroid(ts = buffer_data, N = self.buffer_size))
                    self._osc_msg(address = '/%s/hj_act' % (ch), 
                        msg = self.hjorth_activity(ts = buffer_data, N = self.buffer_size))
                    self._osc_msg(address = '/%s/hj_mob' % (ch), 
                        msg = self.hjorth_mobility(ts = buffer_data, N = self.buffer_size))
                    self._osc_msg(address = '/%s/hj_com' % (ch), 
                        msg = self.hjorth_complexity(ts = buffer_data, N = self.buffer_size))
                s = "...running online - %ssec..." % str(int(time.time() - ticker_start))
                if battery < 20 and (time.time() - ticker_start) > 1:
                    s = ("WARNING: low battery - %d%%!!!" % battery) + s
                print(s, end = '')
                sys.stdout.flush()
                print((b'\x08' * len(s)).decode(), end = '')
                
        except KeyboardInterrupt:
            print("stopping...")
            self.stop()


    def run(self):
        """
        Main loop.
        """

        print("---------------------------------")
        print("Initializing....")
        print("Use ctrl-C to quit anytime...")
        start_time = time.time()
        if self.source == 'offline':
            print("Preparing offline data")
            self._average_ref()
            self.filter_data()
        print("Starting OSC client and server...")
        self.setup_osc_server()
        self.setup_osc_client()
        if self.osc_server_setup and self.osc_client_setup:
            self.osc_thread.start()
            print("OSC up and running...")
        else:
            raise Exception("Something went wrong with the OSC protocol.")
        
        if self.source == 'online':
            # 2 processes simultaneously - one for continuous reading, another for processing
            self.online_data_queue = mp.Queue()
            self.eeg_reader_thread = mp.Process(target = self._thread_reading_online)
            self.eeg_reader_thread.start()
            self.eeg_process_thread = mp.Process(target = self._thread_process_online)
            self.eeg_process_thread.start()

        # time.sleep(3)
        if self.source == 'offline':
            sleep_time = self.sampling_rate / self.buffer_size
            idx = 0
        
        # main loop for offline data
        # loop for online data is handled in the threads
        if self.source == "offline":

            self.prepare_ts_to_send()
            
            try:
                while True:
                    # loop through data using buffer size

                    if idx > self.raw_data.shape[0] - self.buffer_size:
                        print("end of offline data!")
                        print("stopping...")
                        self.stop()
                        break

                    # loop through channels
                    for i in range(1,5):
                        ch = 'ch%d' % i
                        buffer_data = self.send_data[ch][idx : idx+self.buffer_size]
                        
                        # send OSC
                        self._osc_msg(address = '/%s/spec_c' % (ch), 
                            msg = self.spectral_centroid(ts = buffer_data, N = self.buffer_size))
                        self._osc_msg(address = '/%s/hj_act' % (ch), 
                            msg = self.hjorth_activity(ts = buffer_data, N = self.buffer_size))
                        self._osc_msg(address = '/%s/hj_mob' % (ch), 
                            msg = self.hjorth_mobility(ts = buffer_data, N = self.buffer_size))
                        self._osc_msg(address = '/%s/hj_com' % (ch), 
                            msg = self.hjorth_complexity(ts = buffer_data, N = self.buffer_size))

                    # advance index
                    idx += self.buffer_size

                    # print block count
                    s = "...block %d/%d..." % (idx//self.buffer_size, self.raw_data.shape[0]//self.buffer_size)
                    print(s, end = '')
                    sys.stdout.flush()
                    print((b'\x08' * len(s)).decode(), end = '')
                    
                    # wait if needed to match buffer size
                    time.sleep(1. - ((time.time() - start_time) % 1))

            except KeyboardInterrupt:
                print("stopping...")
                self.stop()





