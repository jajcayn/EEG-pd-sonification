from eegOSC_class import eeg_to_osc


ch_list = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4','P3','P4','O1','O2','F7','F8','T3','T4','T5','T6','Fz','Cz','Pz']
e = eeg_to_osc(source = "offline", input_file = "offline-edf/chb01_01.edf", buffer_size = 128//2)
e.run()

# print spectral_centroid(t.shape)