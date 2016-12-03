from eegOSC_class import eeg_to_osc

e = eeg_to_osc(source = "online", serial_number = 'SN20120229000348', buffer_size = 128, dump_online_file = True)
e.run()