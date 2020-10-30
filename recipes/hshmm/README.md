# H-SHMM #
Original model described in the paper: *"A Hierarchical Subspace Model for Language-Attuned Acoustic Unit Discovery"*

To get the results in the paper,
#### First train the hyper-subspace parameters on the source languages ####
`./run_hmonophone.sh data.conf`

#### Then run AUD for each target language ####
`./run_hsubspace_aud.sh timit space/exp_mix/globalphone/globalphone/sw_am_wo_FR_GE_SP_PO/hsubspace_monophone_mfcc_4g_gamma_dirichlet_process_ldim6_100/`
`./run_hsubspace_aud.sh mboshi space/exp_mix/globalphone/globalphone/sw_am_wo_FR_GE_SP_PO/hsubspace_monophone_mfcc_4g_gamma_dirichlet_process_ldim6_100/`
`./run_hsubspace_aud.sh google_lr space/exp_mix/globalphone/globalphone/sw_am_wo_FR_GE_SP_PO/hsubspace_monophone_mfcc_4g_gamma_dirichlet_process_ldim6_100/ yoruba`

#### Data preparation ###
Make sure that each corpus is downloaded and that the paths are correct in `local/${corpus}/prepare_data.sh`

The *TIMIT* and *Globalphone* corpora are not freely available. If you don't have them, remove the *Globalphone* entry from `data.conf`, and simply don't run AUD for *TIMIT*

To use other corpora, create your own `local/${own_corpus_name}/prepare_data.sh`
and edit `data.conf` to have the correct corpus names.
This script should create a directory with:
- `uttids`: list of utterances
- `wav.scp`: map of each utterance to the wav file or a piped command e.g. `utt1 sox /path/to/utt1.wav -r 16k -b 16 -t wav -  |`
`utt2 /path/to/utt2.wav`
- `trans`: phone-level transcription file. Only required for source languages e.g:
`utt1 phone1 phone2 phone3`
`utt2 phone14 phone12 phone7`
- `ali`: optional time-aligned transcriptions used as reference for scoring AUD output, e.g.
`utt1 sil sil sil phone1 phone1 phone1 phone2 phone2 phone3 phone3 phone 3 sil sil sil`
`utt2 sil sil sil phone14 phone14 phone12 phone12 phone7 phone7 phone7 phone7 sil sil`
