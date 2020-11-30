Team name: AUGment

Working paper title: Emotion and Theme Recognition in Music using Attention-based methods

Requirements:
    Create virtual environment and install requirements

    python -m venv venv
    source venv/bin/activate
    pip install -r scripts/requirements.txt

Data: precomputed mel-spectrograms of split-0 of MTG-Jamendo Dataset

Assuming you are working in scripts\teamAUGment folder

1. Preprocessing 
python get_npy.py run 'your_path_to_spectrogram_npy'

2. Train
python main.py --mode 'TRAIN' --subset='moodtheme' --audio_path='your_path_to_spectrogram_npy'

3. Test
python main.py --mode 'TEST' --subset='moodtheme' --audio_path='your_path_to_spectrogram_npy'

4. Fusion of results (Performs late-fusion of predictions.npy of individual models made available in 'results' folder)
python fusion.py -o fusion_results

Note: For enabling AReLU activation (corresponding to Submission2 method), commented code in model.py to be uncommented.  
