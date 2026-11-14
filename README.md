# AI Sign Language Recognition (Real-Time)

Recognizes hand signs from a webcam using **MediaPipe Hands** (21 landmarks) and a
**Random Forest** classifier. Works for alphabets (A–Z) or custom words/gestures.
Includes sentence building and text-to-speech.

## Pipeline
Webcam -> MediaPipe hand landmarks -> normalise (wrist-origin, scale-invariant, 63 features)
-> classifier -> temporal smoothing -> text / speech

## Setup (Python 3.9 – 3.11 recommended)
```bash
python -m venv venv
venv\Scripts\activate        # Windows   (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
```

## Usage
1. **Collect data** (aim for 200+ samples per sign, vary angle/distance/lighting)
   ```bash
   python collect_data.py                              # alphabets A-Z
   python collect_data.py --labels HELLO,YES,NO,THANKS # custom words (keys 1-4)
   ```
   Press a letter (or number) to select a label, SPACE to start/stop recording, q to save & quit.
   Run it multiple times - data is appended to `data/landmarks.csv`.
2. **Train**
   ```bash
   python train.py
   ```
   Prints accuracy/report, saves `models/sign_model.pkl` and `outputs/confusion_matrix.png`.
3. **Run live recognition**
   ```bash
   python recognize.py
   ```
   Hold a sign ~1 s to add it. Keys: `q` quit, `c` clear, `b` backspace, `s` speak.

## Project structure
```
collect_data.py   capture landmark samples
train.py          train + evaluate model
recognize.py      real-time recognition app
utils.py          landmark extraction & normalisation
data/             landmarks.csv (created by you)
models/           sign_model.pkl (created by train.py)
outputs/          confusion matrix
```

## Tips
- Static signs only (letters/poses). Motion signs (J, Z, many words) need a sequence model (LSTM) - a good extension.
- Poor accuracy? Collect more varied data, use good lighting, one signer at a time.
- Change camera with `--camera 1`; tune `--conf` and `--hold` in recognize.py.

## Ideas to extend
LSTM/GRU over 30-frame sequences, two-hand support, Flask/Streamlit web UI, TFLite for mobile.
