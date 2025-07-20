# TireCount Pro

AI-powered tire detection and counting system using pyramid detection method.

## Setup

1. Clone the repository
```bash
git clone https://github.com/Molkaat/tires_detection.git
```

2. Install requirements
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your Roboflow API key:
```
ROBOFLOW_API_KEY=your_api_key_here
```

4. Place input images in the `input_images` folder

5. Run the script:
```bash
python main.py
```

## Project Structure
- `main.py`: Core detection logic
- `streamlit.py`: Web interface
- `input_images/`: Place input images here
- `output/`: Detection results and reports