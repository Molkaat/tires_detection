<img width="839" height="789" alt="Screenshot 2025-07-20 230559" src="https://github.com/user-attachments/assets/42ad9085-5ab0-4f12-ba0a-0fe043a8d042" />
<img width="883" height="834" alt="Screenshot 2025-07-20 224607" src="https://github.com/user-attachments/assets/f0bae4a2-9b4c-4c69-b140-ca1f1d655ee0" />
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
