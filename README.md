# Mushroom Classification MLOps Project

A complete MLOps pipeline for mushroom classification using FastAPI, MLflow, and Docker.

## Features

- 🍄 Mushroom classification using machine learning
- 🚀 FastAPI for real-time predictions
- 📊 MLflow for experiment tracking
- 🐳 Docker for containerization
- 🔄 CI/CD ready
- 📈 Model versioning and tracking

## Quick Start

### Prerequisites

- Python 3.8+
- Docker and Docker Compose (optional)
- Git

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Amarthya-DG/mushroom-classification-mlops.git
cd mushroom-classification-mlops
```

2. Create and activate virtual environment:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix or MacOS:
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Project

#### Option 1: Local Development

1. Train the model:
```bash
python src/models/train.py
```

2. Start the API:
```bash
python src/api/main.py
```

Access the API at http://localhost:8000/docs

#### Option 2: Using Docker

1. Build and run:
```bash
docker-compose up --build
```

Access:
- API: http://localhost:8000/docs
- MLflow: http://localhost:5000

### API Usage

Make predictions using curl:
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "cap_shape": "x",
           "cap_surface": "s",
           "cap_color": "n",
           "has_bruises": true,
           "odor": "p",
           "gill_attachment": "f",
           "gill_spacing": "c",
           "gill_size": "n",
           "gill_color": "k",
           "stalk_shape": "e",
           "stalk_surface_above_ring": "s",
           "stalk_surface_belows_ring": "s",
           "stalk_color_above_ring": "w",
           "stalk_color_below_ring": "w",
           "veil_type": "p",
           "veil_color": "w",
           "number_of_rings": "o",
           "ring_type": "p",
           "spore_print_color": "k",
           "population": "s",
           "habitat": "u"
         }'
```

## Project Structure

```
.
├── models/                # Saved models and artifacts
├── notebooks/            # Jupyter notebooks
├── src/                  # Source code
│   ├── api/             # FastAPI application
│   ├── data/            # Data processing
│   ├── models/          # Model training
│   └── utils/           # Utilities
├── tests/               # Tests
├── .gitignore          # Git ignore file
├── docker-compose.yml  # Docker compose config
├── Dockerfile         # Docker config
├── requirements.txt   # Python dependencies
└── README.md         # Project documentation
```

## Dataset

The project uses the Mushroom Classification dataset from Hugging Face datasets (`mstz/mushroom`). Features include:
- Cap characteristics (shape, surface, color)
- Gill characteristics
- Stalk characteristics
- Veil characteristics
- And more...

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



