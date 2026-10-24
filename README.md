# SAP Flow Graph + LLM Query

A modern web application for exploring SAP business data through interactive graph visualization and natural language querying.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Features

- **Interactive Graph Visualization**: Explore relationships between SAP entities using an intuitive graph explorer
- **Natural Language Querying**: Ask questions about your SAP data in plain English
- **Dataset-Grounded Answers**: All responses are based on actual data from your SAP dataset
- **Beautiful UI**: Modern, responsive interface with light/dark theme support
- **Real-time Statistics**: Dashboard showing key metrics from your dataset

## Supported Entities

- Sales Orders
- Deliveries
- Billing Documents
- Journal Entries / Accounting Documents
- Customers
- Products / Materials
- Plants

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sap-project.git
cd sap-project
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Build the local database:
```bash
python scripts/build_db.py --data-dir data --db data/sap_graph.sqlite
```

5. Start the application:
```bash
streamlit run streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

## Configuration

### LLM Provider (Optional)

For enhanced natural language querying, configure an LLM provider:

Create a `.env` file in the project root:

```env
LLM_API_KEY=your_api_key_here
LLM_BASE_URL=https://api.example.com/v1  # Optional
LLM_MODEL=model-name  # Optional
```

Supported providers:
- OpenRouter
- Groq
- Gemini (free tiers available)

If no API key is provided, the app will still work using heuristic-based query templates.

## Usage

### Graph Explorer

1. Select a node type from the sidebar (e.g., SalesOrder, Delivery)
2. Enter a node ID (e.g., 740507)
3. Set the expansion depth (1-4)
4. Click "Expand neighbors" to visualize the graph

### Chat Interface

Ask questions about your SAP data:

- "Trace billing document 90504248"
- "Which products have the highest number of billing documents?"
- "Which sales orders are delivered but not billed?"

## Project Structure

```
sap-project/
├── data/                      # JSONL dataset files
├── scripts/                   # Database build scripts
│   └── build_db.py
├── src/                       # Source code
│   ├── db.py                  # Database operations
│   ├── llm_client.py          # LLM integration
│   └── queries.py             # Query engine
├── streamlit_app.py           # Main application
├── requirements.txt           # Python dependencies
├── .env.txt                  # Environment variables template
└── README.md                 # This file
```

## Development

### Adding New Query Intents

Edit `streamlit_app.py` to add new heuristic patterns or LLM intents in the `_heuristic_intent` function.

### Customizing the UI

Modify the CSS in the `main()` function of `streamlit_app.py` to change colors, fonts, or layout.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Built with [Streamlit](https://streamlit.io)
- Graph visualization powered by [PyVis](https://pyvis.readthedocs.io)
- Database: [SQLite](https://www.sqlite.org)
