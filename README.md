# SCFinShield-AI

Multi-Tier Supply Chain Finance Fraud Detection and Management System

## Overview

SCFinShield-AI is an advanced fraud detection and management platform designed for supply chain finance operations. The system uses machine learning, graph analysis, and AI-powered investigation tools to identify, analyze, and manage fraudulent activities in multi-tier supply chain transactions. It provides real-time risk assessment, automated investigation support, and comprehensive reporting capabilities to protect organizations from financial fraud.

## Key Features

- **Fraud Detection**: Machine learning models that identify suspicious patterns in invoices and transactions
- **Graph Analysis**: Network-based analysis of supply chain relationships and transaction flows
- **Risk Assessment**: Multi-level risk scoring with configurable thresholds (review, hold, approve)
- **AI-Powered Investigation**: LangGraph-based investigation system for automated fraud analysis
- **Knowledge Base**: RAG-enabled system for storing and retrieving fraud-related information
- **Dashboard**: Real-time visualization of fraud risks and transaction status
- **Document Processing**: PDF extraction and analysis capabilities

## Technology Stack

- **Backend**: FastAPI 0.115.0 with Python 3.11
- **Frontend**: React 18.3.1 with Vite and modern UI libraries
- **Databases**: 
  - Supabase (PostgreSQL) for relational data
  - Neo4j Aura for graph-based analysis
  - Pinecone for vector embeddings
- **ML & AI**: LangChain, LangGraph, Anthropic API, XGBoost, PyTorch
- **Feature Store**: Feast for ML feature management
- **Supporting Libraries**: SQLAlchemy, asyncpg, scikit-learn, pandas, NetworkX

## Project Structure

```
backend/
  api/              API endpoints (fraud, invoices, graph, investigation, dashboard, etc.)
  core/             Configuration, logging, security settings
  db/               Database connections (Neo4j, Supabase)
  models/           SQLAlchemy ORM models
  schemas/          Pydantic request/response schemas
  services/
    ml/             Machine learning model loading and inference
    entities/       Entity extraction and management
    fingerprinting/ Document fingerprinting for duplicate detection
    graph/          Supply chain graph analysis
    ingestion/      Data ingestion and processing
    langgraph/      AI investigation workflows
    rag/            Retrieval-augmented generation system
    reporting/      Report generation
    simulator/      Fraud scenario simulation
    verification/   Transaction verification logic
  utils/            Helper functions and utilities

frontend/
  src/
    components/     Reusable React components
    pages/          Application pages
    routes/         Route definitions
    services/       API client and utilities
    context/        React context for state management
  package.json      Node.js dependencies and scripts
  vite.config.js    Vite build configuration

docker/             Docker configuration files
scripts/            Development and deployment scripts
```

## Prerequisites

Before running the system locally, ensure you have the following installed:

- Python 3.11 or higher
- Node.js 16 or higher and npm
- Git

## Installation and Setup

### Step 1: Clone the Repository

```bash
git clone https://github.com/Atharsh2910/SCFinShield-AI.git
cd SCFinShield-AI
```

### Step 2: Set Up Backend Environment

Navigate to the project root and create a Python virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

### Step 3: Configure Environment Variables

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Edit the `.env` file and add your credentials:

```
# Supabase
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key

# Neo4j
NEO4J_URI=bolt://your_neo4j_instance:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=your_password

# Pinecone
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=your_index_name

# API Keys
GROQ_API_KEY=your_groq_api_key

# Feast (Feature Store)
FEAST_REPO_PATH=backend/services/ml/model_registry

# Frontend URL
VITE_API_BASE_URL=http://localhost:8000

# Risk Thresholds
RISK_THRESHOLD_REVIEW=0.30
RISK_THRESHOLD_HOLD=0.70
```

### Step 4: Install Backend Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Set Up Frontend Environment

Navigate to the frontend directory:

```bash
cd frontend
npm install
```

Create a `.env` file in the frontend directory:

```bash
cp .env.example .env
```

Edit `frontend/.env` to set the API base URL:

```
VITE_API_BASE_URL=http://localhost:8000
```

### Step 6: Run the Backend Server

From the project root directory (with virtual environment activated):

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

The backend API will be available at `http://localhost:8000`

The API documentation is accessible at `http://localhost:8000/docs`

### Step 7: Run the Frontend Application

In a new terminal, navigate to the frontend directory:

```bash
cd frontend
npm run dev
```

The frontend will be available at `http://localhost:5173`

## Running in Production

To run the application in production, use Render deployment configuration:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

The render.yaml file is already configured for deployment with:
- Python 3.11 runtime
- Automatic dependency installation
- Environment variable management

## API Endpoints

The system provides the following main endpoint groups:

- **Health**: System health checks
- **Invoices**: Invoice management and processing
- **Fraud Detection**: Fraud analysis and risk scoring
- **Graph**: Supply chain network analysis
- **Investigation**: AI-powered fraud investigations
- **Dashboard**: Aggregated metrics and reporting
- **Simulator**: Fraud scenario testing
- **Knowledge Base**: Information retrieval and management

Access the interactive API documentation at `/docs` (Swagger UI) or `/redoc` (ReDoc).

## Development Workflow

### Starting Fresh

If you need to start from a clean state:

1. Deactivate and remove the virtual environment:
   ```bash
   deactivate
   rm -rf venv
   ```

2. Clear npm cache and node_modules:
   ```bash
   cd frontend
   rm -rf node_modules package-lock.json
   cd ..
   ```

3. Follow the installation steps again from Step 2.

### Common Issues

**Port Already in Use**: If port 8000 or 5173 is occupied, specify a different port:
```bash
uvicorn backend.main:app --reload --port 8001
npm run dev -- --port 5174
```

**Missing Dependencies**: Ensure all dependencies are installed:
```bash
pip install --upgrade -r requirements.txt
npm install
```

**Environment Variables Not Loading**: Verify the `.env` file is in the correct location and readable. Restart the development server after making changes.

## Configuration

Key configuration settings are managed through environment variables in the `.env` file:

- **Risk Thresholds**: Adjust `RISK_THRESHOLD_REVIEW` and `RISK_THRESHOLD_HOLD` to control fraud detection sensitivity
- **CORS Settings**: Modify allowed origins in the FastAPI configuration
- **Database Connections**: Update connection strings for Supabase, Neo4j, and Pinecone
- **API Keys**: Store and manage third-party API credentials

## Database Setup

The system uses three primary databases:

1. **Supabase (PostgreSQL)**: Stores transaction data, users, and audit logs
2. **Neo4j**: Manages supply chain relationship graphs and entity connections
3. **Pinecone**: Stores vector embeddings for similarity searches

Ensure all three services are accessible and properly configured before starting the application.
