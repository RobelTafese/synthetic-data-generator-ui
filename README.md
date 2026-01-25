# Synthetic Data Generator - Streamlit Web Application

Complete web-based UI for the Auto-Detecting Synthetic Data Generator.

## Features

✅ **No Code Required** - User-friendly web interface  
✅ **File Upload** - Drag and drop CSV files  
✅ **Auto-Detection** - Automatically detects all column types  
✅ **Interactive Dashboard** - Real-time quality validation charts  
✅ **Column Selection** - Choose which columns to visualize  
✅ **Statistical Comparison** - Detailed stats comparison  
✅ **Download Results** - Export synthetic data as CSV  

## Supported Data Types

- Continuous Numeric (e.g., salary, price)
- Discrete Numeric (e.g., ratings, counts)
- Categorical (e.g., status, category)
- Boolean (e.g., yes/no, true/false)
- DateTime (e.g., dates, timestamps)

---

## Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Verify Files

Make sure you have these files in the same directory:
```
synthetic_data_app.py
AUTO-DETECTING_SYNTHETIC_DATA_GENERATOR_ENHANCED.py
requirements.txt
```

---

## Running Locally

### Start the Application

```bash
streamlit run synthetic_data_app.py
```

The app will automatically open in your browser at:
```
http://localhost:8501
```

### Manual Access

If it doesn't open automatically, navigate to:
```
http://localhost:8501
```

---

## Using the Application

### Step 1: Upload Data
1. Go to "📁 Upload & Generate" tab
2. Click "Choose a CSV file" or drag and drop
3. Preview your data

### Step 2: Configure Parameters
1. Set number of rows to generate
2. Adjust advanced options (optional)
3. Click "🚀 Generate Synthetic Data"

### Step 3: Review Quality
1. Go to "📊 Quality Dashboard" tab
2. Select columns to visualize
3. Review distribution comparisons

### Step 4: Download Results
1. Go to "⬇️ Download Results" tab
2. Click "📥 Download Synthetic Data (CSV)"
3. Use the synthetic data in your applications

---

## Deployment to Cloud

### Option 1: Streamlit Cloud (Recommended - Free)

1. **Push to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin <your-github-repo-url>
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud:**
   - Go to https://streamlit.io/cloud
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file: `synthetic_data_app.py`
   - Click "Deploy"

3. **Access Your App:**
   - You'll get a URL like: `https://your-app-name.streamlit.app`
   - Share this URL with your team

### Option 2: Docker Deployment

1. **Create Dockerfile:**
   ```dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   
   COPY . .
   
   EXPOSE 8501
   
   CMD ["streamlit", "run", "synthetic_data_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. **Build Docker Image:**
   ```bash
   docker build -t synthetic-data-generator .
   ```

3. **Run Container:**
   ```bash
   docker run -p 8501:8501 synthetic-data-generator
   ```

4. **Deploy to Cloud:**
   - AWS ECS
   - Azure Container Instances
   - Google Cloud Run

### Option 3: AWS EC2 Deployment

1. **Launch EC2 Instance:**
   - Select Ubuntu 22.04 LTS
   - Instance type: t2.medium or higher
   - Open port 8501 in security group

2. **SSH into Instance:**
   ```bash
   ssh -i your-key.pem ubuntu@your-ec2-ip
   ```

3. **Install Dependencies:**
   ```bash
   sudo apt update
   sudo apt install python3-pip -y
   pip3 install streamlit pandas numpy scikit-learn plotly scipy
   ```

4. **Upload Files:**
   ```bash
   scp -i your-key.pem synthetic_data_app.py ubuntu@your-ec2-ip:~/
   scp -i your-key.pem AUTO-DETECTING_SYNTHETIC_DATA_GENERATOR_ENHANCED.py ubuntu@your-ec2-ip:~/
   ```

5. **Run Application:**
   ```bash
   streamlit run synthetic_data_app.py --server.port=8501 --server.address=0.0.0.0
   ```

6. **Access:**
   - Open: `http://your-ec2-ip:8501`

---

## Configuration

### Advanced Settings

Edit these parameters in the UI or modify defaults in code:

```python
# In synthetic_data_app.py, modify defaults:

max_gmm_components = 3        # Maximum Gaussian components
random_seed = 42              # Reproducibility seed
discrete_threshold = 0.05     # Discrete detection threshold
```

### Performance Tuning

For large datasets (>100K rows):
- Increase EC2 instance size
- Use sampling for preview
- Generate in batches

---

## Troubleshooting

### Issue: "Module not found"
**Solution:**
```bash
pip install -r requirements.txt
```

### Issue: "Port 8501 already in use"
**Solution:**
```bash
# Kill existing process
lsof -ti:8501 | xargs kill -9

# Or use different port
streamlit run synthetic_data_app.py --server.port=8502
```

### Issue: "Memory error with large files"
**Solution:**
- Use smaller dataset for testing
- Increase system RAM
- Sample the data before upload

### Issue: "Charts not displaying"
**Solution:**
```bash
pip install --upgrade plotly scipy
```

---

## File Structure

```
project/
│
├── synthetic_data_app.py                              # Main Streamlit app
├── AUTO-DETECTING_SYNTHETIC_DATA_GENERATOR_ENHANCED.py # Generator engine
├── requirements.txt                                    # Python dependencies
├── README.md                                           # This file
│
└── (generated files)
    ├── synthetic_data_YYYYMMDD_HHMMSS.csv             # Output files
    └── column_info_YYYYMMDD_HHMMSS.json               # Metadata
```

---

## Features by Tab

### Tab 1: Upload & Generate
- File upload (CSV)
- Data preview
- Parameter configuration
- Generate button
- Column detection summary

### Tab 2: Quality Dashboard
- Summary statistics
- Column selector
- Distribution plots (KDE)
- Box plots
- Bar charts for categorical
- Real vs Synthetic comparison

### Tab 3: Statistical Comparison
- Numeric column stats
- Categorical distributions
- DateTime ranges
- Mean/Std comparisons

### Tab 4: Download Results
- Synthetic data preview
- CSV download
- JSON metadata download
- File information

---

## API / Programmatic Usage

You can also use the generator programmatically:

```python
from AUTO-DETECTING_SYNTHETIC_DATA_GENERATOR_ENHANCED import generate_synthetic

# Generate synthetic data
synthetic_df = generate_synthetic(
    file_path='my_data.csv',
    num_rows=1000,
    dashboard=True,
    show_stats=True
)

# Use the synthetic data
print(synthetic_df.head())
synthetic_df.to_csv('output.csv', index=False)
```

---

## Security Considerations

### For Production Deployment:

1. **Authentication:**
   - Add login system (Streamlit supports authentication)
   - Use OAuth or SSO integration

2. **File Upload Limits:**
   ```python
   # In synthetic_data_app.py
   st.set_page_config(max_upload_size=200)  # 200 MB limit
   ```

3. **Data Privacy:**
   - Files are processed in memory
   - No data is stored on server
   - Clear session after use

4. **Rate Limiting:**
   - Implement user quotas
   - Limit generation requests per hour

---

## Performance Benchmarks

| Dataset Size | Rows | Columns | Generation Time | RAM Usage |
|--------------|------|---------|-----------------|-----------|
| Small        | 1K   | 10      | 2-5 sec        | 50 MB     |
| Medium       | 10K  | 20      | 10-20 sec      | 200 MB    |
| Large        | 100K | 50      | 1-3 min        | 1 GB      |
| Extra Large  | 1M   | 50      | 10-20 min      | 4 GB      |

---

## Support

For issues, questions, or feature requests:
- Contact: Robel
- Organization: FED USDS ADVS (Aidvantage)
- Email: [your-email]

---

## License

Internal use only - FED USDS ADVS (Aidvantage)

---

## Version History

**Version 1.0** (January 2025)
- Initial release
- Full UI with 4 tabs
- Support for 5 data types
- Interactive dashboard
- Cloud deployment ready

---

## Next Steps

1. ✅ Install dependencies
2. ✅ Run locally to test
3. ✅ Present to internal team
4. ✅ Deploy to cloud
5. ✅ Demo to client

**You're ready to meet Amit's requirements!**
