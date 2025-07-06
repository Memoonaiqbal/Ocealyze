import streamlit as st
import netCDF4 as nc
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')
import os

# Set page configuration
st.set_page_config(
    page_title="TideTrace",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with beautiful ocean-themed styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@300;400;500;600;700&family=Dancing+Script:wght@400;500;600;700&display=swap');

/* Enhanced Ocean Color Palette */
:root {
    /* Primary Ocean Colors */
    --deep-ocean: #0B1426;
    --midnight-blue: #1B2951;
    --ocean-blue: #2E4F99;
    --wave-blue: #4A90E2;
    --aqua-blue: #00C4FF;
    --sea-foam: #7FDBDA;
    --coral: #FF6B6B;
    --pearl: #F8FFFE;
    
    /* Gradient Colors */
    --ocean-gradient: linear-gradient(135deg, #0B1426 0%, #1B2951 25%, #2E4F99 50%, #4A90E2 75%, #00C4FF 100%);
    --wave-gradient: linear-gradient(45deg, #4A90E2, #00C4FF, #7FDBDA);
    --sunset-gradient: linear-gradient(135deg, #FF6B6B, #4A90E2, #00C4FF);
    
    /* Glass Effects */
    --glass-bg: rgba(255, 255, 255, 0.08);
    --glass-border: rgba(255, 255, 255, 0.15);
    --glass-shadow: rgba(0, 0, 0, 0.2);
    --glass-hover: rgba(255, 255, 255, 0.12);
    
    /* Text Colors */
    --text-primary: #F8FFFE;
    --text-secondary: #B8D4F0;
    --text-accent: #00C4FF;
    --text-muted: #7A9CC6;
}

/* Animated Ocean Background */
.stApp {
    background: var(--ocean-gradient);
    font-family: 'Poppins', sans-serif;
    position: relative;
    overflow-x: hidden;
}

.stApp::before {
    content: '';
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: 
        radial-gradient(circle at 20% 80%, rgba(0, 196, 255, 0.1) 0%, transparent 50%),
        radial-gradient(circle at 80% 20%, rgba(127, 219, 218, 0.1) 0%, transparent 50%),
        radial-gradient(circle at 40% 40%, rgba(74, 144, 226, 0.05) 0%, transparent 50%);
    animation: oceanWaves 20s ease-in-out infinite;
    pointer-events: none;
    z-index: -1;
}

@keyframes oceanWaves {
    0%, 100% { 
        transform: translateY(0px) rotate(0deg);
        opacity: 0.7;
    }
    33% { 
        transform: translateY(-10px) rotate(1deg);
        opacity: 0.9;
    }
    66% { 
        transform: translateY(5px) rotate(-1deg);
        opacity: 0.8;
    }
}

/* Enhanced Main Container */
.main .block-container {
    padding-top: 6rem !important;
    margin-top: 0 !important;
    padding-bottom: 2rem;
    max-width: 1400px;
    backdrop-filter: blur(10px);
}

/* Hide Streamlit Elements */
.stApp > header { display: none !important; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.stDeployButton { display: none; }
.stDecoration { display: none; }

/* Enhanced TideTrace Navigation */
.tidetrace-navbar {
    background: rgba(11, 20, 38, 0.9);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    padding: 1rem 2rem;
    border-bottom: 1px solid var(--glass-border);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    z-index: 1000;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
}

.tidetrace-navbar.hidden {
    transform: translateY(-100%);
}

.tidetrace-title {
    font-family: 'Dancing Script', cursive;
    font-size: 4rem;
    font-weight: 700;
    background: linear-gradient(45deg, #00C4FF, #7FDBDA, #4A90E2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin: 0;
    letter-spacing: 3px;
    text-shadow: 0 0 30px rgba(0, 196, 255, 0.5);
    animation: titleGlow 3s ease-in-out infinite alternate;
    position: relative;
}

.tidetrace-title::after {
    content: '';
    position: absolute;
    bottom: -10px;
    left: 50%;
    transform: translateX(-50%);
    width: 100px;
    height: 3px;
    background: var(--wave-gradient);
    border-radius: 2px;
    animation: waveFlow 2s ease-in-out infinite;
}

@keyframes titleGlow {
    from {
        filter: drop-shadow(0 0 20px rgba(0, 196, 255, 0.7));
    }
    to {
        filter: drop-shadow(0 0 40px rgba(127, 219, 218, 0.9));
    }
}

@keyframes waveFlow {
    0%, 100% { width: 100px; }
    50% { width: 150px; }
}

.main {
    padding-top: 120px !important;
}

/* Enhanced Section Headers */
.sub-header {
    font-size: 2rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 2rem 0 1.5rem;
    padding: 1.5rem 2rem;
    background: var(--glass-bg);
    backdrop-filter: blur(15px);
    -webkit-backdrop-filter: blur(15px);
    border-radius: 20px;
    border: 1px solid var(--glass-border);
    box-shadow: 0 8px 32px var(--glass-shadow);
    position: relative;
    overflow: hidden;
    animation: slideInUp 0.8s cubic-bezier(0.4, 0, 0.2, 1);
}

.sub-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: var(--wave-gradient);
    animation: shimmer 2s ease-in-out infinite;
}

@keyframes shimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

@keyframes slideInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Enhanced Card Containers */
.card-container {
    background: var(--glass-bg);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 24px;
    border: 1px solid var(--glass-border);
    box-shadow: 
        0 8px 32px rgba(0, 0, 0, 0.2),
        inset 0 1px 0 rgba(255, 255, 255, 0.1);
    padding: 2rem;
    margin-bottom: 2rem;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
    animation: fadeInScale 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}

.card-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 1px;
    background: var(--wave-gradient);
}

.card-container:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: 
        0 20px 60px rgba(0, 0, 0, 0.3),
        0 0 40px rgba(0, 196, 255, 0.1),
        inset 0 1px 0 rgba(255, 255, 255, 0.2);
    background: var(--glass-hover);
}

@keyframes fadeInScale {
    from {
        opacity: 0;
        transform: scale(0.9);
    }
    to {
        opacity: 1;
        transform: scale(1);
    }
}

/* Enhanced Metrics */
.stMetric {
    background: var(--glass-bg);
    backdrop-filter: blur(15px);
    -webkit-backdrop-filter: blur(15px);
    border-radius: 16px;
    padding: 1.5rem;
    border: 1px solid var(--glass-border);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
    position: relative;
    overflow: hidden;
}

.stMetric::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 100%;
    height: 2px;
    background: var(--wave-gradient);
    transform: translateX(-100%);
    animation: loadingBar 2s ease-in-out infinite;
}

.stMetric:hover {
    transform: translateY(-4px);
    box-shadow: 0 12px 40px rgba(0, 196, 255, 0.2);
    background: var(--glass-hover);
}

@keyframes loadingBar {
    0% { transform: translateX(-100%); }
    50% { transform: translateX(0%); }
    100% { transform: translateX(100%); }
}

/* Enhanced Sidebar */
[data-testid="stSidebar"] {
    background: rgba(11, 20, 38, 0.95);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-right: 1px solid var(--glass-border);
    box-shadow: 4px 0 20px rgba(0, 0, 0, 0.3);
}

.sidebar-header {
    background: var(--glass-bg);
    backdrop-filter: blur(15px);
    -webkit-backdrop-filter: blur(15px);
    padding: 1rem;
    border-radius: 16px;
    margin: 1rem 0;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    border: 1px solid var(--glass-border);
    position: relative;
    overflow: hidden;
}

.sidebar-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: var(--wave-gradient);
}

.sidebar-header h2 {
    color: var(--text-primary);
    margin: 0;
    font-size: 1.5rem;
    font-weight: 600;
}

/* Enhanced Buttons */
[data-testid="stSidebar"] .stButton > button,
.stButton > button {
    background: var(--glass-bg);
    color: var(--text-primary);
    border: 1px solid var(--glass-border);
    border-radius: 12px;
    padding: 0.8rem 1.2rem;
    font-weight: 500;
    font-size: 0.95rem;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    width: 100%;
    margin-bottom: 0.5rem;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
}

[data-testid="stSidebar"] .stButton > button::before,
.stButton > button::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
    transition: left 0.5s;
}

[data-testid="stSidebar"] .stButton > button:hover,
.stButton > button:hover {
    background: var(--glass-hover);
    color: var(--text-accent);
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(0, 196, 255, 0.2);
    border-color: var(--text-accent);
}

[data-testid="stSidebar"] .stButton > button:hover::before,
.stButton > button:hover::before {
    left: 100%;
}

[data-testid="stSidebar"] .stButton > button:active,
.stButton > button:active {
    transform: translateY(0px);
    box-shadow: 0 4px 15px rgba(0, 196, 255, 0.3);
}

/* Enhanced Chart Containers */
.plotly-chart-container {
    background: var(--glass-bg);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 20px;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    padding: 1.5rem;
    margin: 1.5rem 0;
    border: 1px solid var(--glass-border);
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
}

.plotly-chart-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: var(--wave-gradient);
}

.plotly-chart-container:hover {
    box-shadow: 0 16px 50px rgba(0, 0, 0, 0.3);
    transform: translateY(-4px);
}

/* Enhanced DataFrames */
.stDataFrame {
    border-radius: 16px;
    overflow: hidden;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
    background: var(--glass-bg);
    backdrop-filter: blur(15px);
    -webkit-backdrop-filter: blur(15px);
    border: 1px solid var(--glass-border);
}

/* Enhanced Messages */
.stSuccess {
    background: rgba(127, 219, 218, 0.1);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-left: 4px solid #7FDBDA;
    border-radius: 12px;
    color: var(--text-primary);
}

.stError {
    background: rgba(255, 107, 107, 0.1);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-left: 4px solid #FF6B6B;
    border-radius: 12px;
    color: var(--text-primary);
}

.stWarning {
    background: rgba(255, 193, 7, 0.1);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-left: 4px solid #FFC107;
    border-radius: 12px;
    color: var(--text-primary);
}

.stInfo {
    background: var(--glass-bg);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-left: 4px solid var(--text-accent);
    border-radius: 12px;
    color: var(--text-primary);
}

/* Enhanced Input Elements */
.stSelectbox > div > div,
.stSlider > div,
.stNumberInput > div {
    background: var(--glass-bg);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border-radius: 12px;
    border: 1px solid var(--glass-border);
}

/* Floating Elements Animation */
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
}

.floating {
    animation: float 3s ease-in-out infinite;
}

/* Pulse Animation for Interactive Elements */
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(0, 196, 255, 0.4); }
    70% { box-shadow: 0 0 0 10px rgba(0, 196, 255, 0); }
    100% { box-shadow: 0 0 0 0 rgba(0, 196, 255, 0); }
}

.pulse {
    animation: pulse 2s infinite;
}

/* Responsive Design */
@media (max-width: 768px) {
    .tidetrace-title {
        font-size: 2.5rem;
    }
    
    .card-container {
        padding: 1.5rem;
        margin-bottom: 1.5rem;
    }
    
    .sub-header {
        font-size: 1.5rem;
        padding: 1rem 1.5rem;
    }
}

/* Custom Scrollbar */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(11, 20, 38, 0.3);
}

::-webkit-scrollbar-thumb {
    background: var(--wave-gradient);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(45deg, #00C4FF, #7FDBDA);
}

/* Text Styling */
h1, h2, h3, h4, h5, h6 {
    color: var(--text-primary) !important;
}

p, span, div {
    color: var(--text-secondary) !important;
}

.metric-label {
    color: var(--text-muted) !important;
}

.metric-value {
    color: var(--text-primary) !important;
    font-weight: 600 !important;
}
</style>

<script>
// Enhanced navbar scroll behavior
document.addEventListener('DOMContentLoaded', function() {
    let lastScrollTop = 0;
    let ticking = false;
    
    function updateNavbar() {
        const navbar = document.querySelector('.tidetrace-navbar');
        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
        
        if (scrollTop > lastScrollTop && scrollTop > 100) {
            if (navbar) {
                navbar.classList.add('hidden');
            }
        } else {
            if (navbar) {
                navbar.classList.remove('hidden');
            }
        }
        
        lastScrollTop = scrollTop;
        ticking = false;
    }
    
    function requestTick() {
        if (!ticking) {
            requestAnimationFrame(updateNavbar);
            ticking = true;
        }
    }
    
    window.addEventListener('scroll', requestTick);
    
    // Add floating animation to cards
    setTimeout(() => {
        const cards = document.querySelectorAll('.card-container');
        cards.forEach((card, index) => {
            setTimeout(() => {
                card.style.animationDelay = `${index * 0.1}s`;
                card.classList.add('floating');
            }, index * 100);
        });
    }, 1000);
});
</script>
""", unsafe_allow_html=True)

# Enhanced Plotly theme with ocean colors
custom_theme = {
    "layout": {
        "plot_bgcolor": "rgba(255, 255, 255, 0.05)",
        "paper_bgcolor": "rgba(255, 255, 255, 0.05)",
        "font": {"family": "Poppins, sans-serif", "color": "#F8FFFE", "size": 12},
        "xaxis": {
            "gridcolor": "rgba(255, 255, 255, 0.1)",
            "linecolor": "rgba(255, 255, 255, 0.2)",
            "tickcolor": "rgba(255, 255, 255, 0.2)",
            "titlefont": {"color": "#F8FFFE"}
        },
        "yaxis": {
            "gridcolor": "rgba(255, 255, 255, 0.1)",
            "linecolor": "rgba(255, 255, 255, 0.2)",
            "tickcolor": "rgba(255, 255, 255, 0.2)",
            "titlefont": {"color": "#F8FFFE"}
        },
        "colorway": ["#00C4FF", "#7FDBDA", "#4A90E2", "#FF6B6B", "#2E4F99", "#1B2951"]
    }
}

px.defaults.template = custom_theme

class StreamlitWODAnalyzer:
    def __init__(self):
        self.default_file_path = r"C:\Users\Memoona\Desktop\WOD\WOD1.nc"
        self.parameters = {
            'Temperature': {'variable': 'Temperature', 'unit': '°C', 'color': '#00C4FF', 'icon': '🌡️'},
            'Salinity': {'variable': 'Salinity', 'unit': 'PSU', 'color': '#7FDBDA', 'icon': '🧂'},
            'Oxygen': {'variable': 'Oxygen', 'unit': 'µmol/kg', 'color': '#4A90E2', 'icon': '💨'}
        }

    @st.cache_resource
    def load_data(_self, file_path: str) -> nc.Dataset:
        try:
            if not os.path.exists(file_path):
                st.error(f"📁 File not found at: {file_path}. Please verify the file path.")
                return None
            dataset = nc.Dataset(file_path, 'r')
            return dataset
        except FileNotFoundError:
            st.error(f"❌ FileNotFoundError: The file {file_path} does not exist.")
            return None
        except OSError as e:
            st.error(f"❌ OSError: Failed to open {file_path}. It may be corrupted or not a valid NetCDF file.")
            st.error(f"Error details: {str(e)}")
            return None
        except Exception as e:
            st.error(f"❌ Unexpected error loading NetCDF file {file_path}: {str(e)}")
            return None

    @st.cache_data
    def get_basic_metadata(_self, _dataset: nc.Dataset) -> dict:
        if _dataset is None:
            return None
        try:
            metadata = {
                'total_casts': len(_dataset.dimensions.get('casts', [])),
                'total_temperature_obs': len(_dataset.dimensions.get('Temperature_obs', [])),
                'total_salinity_obs': len(_dataset.dimensions.get('Salinity_obs', [])),
                'total_oxygen_obs': len(_dataset.dimensions.get('Oxygen_obs', [])),
                'depth_obs': len(_dataset.dimensions.get('z_obs', []))
            }

            lat = _dataset.variables.get('lat', None)
            lon = _dataset.variables.get('lon', None)
            if lat is None or lon is None:
                metadata['lat_range'] = (None, None)
                metadata['lon_range'] = (None, None)
            else:
                lat_valid = lat[lat != lat._FillValue] if hasattr(lat, '_FillValue') else lat[:]
                lon_valid = lon[lon != lon._FillValue] if hasattr(lon, '_FillValue') else lon[:]
                metadata['lat_range'] = (float(lat_valid.min()), float(lat_valid.max())) if len(lat_valid) > 0 else (None, None)
                metadata['lon_range'] = (float(lon_valid.min()), float(lon_valid.max())) if len(lon_valid) > 0 else (None, None)

            time_var = _dataset.variables.get('time', None)
            if time_var is None:
                metadata['time_range'] = (None, None)
            else:
                time_vals = time_var[:]
                time_valid = time_vals[time_vals != time_var._FillValue] if hasattr(time_var, '_FillValue') else time_vals
                base_date = datetime(1770, 1, 1)
                min_date = base_date + timedelta(days=float(time_valid.min())) if len(time_valid) > 0 else None
                max_date = base_date + timedelta(days=float(time_valid.max())) if len(time_valid) > 0 else None
                metadata['time_range'] = (min_date, max_date)

            return metadata
        except Exception as e:
            st.error(f"❌ Error extracting metadata: {e}")
            return None

    @st.cache_data
    def get_valid_data(_self, _dataset: nc.Dataset, variable: str, sample_size: int = None) -> np.ndarray:
        try:
            data = _dataset.variables.get(variable, None)
            if data is None:
                st.warning(f"⚠️ Variable '{variable}' not found in dataset.")
                return np.array([])
            if sample_size:
                indices = np.random.choice(len(data), min(sample_size, len(data)), replace=False)
                data = data[indices]
            valid_data = data[data != data._FillValue] if hasattr(data, '_FillValue') else data[:]
            return valid_data.astype(np.float32)
        except Exception as e:
            st.error(f"❌ Error extracting data for {variable}: {e}")
            return np.array([])

    def display_header(self):
        st.markdown(
            """
            <div class="tidetrace-navbar">
                <h1 class="tidetrace-title">TideTrace</h1>
            </div>
            """,
            unsafe_allow_html=True
        )

    def display_overview_metrics(self, metadata: dict):
        st.markdown('<h2 class="sub-header">🌊 Dataset Overview & Ocean Statistics</h2>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="card-container">', unsafe_allow_html=True)
            
            # Main metrics in a grid
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(label="🎯 Total Casts", value=f"{metadata['total_casts']:,}")
            with col2:
                st.metric(label="🌡️ Temperature Obs", value=f"{metadata['total_temperature_obs']:,}")
            with col3:
                st.metric(label="🧂 Salinity Obs", value=f"{metadata['total_salinity_obs']:,}")
            with col4:
                st.metric(label="💨 Oxygen Obs", value=f"{metadata['total_oxygen_obs']:,}")

            # Secondary metrics
            col5, col6, col7 = st.columns(3)
            with col5:
                st.metric(label="📏 Depth Observations", value=f"{metadata['depth_obs']:,}")
            with col6:
                lat_min, lat_max = metadata['lat_range']
                value = f"{lat_min:.1f}° to {lat_max:.1f}°" if lat_min is not None else "N/A"
                st.metric(label="🌍 Latitude Range", value=value)
            with col7:
                lon_min, lon_max = metadata['lon_range']
                value = f"{lon_min:.1f}° to {lon_max:.1f}°" if lon_min is not None else "N/A"
                st.metric(label="🌎 Longitude Range", value=value)

            # Temporal coverage
            min_date, max_date = metadata['time_range']
            if min_date and max_date:
                st.info(f"⏰ **Temporal Coverage:** {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
            else:
                st.info("⏰ **Temporal Coverage:** Data not available")
            
            st.markdown('</div>', unsafe_allow_html=True)

    def create_geographic_map(self, data_cache: dict, sample_size: int = 1000):
        st.markdown('<h2 class="sub-header">🗺️ Global Ocean Measurement Distribution</h2>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="card-container">', unsafe_allow_html=True)
            
            sample_size = st.slider("🎛️ Sample Size for Map", 500, 5000, sample_size, step=500)
            lat = data_cache['lat'][:sample_size]
            lon = data_cache['lon'][:sample_size]
            
            if len(lat) == 0 or len(lon) == 0:
                st.warning("⚠️ No valid geographic data available.")
                st.markdown('</div>', unsafe_allow_html=True)
                return

            fig = px.scatter_mapbox(
                lat=lat,
                lon=lon,
                zoom=1,
                height=600,
                title=f"🌍 Global Distribution of {len(lat):,} Measurement Locations",
                color_discrete_sequence=['#00C4FF']
            )
            
            fig.update_layout(
                mapbox_style="open-street-map",
                margin={"r":0, "t":60, "l":0, "b":0},
                showlegend=False,
                title_font_size=18,
                title_font_color="#F8FFFE",
                plot_bgcolor="rgba(255, 255, 255, 0.05)",
                paper_bgcolor="rgba(255, 255, 255, 0.05)"
            )
            
            st.markdown('<div class="plotly-chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.success(f"📍 Successfully visualized {len(lat):,} measurement locations across global oceans")
            st.markdown('</div>', unsafe_allow_html=True)

    def create_depth_analysis(self, data_cache: dict):
        st.markdown('<h2 class="sub-header">🌊 Ocean Depth Analysis</h2>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="card-container">', unsafe_allow_html=True)
            
            depths = data_cache['z']
            if len(depths) == 0:
                st.warning("⚠️ No valid depth data available.")
                st.markdown('</div>', unsafe_allow_html=True)
                return

            valid_depths = depths[depths > 0]
            if len(valid_depths) == 0:
                st.warning("⚠️ No positive depth data available.")
                st.markdown('</div>', unsafe_allow_html=True)
                return

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📈 Statistical Summary")
                stats_df = pd.DataFrame({
                    'Statistic': ['Minimum Depth', 'Maximum Depth', 'Mean Depth', 'Median Depth', 'Standard Deviation'],
                    'Value (meters)': [
                        f"{valid_depths.min():.1f}",
                        f"{valid_depths.max():.1f}",
                        f"{valid_depths.mean():.1f}",
                        f"{np.median(valid_depths):.1f}",
                        f"{valid_depths.std():.1f}"
                    ]
                })
                st.dataframe(stats_df, use_container_width=True)

            with col2:
                st.markdown("#### 🌊 Depth Zone Distribution")
                ranges = [
                    ("🏖️ Surface (0-50m)", np.sum((valid_depths >= 0) & (valid_depths <= 50))),
                    ("🐠 Shallow (50-200m)", np.sum((valid_depths > 50) & (valid_depths <= 200))),
                    ("🐙 Mid-depth (200-1000m)", np.sum((valid_depths > 200) & (valid_depths <= 1000))),
                    ("🦑 Deep (1000-4000m)", np.sum((valid_depths > 1000) & (valid_depths <= 4000))),
                    ("🕳️ Abyssal (>4000m)", np.sum(valid_depths > 4000))
                ]
                range_df = pd.DataFrame(ranges, columns=['Depth Zone', 'Count'])
                range_df['Percentage'] = (range_df['Count'] / len(valid_depths) * 100).round(2)
                st.dataframe(range_df, use_container_width=True)

            fig = px.histogram(
                x=valid_depths[:2000],
                nbins=50,
                title="📊 Ocean Depth Distribution",
                labels={'x': 'Depth (meters)', 'y': 'Frequency'},
                color_discrete_sequence=['#7FDBDA']
            )
            fig.update_layout(
                height=450, 
                title_font_size=18,
                title_font_color="#F8FFFE"
            )
            
            st.markdown('<div class="plotly-chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    def create_parameter_analysis(self, data_cache: dict):
        st.markdown('<h2 class="sub-header">🔬 Oceanographic Parameter Analysis</h2>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="card-container">', unsafe_allow_html=True)
            
            parameter = st.selectbox("🎯 Select Parameter for Analysis:", list(self.parameters.keys()))
            param_info = self.parameters[parameter]
            data = data_cache[param_info['variable']]
            
            if len(data) == 0:
                st.warning(f"⚠️ No valid data available for {parameter}.")
                st.markdown('</div>', unsafe_allow_html=True)
                return

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"#### {param_info['icon']} {parameter} Statistics")
                stats_data = {
                    'Total Measurements': f"{len(data):,}",
                    'Mean Value': f"{data.mean():.3f} {param_info['unit']}",
                    'Standard Deviation': f"{data.std():.3f} {param_info['unit']}",
                    'Minimum Value': f"{data.min():.3f} {param_info['unit']}",
                    'Maximum Value': f"{data.max():.3f} {param_info['unit']}",
                    'Median Value': f"{np.median(data):.3f} {param_info['unit']}"
                }
                for key, value in stats_data.items():
                    st.metric(label=key, value=value)

            with col2:
                fig = px.histogram(
                    x=data[:2000],
                    nbins=60,
                    title=f"{param_info['icon']} {parameter} Distribution",
                    labels={'x': f"{parameter} ({param_info['unit']})", 'y': 'Frequency'},
                    color_discrete_sequence=[param_info['color']]
                )
                fig.update_layout(height=450, showlegend=False, title_font_size=16)
                st.markdown('<div class="plotly-chart-container">', unsafe_allow_html=True)
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

    def create_depth_profiles(self, data_cache: dict):
        st.markdown('<h2 class="sub-header">📈 Vertical Depth Profiles</h2>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="card-container">', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                max_depth = st.slider("🌊 Maximum Depth (meters)", 0, 6000, 2000, step=100)
            with col2:
                sample_size = st.slider("📊 Sample Size", 500, 5000, 1000, step=500)
            with col3:
                parameter = st.selectbox("🎯 Parameter", list(self.parameters.keys()))

            param_info = self.parameters[parameter]
            depths = data_cache['z']
            param_data = data_cache[param_info['variable']]
            
            if len(depths) == 0 or len(param_data) == 0:
                st.warning(f"⚠️ No valid data for {parameter} or depth measurements.")
                st.markdown('</div>', unsafe_allow_html=True)
                return

            valid_mask = (depths <= max_depth) & (depths > 0)
            param_valid = param_data[valid_mask]
            depth_valid = depths[valid_mask]
            
            if len(param_valid) == 0 or len(depth_valid) == 0:
                st.warning("⚠️ No valid data points after applying depth filter.")
                st.markdown('</div>', unsafe_allow_html=True)
                return

            if len(param_valid) > sample_size:
                indices = np.random.choice(len(param_valid), sample_size, replace=False)
                param_sample = param_valid[indices]
                depth_sample = depth_valid[indices]
            else:
                param_sample = param_valid
                depth_sample = depth_valid

            fig = px.scatter(
                x=param_sample,
                y=depth_sample,
                title=f"{param_info['icon']} {parameter} vs Ocean Depth Profile",
                labels={'x': f"{parameter} ({param_info['unit']})", 'y': 'Depth (meters)'},
                opacity=0.7,
                color_discrete_sequence=[param_info['color']]
            )
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(height=550, showlegend=False, title_font_size=16)
            
            st.markdown('<div class="plotly-chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            st.success(f"📊 Displaying {len(param_sample):,} data points (maximum depth: {max_depth}m)")
            st.markdown('</div>', unsafe_allow_html=True)

    def create_temporal_analysis(self, data_cache: dict):
        st.markdown('<h2 class="sub-header">📅 Temporal Distribution Analysis</h2>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="card-container">', unsafe_allow_html=True)
            
            time_vals = data_cache['time']
            if len(time_vals) == 0:
                st.warning("⚠️ No valid temporal data available.")
                st.markdown('</div>', unsafe_allow_html=True)
                return

            base_date = datetime(1770, 1, 1)
            years = [(base_date + timedelta(days=float(t))).year for t in time_vals]
            year_counts = pd.Series(years).value_counts().sort_index()

            fig = px.line(
                x=year_counts.index,
                y=year_counts.values,
                title="📈 Oceanographic Measurements Timeline",
                labels={'x': 'Year', 'y': 'Number of Measurements'},
                color_discrete_sequence=['#4A90E2']
            )
            fig.update_layout(height=450, showlegend=False, title_font_size=16)
            fig.update_traces(line_width=3)
            
            st.markdown('<div class="plotly-chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            decades = [year // 10 * 10 for year in years]
            decade_counts = pd.Series(decades).value_counts().sort_index()
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### 📊 Measurements by Decade")
                decade_df = pd.DataFrame({
                    'Decade': [f"{d}s" for d in decade_counts.index],
                    'Count': decade_counts.values,
                    'Percentage': (decade_counts.values / len(years) * 100).round(2)
                })
                st.dataframe(decade_df, use_container_width=True)

            with col2:
                fig_pie = px.pie(
                    values=decade_counts.values,
                    names=[f"{d}s" for d in decade_counts.index],
                    title="🥧 Temporal Distribution by Decade",
                    color_discrete_sequence=['#00C4FF', '#7FDBDA', '#4A90E2', '#FF6B6B', '#2E4F99', '#1B2951']
                )
                fig_pie.update_layout(height=450, showlegend=True, title_font_size=16)
                st.markdown('<div class="plotly-chart-container">', unsafe_allow_html=True)
                st.plotly_chart(fig_pie, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            st.markdown('</div>', unsafe_allow_html=True)

    def create_kmeans_clustering(self, data_cache: dict):
        st.markdown('<h2 class="sub-header">🔄 K-Means Clustering Analysis</h2>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="card-container">', unsafe_allow_html=True)
            
            n_clusters = st.slider("🎛️ Number of Clusters", 2, 10, 4, step=1)
            sample_size = st.slider("📊 Sample Size for Clustering", 500, 5000, 1000, step=500)

            temp = data_cache['Temperature']
            sal = data_cache['Salinity']
            oxy = data_cache['Oxygen']
            
            if len(temp) == 0 or len(sal) == 0 or len(oxy) == 0:
                st.warning("⚠️ Insufficient data for clustering analysis.")
                st.markdown('</div>', unsafe_allow_html=True)
                return

            common_indices = np.intersect1d(np.intersect1d(np.arange(len(temp)), np.arange(len(sal))), np.arange(len(oxy)))
            if len(common_indices) < n_clusters:
                st.warning("⚠️ Not enough data points for clustering.")
                st.markdown('</div>', unsafe_allow_html=True)
                return

            if len(common_indices) > sample_size:
                sample_indices = np.random.choice(common_indices, sample_size, replace=False)
            else:
                sample_indices = common_indices

            data = np.vstack((temp[sample_indices], sal[sample_indices], oxy[sample_indices])).T
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            labels = kmeans.fit_predict(data_scaled)

            colors = ['#00C4FF', '#7FDBDA', '#4A90E2', '#FF6B6B', '#2E4F99', '#1B2951']
            color_map = {str(i): colors[i % len(colors)] for i in range(n_clusters)}

            fig = px.scatter_3d(
                x=data[:, 0], y=data[:, 1], z=data[:, 2],
                color=labels.astype(str),
                labels={'x': 'Temperature (°C)', 'y': 'Salinity (PSU)', 'z': 'Oxygen (µmol/kg)'},
                title=f"🔄 3D K-Means Clustering (n={n_clusters} clusters)",
                opacity=0.8,
                color_discrete_map=color_map
            )
            fig.update_layout(height=600, title_font_size=16)
            
            st.markdown('<div class="plotly-chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            cluster_stats = pd.DataFrame({
                'Cluster': range(n_clusters),
                'Count': [np.sum(labels == i) for i in range(n_clusters)],
                'Mean Temperature (°C)': [data[labels == i, 0].mean().round(3) for i in range(n_clusters)],
                'Mean Salinity (PSU)': [data[labels == i, 1].mean().round(3) for i in range(n_clusters)],
                'Mean Oxygen (µmol/kg)': [data[labels == i, 2].mean().round(3) for i in range(n_clusters)]
            })
            
            st.markdown("#### 📊 Cluster Statistics Summary")
            st.dataframe(cluster_stats, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

    def create_decision_tree_classification(self, data_cache: dict):
        st.markdown('<h2 class="sub-header">🌳 Water Mass Classification</h2>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="card-container">', unsafe_allow_html=True)
            
            sample_size = st.slider("📊 Sample Size for Classification", 500, 5000, 1000, step=500)

            temp = data_cache['Temperature']
            sal = data_cache['Salinity']
            
            if len(temp) == 0 or len(sal) == 0:
                st.warning("⚠️ No valid temperature or salinity data.")
                st.markdown('</div>', unsafe_allow_html=True)
                return

            common_indices = np.intersect1d(np.arange(len(temp)), np.arange(len(sal)))
            if len(common_indices) == 0:
                st.warning("⚠️ No overlapping temperature and salinity data.")
                st.markdown('</div>', unsafe_allow_html=True)
                return

            if len(common_indices) > sample_size:
                sample_indices = np.random.choice(common_indices, sample_size, replace=False)
            else:
                sample_indices = common_indices

            data = np.vstack((temp[sample_indices], sal[sample_indices])).T
            labels = []
            
            for t, s in data:
                if t > 20 and 34 < s < 36:
                    labels.append("🌴 Tropical")
                elif t < 5 and s > 34:
                    labels.append("🧊 Polar")
                elif 5 <= t <= 20 and 33 <= s <= 35:
                    labels.append("🌊 Temperate")
                else:
                    labels.append("🌐 Other")
            
            labels = np.array(labels)
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            
            clf = DecisionTreeClassifier(max_depth=5, random_state=42)
            clf.fit(data_scaled, labels)
            predictions = clf.predict(data_scaled)
            accuracy = accuracy_score(labels, predictions)

            color_map = {
                "🌴 Tropical": "#FF6B6B",
                "🧊 Polar": "#00C4FF",
                "🌊 Temperate": "#4A90E2",
                "🌐 Other": "#7FDBDA"
            }

            fig = px.scatter(
                x=data[:, 0], y=data[:, 1], color=labels,
                labels={'x': 'Temperature (°C)', 'y': 'Salinity (PSU)'},
                title="🌳 Water Mass Classification Results",
                opacity=0.8,
                color_discrete_map=color_map
            )
            fig.update_layout(height=550, title_font_size=16)
            
            st.markdown('<div class="plotly-chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            class_counts = pd.Series(labels).value_counts()
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Classification Statistics")
                stats_df = pd.DataFrame({
                    'Water Mass Type': class_counts.index,
                    'Count': class_counts.values,
                    'Percentage': (class_counts.values / len(labels) * 100).round(2)
                })
                st.dataframe(stats_df, use_container_width=True)

            with col2:
                st.markdown("#### 🎯 Model Performance")
                st.metric(label="Classification Accuracy", value=f"{accuracy:.2%}")
                st.success(f"🎯 The decision tree classifier achieved {accuracy:.2%} accuracy on the dataset.")
            
            st.markdown('</div>', unsafe_allow_html=True)

    def create_time_series_analysis(self, data_cache: dict):
        st.markdown('<h2 class="sub-header">📈 Temperature Time Series Analysis</h2>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="card-container">', unsafe_allow_html=True)
            
            time_vals = data_cache['time']
            temp = data_cache['Temperature']
            
            if len(time_vals) == 0 or len(temp) == 0:
                st.warning("⚠️ No valid time or temperature data.")
                st.markdown('</div>', unsafe_allow_html=True)
                return

            common_indices = np.intersect1d(np.arange(len(time_vals)), np.arange(len(temp)))
            if len(common_indices) == 0:
                st.warning("⚠️ No overlapping time and temperature data.")
                st.markdown('</div>', unsafe_allow_html=True)
                return

            time_vals = time_vals[common_indices]
            temp = temp[common_indices]

            base_date = datetime(1770, 1, 1)
            years = [(base_date + timedelta(days=float(t))).year for t in time_vals]
            yearly_means = pd.DataFrame({'Year': years, 'Temperature': temp}).groupby('Year').mean().reset_index()

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=yearly_means['Year'],
                y=yearly_means['Temperature'],
                mode='lines+markers',
                name='Mean Temperature',
                line=dict(color='#00C4FF', width=3),
                marker=dict(size=6)
            ))

            z = np.polyfit(yearly_means['Year'], yearly_means['Temperature'], 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=yearly_means['Year'],
                y=p(yearly_means['Year']),
                mode='lines',
                name='Trend Line',
                line=dict(color='#FF6B6B', dash='dash', width=3)
            ))

            fig.update_layout(
                title="📈 Long-term Temperature Trend Analysis",
                xaxis_title="Year",
                yaxis_title="Temperature (°C)",
                height=500,
                title_font_size=16
            )
            
            st.markdown('<div class="plotly-chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            trend_slope = z[0]
            trend_direction = "📈 Increasing" if trend_slope > 0 else "📉 Decreasing"
            st.info(f"**Temperature Trend:** {trend_direction} (slope: {trend_slope:.4f} °C/year)")
            st.markdown('</div>', unsafe_allow_html=True)

    def create_prediction_section(self, data_cache: dict):
        st.markdown('<h2 class="sub-header">🔮 Parameter Prediction</h2>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="card-container">', unsafe_allow_html=True)
            
            parameter = st.selectbox("🎯 Select Parameter to Predict:", list(self.parameters.keys()))
            param_info = self.parameters[parameter]

            st.markdown("#### 📥 Input Features for Prediction")
            col1, col2 = st.columns(2)
            with col1:
                lat = st.number_input("🌍 Latitude (°)", -90.0, 90.0, 0.0, step=0.1)
                lon = st.number_input("🌎 Longitude (°)", -180.0, 180.0, 0.0, step=0.1)
            with col2:
                depth = st.number_input("📏 Depth (meters)", 0.0, 6000.0, 0.0, step=10.0)
                year = st.number_input("📅 Year", 1900, 2025, 2020, step=1)

            data = {
                'lat': data_cache['lat'],
                'lon': data_cache['lon'],
                'z': data_cache['z'],
                'time': data_cache['time'],
                parameter: data_cache[param_info['variable']]
            }

            df = pd.DataFrame(data)
            df = df.dropna()
            if len(df) < 10:
                st.warning(f"⚠️ Insufficient valid data for {parameter} prediction.")
                st.markdown('</div>', unsafe_allow_html=True)
                return

            base_date = datetime(1770, 1, 1)
            df['year'] = [(base_date + timedelta(days=float(t))).year for t in df['time']]

            X = df[['lat', 'lon', 'z', 'year']]
            y = df[parameter]

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)

            input_data = np.array([[lat, lon, depth, year]])
            prediction = model.predict(input_data)[0]

            tree_predictions = np.array([tree.predict(input_data) for tree in model.estimators_])
            uncertainty = np.std(tree_predictions)

            st.markdown(f"#### 🔮 Prediction Result for {param_info['icon']} {parameter}")
            st.metric(
                label=f"Predicted {parameter}",
                value=f"{prediction:.3f} {param_info['unit']}",
                delta=f"±{uncertainty:.3f} {param_info['unit']} (Uncertainty)"
            )

            st.markdown(f"#### 📊 Prediction in Context")
            sample_size = min(1000, len(df))
            sample_df = df.sample(sample_size, random_state=42)

            fig = px.scatter(
                x=sample_df[parameter],
                y=sample_df['z'],
                title=f"{param_info['icon']} {parameter} vs Depth with Prediction",
                labels={'x': f"{parameter} ({param_info['unit']})", 'y': 'Depth (meters)'},
                opacity=0.6,
                color_discrete_sequence=[param_info['color']]
            )
            fig.add_scatter(
                x=[prediction],
                y=[depth],
                mode='markers',
                marker=dict(size=15, color='#FF6B6B', symbol='star'),
                name='Prediction'
            )
            fig.update_yaxes(autorange="reversed")
            fig.update_layout(height=550, showlegend=True, title_font_size=16)
            
            st.markdown('<div class="plotly-chart-container">', unsafe_allow_html=True)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            st.success(f"✅ Predicted {parameter} for the given inputs with uncertainty estimate.")
            st.markdown('</div>', unsafe_allow_html=True)

    def create_export_section(self, data_cache: dict, metadata: dict):
        st.markdown('<h2 class="sub-header">💾 Data Export & Reporting</h2>', unsafe_allow_html=True)
        
        with st.container():
            st.markdown('<div class="card-container">', unsafe_allow_html=True)
            
            if st.button("📊 Generate Comprehensive Summary Report", type="primary"):
                summary_data = []
                for param, param_info in self.parameters.items():
                    data = data_cache[param_info['variable']]
                    if len(data) == 0:
                        continue
                    summary_data.append({
                        'Parameter': f"{param_info['icon']} {param}",
                        'Unit': param_info['unit'],
                        'Total_Measurements': len(data),
                        'Mean': round(data.mean(), 4),
                        'Standard_Deviation': round(data.std(), 4),
                        'Minimum': round(data.min(), 4),
                        'Maximum': round(data.max(), 4),
                        'Median': round(np.median(data), 4)
                    })

                if not summary_data:
                    st.warning("⚠️ No valid data available for export.")
                    st.markdown('</div>', unsafe_allow_html=True)
                    return

                summary_df = pd.DataFrame(summary_data)
                st.markdown("#### 📋 Comprehensive Statistical Summary")
                st.dataframe(summary_df, use_container_width=True)

                csv = summary_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Summary Report (CSV)",
                    data=csv,
                    file_name="tidetrace_oceanographic_summary.csv",
                    mime="text/csv"
                )

                st.success("✅ Summary report generated successfully!")
            
            st.markdown('</div>', unsafe_allow_html=True)

def main():
    analyzer = StreamlitWODAnalyzer()
    file_path = analyzer.default_file_path
    
    with st.spinner("🌊 Loading and processing oceanographic data..."):
        dataset = analyzer.load_data(file_path)
        if dataset is None:
            st.error("❌ Failed to load oceanographic data. Please verify the file path and try again.")
            st.stop()

    with st.spinner("📊 Extracting metadata and preparing data cache..."):
        metadata = analyzer.get_basic_metadata(dataset)
        if metadata is None:
            st.error("❌ Failed to extract dataset metadata.")
            st.stop()
        
        data_cache = {
            'lat': analyzer.get_valid_data(dataset, 'lat', sample_size=5000),
            'lon': analyzer.get_valid_data(dataset, 'lon', sample_size=5000),
            'z': analyzer.get_valid_data(dataset, 'z', sample_size=5000),
            'time': analyzer.get_valid_data(dataset, 'time', sample_size=5000),
            'Temperature': analyzer.get_valid_data(dataset, 'Temperature', sample_size=5000),
            'Salinity': analyzer.get_valid_data(dataset, 'Salinity', sample_size=5000),
            'Oxygen': analyzer.get_valid_data(dataset, 'Oxygen', sample_size=5000)
        }

    analyzer.display_header()

    with st.sidebar:
        st.markdown('<div class="sidebar-header"><h2>🌊 Ocean Analytics Dashboard</h2></div>', unsafe_allow_html=True)
        
        analysis_options = [
            ("🏠 Overview", "overview"),
            ("🗺️ Geographic Distribution", "geographic"),
            ("🌊 Depth Analysis", "depth"),
            ("🔬 Parameter Analysis", "parameter"),
            ("📈 Depth Profiles", "profiles"),
            ("📅 Temporal Analysis", "temporal"),
            ("🔄 K-Means Clustering", "clustering"),
            ("🌳 Water Mass Classification", "classification"),
            ("📈 Time Series Analysis", "timeseries"),
            ("🔮 Parameter Prediction", "prediction"),
            ("💾 Export Data", "export")
        ]
        
        selected_analysis = st.radio("Select Analysis Type:", analysis_options, format_func=lambda x: x[0])

    if selected_analysis[1] == "overview":
        analyzer.display_overview_metrics(metadata)
    elif selected_analysis[1] == "geographic":
        analyzer.create_geographic_map(data_cache)
    elif selected_analysis[1] == "depth":
        analyzer.create_depth_analysis(data_cache)
    elif selected_analysis[1] == "parameter":
        analyzer.create_parameter_analysis(data_cache)
    elif selected_analysis[1] == "profiles":
        analyzer.create_depth_profiles(data_cache)
    elif selected_analysis[1] == "temporal":
        analyzer.create_temporal_analysis(data_cache)
    elif selected_analysis[1] == "clustering":
        analyzer.create_kmeans_clustering(data_cache)
    elif selected_analysis[1] == "classification":
        analyzer.create_decision_tree_classification(data_cache)
    elif selected_analysis[1] == "timeseries":
        analyzer.create_time_series_analysis(data_cache)
    elif selected_analysis[1] == "prediction":
        analyzer.create_prediction_section(data_cache)
    elif selected_analysis[1] == "export":
        analyzer.create_export_section(data_cache, metadata)

if dataset and hasattr(dataset, 'close'):
        try:
            dataset.close()
        except (OSError, RuntimeError) as e:
            pass
        except Exception as e:
            st.warning(f"⚠️ Note: Dataset cleanup encountered an issue: {str(e)}")

if __name__ == "__main__":
    main()
